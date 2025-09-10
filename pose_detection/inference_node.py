#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sensor_msgs
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from  collections import deque
from std_msgs.msg import Int32


def hand_up(kp):
    """
    Check if exactly one hand is above the head for a single person.
    kp: Keypoints object from YOLOv8
    Returns True if exactly one hand is above the head, False otherwise.
    """
    if kp is None or len(kp) == 0:
        return False

    # 1. Extract the underlying data tensor (shape: 1,17,3)
    kp_tensor = kp.data  # torch.Tensor

    # 2. Move to CPU and convert to numpy
    kp_array = kp_tensor#.cpu().numpy()  # shape (1,17,3)

    # 3. Remove batch dimension
    kp_array = kp_array[0]  # now shape (17,3)

    # 4. Keypoints
    nose = kp_array[0][:2]       # Head reference
    l_wrist, r_wrist = kp_array[9][:2], kp_array[10][:2]

    # 5. In images, smaller y = higher
    l_above = l_wrist[1] < nose[1]
    r_above = r_wrist[1] < nose[1]

    # 6. Exactly one hand above head
    return (l_above != r_above).cpu().numpy()



class PoseDetectionNode(Node):
    def __init__(self):
        super().__init__('pose_detection_node')

        self.declare_parameter('image_topic',value='/front_camera/image_color/compressed')
        self.declare_parameter('crop_top',value=630)
        self.declare_parameter('crop_bottom',value=1430)
        self.declare_parameter('crop_left',value=0)
        self.declare_parameter('crop_right',value=2448)
        self.declare_parameter('downscaling_factor',value=2)
        self.declare_parameter('model_path',value='')

        # Get parameter values from the parameter server
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value  
        self.crop_top = self.get_parameter('crop_top').get_parameter_value().integer_value
        self.crop_bottom = self.get_parameter('crop_bottom').get_parameter_value().integer_value
        self.crop_left = self.get_parameter('crop_left').get_parameter_value().integer_value
        self.crop_right = self.get_parameter('crop_right').get_parameter_value().integer_value
        downscaling_factor = self.get_parameter('downscaling_factor').get_parameter_value().integer_value
        model_path=self.get_parameter('model_path').get_parameter_value().string_value

        self.H=(self.crop_bottom-self.crop_top)//downscaling_factor
        self.W=(self.crop_right-self.crop_left)//downscaling_factor

        self.model = YOLO(model_path)


        # Running average buffer (fixed length 10)
        self.hand_history = deque(maxlen=10)

        # CV bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Subscriber: raw camera image
        self.image_subscriber = self.create_subscription(
            sensor_msgs.msg.CompressedImage,
            image_topic,  # Adjust the topic to your camera input
            self.image_callback,
            5
        )

        # Publisher: annotated image
        self.publisher = self.create_publisher(
            sensor_msgs.msg.Image,
            '/pose_detection/image',
            5
        )

        # Publisher: bounding box centers
        self.center_publisher = self.create_publisher(
            Int32,
            '/pose_detection/dist_to_center',
            5
        )

        self.hand_up_publisher = self.create_publisher(
            Bool,
            '/pose_detection/hand_up',
            5
        )


        self.get_logger().info("Pose detection node initialized.")

    def crop_and_scale(self,image):
        image=image[self.crop_top:self.crop_bottom,self.crop_left:self.crop_right]
        image=cv2.resize(image,(self.W,self.H))
        return image
    

    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image=self.crop_and_scale(image)

            # Run inference
            results = self.model(image)

            # Extract bounding boxes
            boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
            if boxes.shape[0] == 0:
                # No detections -> publish plain image
                ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                self.publisher.publish(ros_image)
                return

            # Find largest bounding box
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = areas.argmax()
            x1, y1, x2, y2 = boxes[largest_idx].astype(int)

            # Copy image for drawing
            annotated_frame = image.copy()

            # Draw only the largest bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            final_hand_up = False  # default

            if results[0].keypoints is not None:
                kp = results[0].keypoints[largest_idx]

                # Raw detection
                hand_above = hand_up(kp)

                # Append to history buffer
                self.hand_history.append(hand_above)

                # Compute smoothed value: majority vote
                if len(self.hand_history) > 0:
                    votes = sum(self.hand_history)
                    final_hand_up = votes > (len(self.hand_history) // 2)

                # Draw text if smoothed value is True
                if final_hand_up:
                    cv2.putText(
                        annotated_frame,
                        "System on",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                
                else:
                    cv2.putText(
                        annotated_frame,
                        "System off",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )


                # Publish Bool (smoothed)
                msg = Bool()
                msg.data = bool(final_hand_up)
                self.hand_up_publisher.publish(msg)

            # Publish annotated image
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.publisher.publish(ros_image)

            # Publish bbox center
            cx = float((x1 + x2) // 2.0)
            #cy = float((y1 + y2) / 2.0)
            
            cam_W = self.crop_right-self.crop_left
            dist_to_center = cam_W-cx

            msg = Int32()
            msg.data = int(dist_to_center)
            self.center_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")




def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()






