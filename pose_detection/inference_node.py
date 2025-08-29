#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sensor_msgs
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from geometry_msgs.msg import PointStamped

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
            PointStamped,
            '/pose_detection/bbox_centers',
            10
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

            # Draw results on the image
            annotated_frame = results[0].plot()

            # Convert back to ROS Image
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')

            #ros_image = self.bridge.cv2_to_compressed_imgmsg(annotated_frame)

            # Publish result
            self.publisher.publish(ros_image)


            boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
            if boxes.shape[0] == 0:
                return  # no detections, nothing to publish

            # Compute areas and find largest bbox
            areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
            largest_idx = areas.argmax()
            x1, y1, x2, y2 = boxes[largest_idx]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Publish only the center of the largest bounding box
            center_msg = PointStamped()
            center_msg.header.stamp = self.get_clock().now().to_msg()
            center_msg.point.x = cx
            center_msg.point.y = cy
            center_msg.point.z = float('nan')
            self.center_publisher.publish(center_msg)

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






