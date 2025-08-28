#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sensor_msgs
#from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO

class PoseDetectionNode(Node):
    def __init__(self):
        super().__init__('pose_detection_node')

        # Load YOLOv8 pose model
        self.get_logger().info("Loading YOLOv8 pose model...")
        self.model = YOLO("/home/eerik/checkpoints/yolov8n-pose.pt")

        # CV bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Subscriber: raw camera image
        self.image_subscriber = self.create_subscription(
            sensor_msgs.msg.CompressedImage,
            '/front_camera/image_color/compressed',  # Adjust the topic to your camera input
            self.image_callback,
            5
        )

        # Publisher: annotated image
        self.publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage,
            '/pose_detection/image',
            5
        )


        self.get_logger().info("Pose detection node initialized.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run inference
            results = self.model(cv_image)

            # Draw results on the image
            annotated_frame = results[0].plot()

            # Convert back to ROS Image
            #ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')

            ros_image = self.bridge.cv2_to_compressed_imgmsg(annotated_frame)

            # Publish result
            self.publisher.publish(ros_image)

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






