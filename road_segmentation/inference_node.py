import sys
import torch
import rclpy
from rclpy.node import Node
import sensor_msgs
from cv_bridge import CvBridge
import cv2
import numpy as np
import torchvision.transforms as T
import torchvision.models.segmentation as segmentation
from PIL import Image
import time
from matplotlib.colors import LinearSegmentedColormap


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

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


        #--------Subscriber---------------------------------------------
        self.image_subscriber = self.create_subscription(
            sensor_msgs.msg.CompressedImage,
            image_topic,  # Adjust the topic to your camera input
            self.image_callback,
            5
        )

        #----------Publishers-----------------------------------------
        #road mask
        self.overlaid_road_mask_publisher = self.create_publisher(
            sensor_msgs.msg.Image,
            '/road/mask/overlaid',
            5
        )

        self.compressed_overlaid_road_mask_publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage,
            '/road/mask/overlaid/compressed',
            5
        )

        self.road_mask_publisher = self.create_publisher(
            sensor_msgs.msg.Image,
            '/road/mask',
            5
        )

        self.compressed_road_mask_publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage,
            '/road/mask/compressed',
            5    
        )

        #road prob
        self.overlaid_road_prob_publisher = self.create_publisher(
            sensor_msgs.msg.Image,
            '/road/prob/overlaid',
            5
        )

        self.compressed_overlaid_road_prob_publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage,
            '/road/prob/overlaid/compressed',
            5
        )

        self.road_prob_publisher = self.create_publisher(
            sensor_msgs.msg.Image,
            '/road/prob',
            5
        )

        self.compressed_road_prob_publisher = self.create_publisher(
            sensor_msgs.msg.CompressedImage,
            '/road/prob/compressed',
            5    
        )


        self.bridge = CvBridge()

        #-----------define road prediction model-------------------------------------------
        model= segmentation.deeplabv3_resnet50(num_classes=2)
        model.load_state_dict(torch.load(model_path,weights_only=True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        self.model=model

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
        ])


    def overlay_mask(self,road):
        color_mask=np.zeros((road.shape[0], road.shape[1], 3), dtype=np.uint8)
        color_mask[road == 1] = [0, 255, 0]
        blended = cv2.addWeighted(self.image, 1.0, color_mask, 0.3, 0)
        return blended
    
    def overlay_heatmap(self,label):
        green_to_red = LinearSegmentedColormap.from_list('red_orange_green', ['red','orange', 'green'])
        heatmap_colored = green_to_red(label)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_colored_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(self.image, 0.5, heatmap_colored_bgr, 0.5, 0)
        return overlay

    
    def crop_and_scale(self,image):
        image=image[self.crop_top:self.crop_bottom,self.crop_left:self.crop_right]
        image=cv2.resize(image,(self.W,self.H))
        return image
    
    def run_inference(self):
        with torch.no_grad():
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB (OpenCV loads in BGR by default)
            image_tensor=self.img_transform(Image.fromarray(img_rgb)).to(self.device).unsqueeze(0) #add batch dim
            out=self.model(image_tensor)['out']
            out=torch.nn.functional.softmax(out,dim=1)[:,0,:,:] #convert to propability
            out=out.squeeze(0) #remove batch dim
            out=out.cpu().numpy()
        return out


    def image_callback(self, msg):

        start = time.time()
        image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image=self.crop_and_scale(image)

        out=self.run_inference()
        road_prob=(out*255).astype('uint8')
        road_mask=((out>0.5)*255).astype('uint8')
        overlaid_mask=self.overlay_mask(out>0.5)
        overlaid_prob = self.overlay_heatmap(out)

        #road masks
        ros_overlaid_mask = self.bridge.cv2_to_imgmsg(overlaid_mask,encoding='bgr8')
        self.overlaid_road_mask_publisher.publish(ros_overlaid_mask)

        ros_compressed_overlaid_mask = self.bridge.cv2_to_compressed_imgmsg(overlaid_mask)
        self.compressed_overlaid_road_mask_publisher.publish(ros_compressed_overlaid_mask)

        ros_mask = self.bridge.cv2_to_imgmsg(road_mask,encoding='mono8')
        self.road_mask_publisher.publish(ros_mask)

        ros_compressed_mask = self.bridge.cv2_to_compressed_imgmsg(road_mask)
        self.compressed_road_mask_publisher.publish(ros_compressed_mask)

        #road probs
        ros_overlaid_prob = self.bridge.cv2_to_imgmsg(overlaid_prob,encoding='bgr8')
        self.overlaid_road_prob_publisher.publish(ros_overlaid_prob)

        ros_compressed_overlaid_prob = self.bridge.cv2_to_compressed_imgmsg(overlaid_prob)
        self.compressed_overlaid_road_prob_publisher.publish(ros_compressed_overlaid_prob)

        ros_prob = self.bridge.cv2_to_imgmsg(road_prob,encoding='mono8')
        self.road_prob_publisher.publish(ros_prob)

        ros_compressed_prob = self.bridge.cv2_to_compressed_imgmsg(road_prob)
        self.compressed_road_prob_publisher.publish(ros_compressed_prob)

        #log the current fps to the terminal
        fps=1/(time.time()-start)
        self.get_logger().info(f'FPS: {fps:.2f}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()