
## Usage
1. create a new python venv and install dependencies with pip. Using venv avoids version conflicts with your system-level libraries. 
```
python3 -m venv <env name>
source <path to env>/bin/activate
pip3 install -r <path to requirements.txt>
```
2. Download trained weights for road segmentation from [this link](https://drive.google.com/file/d/14lupCHV5pTs1rM5lTA0xl7trN7DdpVif/view?usp=drive_link)
3. In the launch file change the `venv_python_path` so that it points to the site packages of the created venv. 
4. Set parameters in `config/params.yaml`
5. Build your ROS2 workspace with `colcon build`
4. Publish images to the topic defined in `params.yaml`
5. Launch the node with `ros2 launch road_segmentation launch.py`. Detection are published to topics starting with `/road`:
     - `/mask` refers to binary detection: each pixel is either road or background
     - `/prob` refers to continuous detection: a higher value means a higher road probability
     - `/overlaid` overlays the detection with the image for visualization purposes
     - `/compressed` compressed the output for lower bitrate. If you want to record any output to a bagfile, use the compressed topics. The uncompressed images take up a lot of space.   
