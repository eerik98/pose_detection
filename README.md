
## Usage
1. create a new python venv and install dependencies with pip. Using venv avoids version conflicts with your system-level libraries. 
```
python3 -m venv <env name>
source <path to env>/bin/activate
pip3 install -r <path to requirements.txt>
```
2. In the launch file change the `venv_python_path` so that it points to the site packages of the created venv. 
3. Build your ROS2 workspace with `colcon build`
4. Publish images to the topic defined in `params.yaml`
5. Launch the node with `ros2 launch pose_detection launch.py`.
