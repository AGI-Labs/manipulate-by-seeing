import torch
import yaml 
from polymetis import RobotInterface
from policy import MyPDPolicy
import time
import numpy as np
import pyrealsense2 as rs

def robot_setup(home_pos, gain_type):
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="172.16.0.4",
    )

    # Reset
    robot.set_home_pose(torch.Tensor(home_pos))
    robot.go_home()

    with open("pd_gains.yaml", "r") as stream:
        gain_dict = yaml.safe_load(stream)
    gains = gain_dict[gain_type]
    
    # Create policy instance
    q_initial = robot.get_joint_positions()
    default_kq = torch.Tensor(gains['kq'])
    default_kqd = torch.Tensor(gains['kqd'])
    policy = MyPDPolicy(
        joint_pos_current=q_initial,
        kq=default_kq,
        kqd=default_kqd,
    )
    
    # Send policy
    print("\nRunning PD policy...")
    robot.send_torch_policy(policy, blocking=False)

    return robot, policy

class Rate:
    def __init__(self, frequency):
        self._period = 1.0 / frequency
        self._last = time.time()

    def sleep(self):
        current_delta = time.time() - self._last
        sleep_time = max(0, self._period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self._last = time.time()

# Tracking Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('15322110086')
config.enable_stream(rs.stream.pose)
pipeline.start(config)

home_pos = [0.1828, -0.4909, -0.0093, -2.4412,  0.2554,  3.1310,  0.5905]
gain_type = 'record'
robot, policy = robot_setup(home_pos, gain_type)

hz = 5.0
time_to_go = 60.0
steps = int(time_to_go*hz)
rate = Rate(hz)

robot_poses = []
tool_poses = []

i = 0
while i < steps:

    frames = pipeline.wait_for_frames()
    pose = frames.get_pose_frame()

    # Validate pose is valid
    if not pose:
        continue

    # process pose data
    data = pose.get_pose_data()
    tool_poses.append(np.array([data.translation.x, data.translation.y, data.translation.z, data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w]))

    e, q = robot.get_ee_pose()
    robot_poses.append(np.concatenate((e, q)))
    rate.sleep()
    i += 1

robot.terminate_current_policy()

np.savez("calibration.npz", robot_poses = np.array(robot_poses), tool_poses = np.array(tool_poses))