import numpy as np
import torch
import torch
import torchcontrol as toco
from typing import Dict
from polymetis import RobotInterface
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik
from scipy.spatial.transform import Rotation as R
import cv2
import time

from torchvision import transforms
import pyrealsense2 as rs

from dynamixel.dynamixel_py import dxl


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


class MyPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, joint_pos_current, kq, kqd, **kwargs):
        """
        Args:
            joint_pos_current (torch.Tensor):   Joint positions at initialization
            kq, kqd (torch.Tensor):             PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.q_desired = torch.nn.Parameter(joint_pos_current)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Execute PD control
        output = self.feedback(q_current, qd_current, self.q_desired,
                               torch.zeros_like(qd_current))

        return {"joint_torques": output}


class GripperController:
    ACTUATOR_IDS = [1]

    def __init__(self):
        self.gripper = dxl(motor_id=self.ACTUATOR_IDS,
                           motor_type='X',
                           baudrate=57600,
                           devicename='/dev/ttyUSB0')
        self.gripper.open_port()
        self.gripper.engage_motor(motor_id=self.ACTUATOR_IDS, enable=True)

    def close(self):
        self.gripper.set_des_pos(self.ACTUATOR_IDS, [None])

    def open(self):
        self.gripper.set_des_pos(self.ACTUATOR_IDS, [None])

    def shutdown(self):
        self.gripper.engage_motor(motor_id=self.ACTUATOR_IDS, enable=False)


class RobotController:

    def __init__(self, my_yaml, task):

        # camera motion to robot motion
        self.T_br_bt = np.array([[0.6965, 0.6512, 0.3015, 0.1061],
                                 [0.7176, -0.6284, -0.3003, -0.1021],
                                 [-0.0061, 0.4255, -0.9049, 0.0678],
                                 [0.0000, 0.0000, 0.0000, 1.0000]])

        # camera initialize
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # robot initialize
        self.gripper_close = False
        self.gripper_controller = GripperController()

        FRANKA_XML = assets.get_contents(
            '/home/nitro/work/distl/manipulate-by-seeing/franka_sim/franka_panda_no_gripper.xml')
        self.physics = mujoco.Physics.from_xml_string(FRANKA_XML)

        self.robot = RobotInterface(ip_address=my_yaml["ip_address"],
                                    enforce_version=False)

        gains = my_yaml['pd_gains']
        q_initial = self.robot.get_joint_positions()

        default_kq = torch.Tensor(gains['kq'])
        default_kqd = torch.Tensor(gains['kqd'])

        policy = MyPDPolicy(
            joint_pos_current=q_initial,
            kq=default_kq,
            kqd=default_kqd,
        )

        print("\nSending PD policy...")
        self.robot.send_torch_policy(policy, blocking=False)
        self.home_pos = my_yaml["home_pos"][task]

        # transform
        r3m = my_yaml['r3m']
        self.r3m = r3m
        if r3m:
            self.color_transform = transforms.Compose(
                [transforms.Resize((240, 240)),
                 transforms.CenterCrop(224)])
        else:
            self.color_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.gripper_transform = transforms.Compose(
            [transforms.Resize((240, 240)),
             transforms.CenterCrop(224)])

        # rate
        hz = 50.0
        self.rate = Rate(hz)

    def go_home(self):
        self.gripper_controller.open()
        self.gripper_close = False
        q_initial = self.robot.get_joint_positions()
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=q_initial,
            goal=torch.Tensor(self.home_pos),
            time_to_go=3,
            hz=50)
        joint_positions = [waypoint["position"] for waypoint in waypoints]

        for joint_position in joint_positions:
            self.robot.update_current_policy({"q_desired": joint_position})
            self.rate.sleep()

        jp = self.robot.get_joint_positions()
        state = np.zeros(14)
        state[:7] = np.array(jp) - np.array(
            [0, 0, 0, 0, 0., np.pi / 2, np.pi / 4], dtype=np.float32)
        self.physics.set_state(state)
        self.physics.step()

    def get_camera_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        if self.r3m:
            current_image = self.color_transform(
                torch.from_numpy(color_image).permute((2, 0, 1)))
        else:
            current_image = self.color_transform(
                torch.from_numpy(color_image / 255.).permute((2, 0, 1)))
        current_image = torch.unsqueeze(current_image, 0).float()

        gripper_image = self.gripper_transform(
            torch.from_numpy(color_image).permute((2, 0, 1)))
        gripper_image = torch.unsqueeze(gripper_image, 0).float()

        return current_image, gripper_image

    def get_goal_image(self, goal_path):
        goal_image = cv2.imread(goal_path)
        goal_image = cv2.cvtColor(goal_image, cv2.COLOR_BGR2RGB)
        if self.r3m:
            goal_image = self.color_transform(
                torch.from_numpy(goal_image).permute((2, 0, 1)))
        else:
            goal_image = self.color_transform(
                torch.from_numpy(goal_image / 255.).permute((2, 0, 1)))
        goal_image = torch.unsqueeze(goal_image, 0).float()

        return goal_image

    def step(self, robot_action, gripper_action):
        e, q = self.robot.get_ee_pose()
        T_srbr0 = np.eye(4).astype(np.float32)
        T_srbr0[:3, :3] = R.from_quat(q).as_matrix()
        T_srbr0[:3, 3] = e
        current_pos = self.robot.get_joint_positions()

        if gripper_action.item() > 0.8 and not self.gripper_close:
            self.gripper_close = True
            self.gripper_controller.close()
        elif gripper_action.item() < 0.8:
            self.gripper_close = False
            self.gripper_controller.open()

        T_srbr1 = T_srbr0 @ self.T_br_bt @ robot_action @ np.linalg.inv(
            self.T_br_bt)
        T_srbr1[2, 3] = min(0.75, T_srbr1[2, 3])

        quat = R.from_matrix(T_srbr1[:3, :3]).as_quat()
        # solve for ik
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name='end_effector',
            target_pos=T_srbr1[:3, 3],
            target_quat=np.array([quat[3], quat[0], quat[1], quat[2]]))

        qpos = result.qpos
        state = np.zeros(14)
        state[:7] = qpos
        self.physics.set_state(state)
        self.physics.step()

        # running on robot
        next_pos = qpos + np.array([0, 0, 0, 0, 0., np.pi / 2, np.pi / 4],
                                   dtype=np.float32)
        all_poses = np.linspace(current_pos, next_pos, num=100)
        for pos in all_poses:
            self.robot.update_current_policy(
                {"q_desired": torch.from_numpy(pos)})
            self.rate.sleep()
