import numpy as np
import torch
from robot_setup import RobotController

from models.passive import dynamics_model, state_feature
from models.gripper import gripper_action
import torch.nn.functional as F
import yaml
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Task Name.")

my_yaml = yaml.load(open('config/passive.yaml'), Loader=yaml.FullLoader)

args = parser.parse_args()
args.__dict__.update(my_yaml)

if args.task in ['pushing', 'pick_and_place', 'door_opening']:
    args.action_dim = 3
elif args.task in ['knob_turning']:
    args.action_dim = 12

# initialize robot
my_robot = RobotController(my_yaml, args.task)
my_robot.go_home()

# initialize model

device = 'cuda'
d = dynamics_model(args.action_dim)
s = state_feature()
gripper_model = gripper_action()

d = d.to(device)
s = s.to(device)
gripper_model = gripper_model.to(device)

d.load_state_dict(
    torch.load(my_yaml['model_weights']['d_{}'.format(args.task)]))
d.eval()
s.load_state_dict(
    torch.load(my_yaml['model_weights']['s_{}'.format(args.task)]))
s.eval()

gripper_model.load_state_dict(torch.load(
    my_yaml['gripper_weights'][args.task]))
gripper_model.eval()

with open('data_loader/action_{}.pickle'.format(args.task), 'rb') as handle:
    random_actions = pickle.load(handle)
candidate_actions = np.stack(random_actions)

if args.task in ['knob_turning']:
    candidate_actions = candidate_actions.reshape(-1, 16)[:, :12]

candidate_actions = torch.from_numpy(candidate_actions).float().to(device)  # M * D

# load goal image
goal_image = my_robot.get_goal_image('goal_{}.png'.format(args.task))

user_in = input("Begin Testing?")
if user_in == 'y':
    with torch.no_grad():
        while True:
            current_image, gripper_image = my_robot.get_camera_frame()

            gripper_action = gripper_model(gripper_image.to(device))

            current_state = s(current_image.to(device))  # 1 * D
            goal_state = s(goal_image.to(device))  # 1 * D

            repeat_current = current_state.repeat(candidate_actions.size(0),
                                                  1)  # M * D
            repeat_goal = goal_state.repeat(candidate_actions.size(0),
                                            1)  # M * D

            candidate_states = d(repeat_current, candidate_actions)  # M * D

            similarity = F.cosine_similarity(candidate_states,
                                             repeat_goal,
                                             dim=-1)  # M

            picked_action_idx = torch.argmax(similarity)
            picked_action = candidate_actions[picked_action_idx]

            T_bt0_bt1 = np.eye(4)
            if args.task in ['knob_turning']:
                picked_action = picked_action.cpu().detach().numpy().reshape(3, 4) # 3 * 4
                T_bt0_bt1[:3, :] = picked_action
            else:
                picked_action = picked_action.cpu().detach().numpy() # 3
                T_bt0_bt1[:3, 3] = picked_action * 0.04

            my_robot.step(T_bt0_bt1, gripper_action)