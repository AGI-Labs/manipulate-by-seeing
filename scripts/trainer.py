import os
import os.path as osp
import glob
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader.passive_loader import PassiveDataset
from models.passive import dynamics_model, state_feature
from models.losses import SpatialLoss, DynamicsLoss
import torch.nn.functional as F
import yaml

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true")
parser.add_argument("--experiment", type=str, help="Experiment Name.")
parser.add_argument("--task", type=str, help="Task Name.")
args = parser.parse_args()

with open('config/passive.yaml') as f:
    args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))


if args.task in ['pushing', 'pick_and_place', 'door_opening']:
    args.action_dim = 3
elif args.task in ['knob_turning']:
    args.action_dim = 12

np.random.seed(6767)

wandb.init(
    project="manipulate_passive",
    name=args.experiment,
    # Track hyperparameters and run metadata
    config={
        "batch_size": args.batch_size,
        "epoch_num": args.epoch_num,
        "task": args.task
    })


def evaluate(s, d, device, spatial, dynamics, val_loader):

    s.eval()
    d.eval()

    val_total_losses = []
    val_spatial_losses = []
    val_dynamics_losses = []

    val_translation_direction_similarities = []

    with torch.no_grad():
        for (current_image, next_image, target_image, gt_action,
             candidate_actions, best_candidate_index) in val_loader:
            current_image = current_image.to(device)
            next_image = next_image.to(device)
            target_image = target_image.to(device)

            gt_action = gt_action.to(device)  # B * dim(A)
            candidate_actions = candidate_actions.to(device)  # B * M * dim(A)
            best_candidate_index = best_candidate_index.to(device)

            current_state = s(current_image)  # B * D
            next_state = s(next_image)  # B * D
            target_state = s(target_image)  # B * D

            predict_state = d(current_state, gt_action)  # B * D
            repeat_current = current_state.unsqueeze(1).repeat(
                1, candidate_actions.size(1), 1)  # B * M * D
            candidate_states = d(repeat_current,
                                 candidate_actions)  # B * M * D
            repeat_target = target_state.unsqueeze(1).repeat(
                1, candidate_actions.size(1), 1)  # B * M * D

            similarity = F.cosine_similarity(candidate_states,
                                             repeat_target,
                                             dim=-1)  # B * M

            l_s = spatial(similarity, best_candidate_index)
            l_d = dynamics(predict_state, next_state)

            loss = l_s + l_d

            picked_action_idx = torch.argmax(similarity, dim=-1)
            picked_actions = candidate_actions[torch.arange(
                candidate_actions.size(0)), picked_action_idx]  # B * dim(A)

            translation_direction_similarity = torch.mean(
                F.cosine_similarity(picked_actions, gt_action, dim=1))
            val_translation_direction_similarities.append(
                translation_direction_similarity.item())

            val_total_losses.append(loss.item())
            val_spatial_losses.append(l_s.item())
            val_dynamics_losses.append(l_d.item())

    val_total_loss = np.mean(val_total_losses)
    val_spatial_loss = np.mean(val_spatial_losses)
    val_dynamics_loss = np.mean(val_dynamics_losses)
    val_translation_direction_similarity = np.mean(
        val_translation_direction_similarities)

    return val_total_loss, val_spatial_loss, val_dynamics_loss, val_translation_direction_similarity


valid_videos = []

if args.task == 'mix':
    tasks = sorted(os.listdir(args.root_dir))
    for task in tasks:
        videos = sorted(os.listdir(osp.join(args.root_dir, task)))
        for video in videos:
            valid_videos.append(osp.join(args.root_dir, task, video))
    valid_videos = np.array(valid_videos)
else:
    videos = sorted(os.listdir(osp.join(args.root_dir, args.task)))
    for video in videos:
        valid_videos.append(osp.join(args.root_dir, args.task, video))
    valid_videos = np.array(valid_videos)

data_len = len(valid_videos)
data_list = np.arange(data_len)
np.random.shuffle(data_list)
train_list = data_list[:int(data_len * 0.8)]
val_list = data_list[int(data_len * 0.8):]

passive_train = PassiveDataset(mode='train',
                               valid_videos=valid_videos,
                               train_list=train_list,
                               val_list=val_list,
                               task=args.task)
train_loader = DataLoader(passive_train,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4)

passive_val = PassiveDataset(mode='val',
                             valid_videos=valid_videos,
                             train_list=train_list,
                             val_list=val_list,
                             task=args.task)
val_loader = DataLoader(passive_val,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4)

device = 'cuda'
d = dynamics_model(args.action_dim)
s = state_feature()

d = d.to(device)
s = s.to(device)

if args.resume == True:
    file_name = sorted(glob.glob(args.model_dir + "/model_d/*"),
                       reverse=False)[-1]
    d.load_state_dict(torch.load(file_name))
    print('load dynamics model from:{}'.format(file_name))
    file_name = sorted(glob.glob(args.model_dir + "/model_s/*"),
                       reverse=False)[-1]
    s.load_state_dict(torch.load(file_name))
    print('load state model from:{}'.format(file_name))

spatial = SpatialLoss()
dynamics = DynamicsLoss()

# freeze r3m backbone
# for name, p in s.named_parameters():
#     if "image_feature" in name:
#         p.requires_grad = False

# params = list(d.parameters()) + list(filter(lambda p: p.requires_grad, s.parameters())) + list(a.parameters())
params = list(d.parameters()) + list(s.parameters())
optimizer = Adam(params=params, lr=1e-3)

save_dir = osp.join(args.save_dir, 'weights', '{}'.format(args.experiment))
if not osp.isdir(osp.join(save_dir, 'model_d')):
    os.makedirs(osp.join(save_dir, 'model_d'), exist_ok=True)
if not osp.isdir(osp.join(save_dir, 'model_s')):
    os.makedirs(osp.join(save_dir, 'model_s'), exist_ok=True)

for epoch in range(args.epoch_num):

    if epoch % 5 == 0:
        val_total_loss, val_spatial_loss, val_dynamics_loss, val_translation_direction_similarity = evaluate(
            s, d, device, spatial, dynamics, val_loader)
        print(
            'val total loss: {}, val spatial loss: {}, val dynamics loss: {}, val translation direction similarity:{}'
            .format(val_total_loss, val_spatial_loss, val_dynamics_loss,
                    val_translation_direction_similarity))
        wandb.log(
            {
                "val/total_loss": val_total_loss,
                "val/spatial_loss": val_spatial_loss,
                "val/dynamics_loss": val_dynamics_loss,
                "val/translation_similarity":
                val_translation_direction_similarity,
            },
            step=epoch)

        torch.save(s.state_dict(),
                   osp.join(save_dir, 'model_s', '{:04d}.ckpt'.format(epoch)))
        torch.save(d.state_dict(),
                   osp.join(save_dir, 'model_d', '{:04d}.ckpt'.format(epoch)))

    train_total_losses = []
    train_spatial_losses = []
    train_dynamics_losses = []

    train_translation_direction_similarities = []

    s.train()
    d.train()

    for batch, (current_image, next_image, target_image, gt_action,
                candidate_actions,
                best_candidate_index) in enumerate(train_loader):
        current_image = current_image.to(device)
        next_image = next_image.to(device)
        target_image = target_image.to(device)

        gt_action = gt_action.to(device)  # B * dim(A)
        candidate_actions = candidate_actions.to(device)  # B * M * dim(A)
        best_candidate_index = best_candidate_index.to(device)

        current_state = s(current_image)  # B * D
        next_state = s(next_image)  # B * D
        target_state = s(target_image)  # B * D

        predict_state = d(current_state, gt_action)  # B * D
        repeat_current = current_state.unsqueeze(1).repeat(
            1, candidate_actions.size(1), 1)  # B * M * D
        candidate_states = d(repeat_current, candidate_actions)  # B * M * D
        repeat_target = target_state.unsqueeze(1).repeat(
            1, candidate_actions.size(1), 1)  # B * M * D

        similarity = F.cosine_similarity(candidate_states,
                                         repeat_target,
                                         dim=-1)  # B * M

        l_s = spatial(similarity, best_candidate_index)
        l_d = dynamics(predict_state, next_state)

        loss = l_s + l_d

        picked_action_idx = torch.argmax(similarity, dim=-1)
        picked_actions = candidate_actions[torch.arange(
            candidate_actions.size(0)), picked_action_idx]  # B * dim(A)

        translation_direction_similarity = torch.mean(
            F.cosine_similarity(picked_actions, gt_action, dim=1))
        train_translation_direction_similarities.append(
            translation_direction_similarity.item())

        train_total_losses.append(loss.item())
        train_spatial_losses.append(l_s.item())
        train_dynamics_losses.append(l_d.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        'train total loss: {}, train spatial loss: {}, train dynamics loss: {}, translation direction similarity:{}'
        .format(np.mean(train_total_losses), np.mean(train_spatial_losses),
                np.mean(train_dynamics_losses),
                np.mean(train_translation_direction_similarities)))
    wandb.log(
        {
            "train/total_loss":
            np.mean(train_total_losses),
            "train/spatial_loss":
            np.mean(train_spatial_losses),
            "train/dynamics_loss":
            np.mean(train_dynamics_losses),
            "train/translation_similarity":
            np.mean(train_translation_direction_similarities),
        },
        step=epoch + 1)

wandb.finish()