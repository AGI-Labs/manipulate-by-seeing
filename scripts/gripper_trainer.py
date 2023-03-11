import os
import os.path as osp
import glob
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader.gripper_loader import GripperDataset
from models.gripper import gripper_action
from models.losses import GripperLoss
import yaml

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action="store_true")
parser.add_argument("--experiment", type=str, help="Experiment Name.")
parser.add_argument("--task", type=str, help="Task Name.")
args = parser.parse_args()

with open('passive.yaml') as f:
    args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))


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


def evaluate(g, device, criterion, val_loader):

    g.eval()

    val_losses = []

    with torch.no_grad():
        for (current_image, gt_action) in val_loader:
            current_image = current_image.to(device)
            gt_action = gt_action.to(device)

            predict_action = g(current_image)

            loss = criterion(predict_action, gt_action)

            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)

    return val_loss


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

data_len = len(valid_videos[:101])
data_list = np.arange(data_len)
np.random.shuffle(data_list)
train_list = data_list[:int(data_len * 0.8)]
val_list = data_list[int(data_len * 0.8):]

bc_train = GripperDataset(mode='train',
                          valid_videos=valid_videos,
                          train_list=train_list,
                          val_list=val_list,
                          task=args.task)
trainloader = DataLoader(bc_train,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=4)

bc_val = GripperDataset(mode='val',
                        valid_videos=valid_videos,
                        train_list=train_list,
                        val_list=val_list,
                        task=args.task)
valloader = DataLoader(bc_val,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=4)

device = 'cuda'
g = gripper_action()
g = g.to(device)

if args.resume == True:
    file_name = sorted(glob.glob(args.model_dir + "/*"), reverse=False)[-1]
    g.load_state_dict(torch.load(file_name))

criterion = GripperLoss()
params = list(g.parameters())
optimizer = Adam(params=params, lr=1e-3)

save_dir = osp.join(args.save_dir, 'weights', '{}'.format(args.experiment))
if not osp.isdir(osp.join(save_dir)):
    os.makedirs(osp.join(save_dir), exist_ok=True)

for epoch in range(args.epoch_num):

    if epoch % 5 == 0:
        val_loss = evaluate(g, device, criterion, valloader)
        print('val loss: {}'.format(val_loss))
        wandb.log({
            "val/loss": val_loss,
        }, step=epoch)

        torch.save(g.state_dict(),
                   osp.join(save_dir, '{:04d}.ckpt'.format(epoch)))

    train_losses = []

    g.train()

    for batch, (current_image, gt_action) in enumerate(trainloader):
        current_image = current_image.to(device)
        gt_action = gt_action.to(device)

        predict_action = g(current_image)

        loss = criterion(predict_action, gt_action)

        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('train loss: {}'.format(np.mean(train_losses)))
    wandb.log({
        "train/loss": np.mean(train_losses),
    }, step=epoch + 1)

wandb.finish()