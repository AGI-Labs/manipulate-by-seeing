#!/usr/bin/env python3

import os
import os.path as osp

from numpy.core.records import record
import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import time
from pynput import mouse

# Depth Camera
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('048122070002')
config_1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# Tracking Camera
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('15322110086')
config_2.enable_stream(rs.stream.pose)


# Start streaming from both cameras
pipeline_1.start(config_1)
align_to = rs.stream.color
align = rs.align(align_to)

pipeline_2.start(config_2)

# env_dir = '/media/jianrenw/jianrenw/jianrenw'
env_dir = '/home/jianrenw/Research/data/jianrenw'
# tasks = ['push','stack','close','fold_cloth','flatten_cloth','pick_place','wiping_table','pushing_rope','sweeping','pouring','scooping','rotate_knob','open_cabinet','open_drawer','grasp','plugging']
tasks = ['pushing_rope','sweeping','pick_place','wiping_table','grasp']

collected_num = len(os.listdir(env_dir))

tasks_name = tasks[collected_num // 100]
trajectory_idx = collected_num % 100

save_fold = '{}_{:04d}'.format(tasks_name, trajectory_idx)

save_dir = osp.join(env_dir, save_fold)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

recording = False
recorded = False

def on_click(x, y, button, pressed):
    global recording
    if pressed:
        if recording:
            recording = False
        else:
            recording = True

listener = mouse.Listener(on_click=on_click)
listener.start()

all_points = []


while not recorded:

    while recording:
        if not recorded:
            i = 0
            start_time = time.time()
            print('start')
            recorded = True
        
        # get rgbd data
        frames_1 = pipeline_1.wait_for_frames()
        aligned_frames = align.process(frames_1)
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 1280x720 depth image
        color_frame = aligned_frames.get_color_frame()

        # get pose information
        frames_2 = pipeline_2.wait_for_frames()
        pose = frames_2.get_pose_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame or not pose:
            continue

        # process rgbd data
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # process pose data
        data = pose.get_pose_data()

        # save data point
        position = np.array([data.translation.x, data.translation.y,data.translation.z])
        rotation = np.array([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])
        velocity = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        acceleration = np.array([data.acceleration.x, data.acceleration.y, data.acceleration.z])
        
        point = {'rgb': color_image, 'd': depth_image, 'p': position, 'r': rotation, 'v': velocity, 'a':acceleration}

        with open(osp.join(save_dir, '{:04d}.pickle'.format(i)), 'wb') as handle:
            pickle.dump(point, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        i += 1
        
end_time = time.time()
print('end, video length: {}, frames: {}'.format(end_time - start_time, i))