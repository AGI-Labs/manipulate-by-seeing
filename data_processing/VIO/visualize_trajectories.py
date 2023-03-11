#!/usr/bin/env python3

import os
import os.path as osp
import cv2
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt

# trajectory_dir = '/home/jianrenw/Research/data/jianrenw/fold_cloth_0078'
trajectory_dir = '/home/jianrenw/Downloads/flatten_cloth/flatten_cloth_0035'
save_dir = '.'
writer = imageio.get_writer(osp.join(save_dir, 'test.mp4'), fps=30)

frames = sorted(os.listdir(trajectory_dir))
color_images = []
depth_images = []
positions = []

for frame in frames:
    with open(osp.join(trajectory_dir, frame), 'rb') as handle:
        frame_data = pickle.load(handle)
    color_image = frame_data['rgb']
    color_images.append(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_image = frame_data['d']
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_images.append(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    position = frame_data['p']
    positions.append(position)

for i in range(len(frames)-1):
    translation = positions[i+1]-positions[i]
    cv2.arrowedLine(color_images[i], (640, 360), (int(
                        640 + translation[0]*10000), int(360 - translation[1]*10000)), (0, 0, 0), 4)
    cat_image = np.hstack((color_images[i], depth_images[i]))
    writer.append_data(cat_image)
writer.close()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
positions = np.array(positions)
x = positions[:,0]
y = - positions[:,2]
z = positions[:,1]

ax.plot(x,y,z)

max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('$X$', fontsize=20, rotation=150)
ax.set_ylabel('$Y$')
ax.set_zlabel(r'$Z$', fontsize=30, rotation=60)

plt.show()


