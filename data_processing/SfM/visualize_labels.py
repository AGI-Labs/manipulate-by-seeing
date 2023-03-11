import os
import os.path as osp
import json
import glob
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image, ImageFile
import imageio
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str)
args = parser.parse_args()


transform = transforms.Compose([  # [1]
    transforms.Resize(240),  # [2]
    transforms.CenterCrop(224)
])

videos = sorted(glob.glob(args.root_dir + "/push_train/train/*"))

save_dir = osp.join('/project_data/held/jianrenw/nrns', 'visualization', 'push_train', 'label')
for video in videos:
    writer = imageio.get_writer(osp.join(save_dir, '{}.mp4'.format(video.split('/')[-1])), fps=10)
    if os.path.isfile(osp.join(video, 'results.json')):
        with open(osp.join(video, 'results.json'), 'r') as json_data:
            label = json.load(json_data)
        image_paths = label['image_path']
        valid_index = label['valid_index']
        centers = label['centers']
        translations = label['translations']
        rotations = label['rotations']
        j = 0
        for i, image_path in enumerate(image_paths[:-1]):
            try:
                image = np.array(transform(Image.open(image_path)))
            except OSError:
                if i == valid_index[j]:
                    j += 1
                continue
            try:
                if i == valid_index[j]:
                    cv2.arrowedLine(image, (112, 112), (int(
                        112 + translations[j][0]*70), int(112 + translations[j][1]*70)), (0, 0, 0), 4)
                    j += 1
                    writer.append_data(image)
                else:
                    writer.append_data(image)
            except IndexError:
                writer.append_data(image)
    writer.close()
