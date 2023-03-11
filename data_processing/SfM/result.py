import os
import os.path as osp
import glob
import argparse
import numpy as np
from numpy.linalg import inv
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()


# videos = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)
videos = sorted(glob.glob(args.dir + "/*"), reverse=False)
for i, video in enumerate(videos):
    if not osp.isdir(osp.join(video, "cache")):
        continue
    tmp_folder = glob.glob(video + "/cache/StructureFromMotion/*")[0]
    if not osp.isfile(osp.join(tmp_folder, 'cameras.sfm')):
        continue
    with open(osp.join(tmp_folder, 'cameras.sfm'), 'r') as json_data:
        camera = json.load(json_data)
    view_pose_ids = np.array([x["poseId"] for x in camera["views"]])
    # image_paths = np.array(
    #     [x["path"].replace('\\', '').replace('masked_imgs', 'imgs') for x in camera["views"]])
    image_paths = np.array(
        [x["path"].replace('\\', '') for x in camera["views"]])
    temporal_order = np.argsort(image_paths)
    temporal_order = np.argsort(image_paths)
    view_pose_ids = view_pose_ids[temporal_order]
    image_paths = image_paths[temporal_order]

    camera_pose_ids = np.array([x["poseId"] for x in camera["poses"]])
    rots = np.array(
        [np.array(x["pose"]["transform"]["rotation"]).astype(float).reshape((3, 3)).T for x in camera["poses"]])
    centers = np.array(
        [np.array(x["pose"]["transform"]["center"]).astype(float) for x in camera["poses"]])

    # get image ids if image are valid
    valid_index = []
    camera_real_index = []
    for i, view_pose_id in enumerate(view_pose_ids):
        idx = np.where(camera_pose_ids == view_pose_id)[0]
        if idx.size > 0:
            valid_index.append(i)
            camera_real_index.append(idx[0])

    rots = rots[camera_real_index]
    centers = centers[camera_real_index]

    translations = []
    rotations = []
    for i in range(len(camera_real_index) - 1):
        center1 = centers[i]
        center2 = centers[i+1]
        rot1 = rots[i]
        rot2 = rots[i+1]

        translation = center2 - center1
        rotation = np.matmul(rot2, inv(rot1))
        translations.append(translation)
        rotations.append(rotation.tolist())

    # normalize translations
    scale_factor = np.max(np.abs(translations))
    translations = np.array(translations) / scale_factor

    results = {}
    results['image_path'] = image_paths.tolist()
    results['valid_index'] = valid_index
    results['rots'] = rots.tolist()
    results['centers'] = centers.tolist()
    results['translations'] = translations.tolist()
    results['rotations'] = rotations
    results['scale_factor'] = scale_factor

    with open(osp.join(video, 'results.json'), 'w') as fp:
        json.dump(results, fp)
