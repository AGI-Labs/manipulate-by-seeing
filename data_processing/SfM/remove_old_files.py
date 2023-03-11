import os
import os.path as osp
import glob
import argparse
import numpy as np
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()

videos = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)
for video in videos:
    process_folders = sorted(glob.glob(video + "/cache/*"), reverse=False)
    for process_folder in process_folders:
        different_processes = np.array(sorted(glob.glob(process_folder + "/*"), reverse=False))
        timestamp = []
        for different_process in different_processes:
            timestamp.append(os.stat(different_process).st_mtime)
        time_order = np.argsort(timestamp)
        different_processes = different_processes[time_order]
        for i in range(len(different_processes)-1):
            shutil.rmtree(different_processes[i])