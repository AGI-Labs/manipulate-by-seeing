import os
import os.path as osp
import glob
import argparse
import subprocess
import imageio


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dir', type=str)
#     parser.add_argument('--id', type=int)

#     args = parser.parse_args()

#     video = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)[args.id]
#     frame_dir = osp.join(video, 'masked_imgs')
#     reconstruction_dir = osp.join(video, 'reconstruction')
#     # if os.path.exists(reconstruction_dir):
#     #     return
#     print("======== Processing Video ==========\n\n")
#     subprocess.call("/project_data/held/jianrenw/nrns/meshroom/meshroom_batch --input " + frame_dir +
#                     " --output " + video + "/reconstruction --cache " + video + "/cache", shell=True)

# main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()

    video = sorted(glob.glob(args.dir + "/*"), reverse=False)[args.id]
    frame_dir = osp.join(video, 'images')
    print("======== Processing Video ==========\n\n")
    subprocess.call("/project_data/held/jianrenw/nrns/meshroom/meshroom_batch --input " + frame_dir +
                    " --output " + video + "/reconstruction --cache " + video + "/cache", shell=True)

main()