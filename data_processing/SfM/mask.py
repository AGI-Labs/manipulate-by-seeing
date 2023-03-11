import os
import os.path as osp
import glob
import argparse
import subprocess
import imageio
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()

    video = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)[args.id]

    print("======== Draw Mask on Frames ==========\n\n")
    imgs = sorted(glob.glob(video + "/imgs/*"))
    for i in imgs:
        title = i.split("/")[-1]
        shape = [(750, 550), (1250, 1080)]
        # Creating rectangle
        im = Image.open(i)
        img1 = ImageDraw.Draw(im)
        img1.rectangle(shape, fill="black", outline="black")

        if not os.path.exists(video + "/masked_imgs/"):
            os.mkdir(video + "/masked_imgs/")
        print(video + "/masked_imgs/" + title)
        im.save(video + "/masked_imgs/" + title)
        
main()
