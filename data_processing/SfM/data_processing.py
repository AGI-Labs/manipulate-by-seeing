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
    parser.add_argument('--frame_rate', type=float,
                        help='Frame rate', default=0.2)
    parser.add_argument('--id', type=int)

    args = parser.parse_args()

    video = sorted(glob.glob(args.dir + "/vids/*"), reverse=False)[args.id]

    endpath = args.dir + "/processes/"
    if not os.path.exists(endpath):
        os.mkdir(endpath)

    print("======== Splitting videos into frames ==========\n\n")
    vid_name = video.strip("/").split("/")[-1]
    process_dir = osp.join(args.dir, 'processes', '{}'.format(
        vid_name[:-4]))
    if not os.path.exists(process_dir):
        os.mkdir(process_dir)
    im = imageio.get_reader(video)
    fps = im.get_meta_data()['fps']
    frame_dir = osp.join(process_dir, 'imgs')
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
    mask_frame_dir = osp.join(process_dir, 'masked_imgs')
    if not os.path.exists(mask_frame_dir):
        os.mkdir(mask_frame_dir)
    count = 1
    for i, frame in enumerate(im):
        if i % int(fps * args.frame_rate) == 0:
            imageio.imwrite(osp.join(frame_dir,
                            'theframe{:04d}.png'.format(count)), frame)
            count += 1

        shape = [(820, 600), (1250, 1080)]
        # Creating rectangle
        img1 = ImageDraw.Draw(frame)
        img1.rectangle(shape, fill="black", outline="black")

        frame.save(osp.join(mask_frame_dir,
                            'theframe{:04d}.png'.format(count)))

    print("======== Processing Video ==========\n\n")
    subprocess.call("/project_data/held/jianrenw/nrns/meshroom/meshroom_batch --input " + mask_frame_dir +
                    " --output " + process_dir + "/reconstruction --cache " + process_dir + "/cache", shell=True)


main()
