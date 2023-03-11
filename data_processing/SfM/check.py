import os
import os.path as osp
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()
with open(
        '/project_data/held/jianrenw/nrns/new_run.sh','w') as file:
    videos = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)
    no_recontruction_ids = []
    no_camera_ids = []
    for i, video in enumerate(videos):
        reconstruction_dir = osp.join(video, 'reconstruction')
        if not os.path.exists(reconstruction_dir):
            no_recontruction_ids.append(i)
        tmp_folder = glob.glob(video + "/cache/StructureFromMotion/*")[0]
        if not os.path.isfile(osp.join(tmp_folder, 'cameras.sfm')):
            no_camera_ids.append(i)
    for i in no_camera_ids:
        if i not in no_recontruction_ids:
            command_line = "python sbatch.py launch --cmd=\"python data_cleaning/sfm.py --dir /project_data/held/jianrenw/nrns/mix_videos --id {}\" --name=\"sfm_{:03d}\"\n".format(i,i)
            file.write(command_line)


#     for i, video in enumerate(videos):
#         reconstruction_dir = osp.join(video, 'reconstruction')
#         tmp_folder = glob.glob(video + "/cache/StructureFromMotion/*")[0]
#         if not os.path.exists(reconstruction_dir) or not os.path.isfile(osp.join(tmp_folder, 'cameras.sfm')):
#             command_line = "python sbatch.py launch --cmd=\"python data_cleaning/sfm.py --dir /project_data/held/jianrenw/nrns/mix_videos --id {}\" --name=\"sfm_{:03d}\"\n".format(i,i)
#             file.write(command_line)

# videos = sorted(glob.glob(args.dir + "/processes/*"), reverse=False)
# no_recontruction_ids = []
# no_camera_ids = []
# for i, video in enumerate(videos):
#     reconstruction_dir = osp.join(video, 'reconstruction')
#     if not os.path.exists(reconstruction_dir):
#         no_recontruction_ids.append(i)
#     tmp_folder = glob.glob(video + "/cache/StructureFromMotion/*")[0]
#     if not os.path.isfile(osp.join(tmp_folder, 'cameras.sfm')):
#         no_camera_ids.append(i)
# for i in no_camera_ids:
#     if i not in no_recontruction_ids:
#         print(i)