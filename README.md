# Learning to Manipulate by Learning to See

##### [[Project Page]](https://agi-labs.github.io/manipulate-by-seeing/) [[Paper]]() [[Video]]()

Jianren Wang<sup>*1</sup>, Sudeep Dasari<sup>*1</sup>, Mohan Kumar<sup>1</sup>, Shubham Tulsiani<sup>1</sup>, Abhinav Gupta<sup>1</sup>

<sup>1</sup>Carnegie Mellon University, * indicates equal contribution

##

<img src="images/teaser.gif" width="400">

## Data

Data is located [here](https://drive.google.com/drive/folders/1Z_B5_z0LeYX7cnkAUGZksn25exYhuUXJ?usp=share_link).

## Usage
Below are example scripts for training and testing on franka emika.
### Setup

1. Clone repo.
```shell
git clone https://github.com/AGI-Labs/manipulate-by-seeing
```
2. Create and activate conda environment.
```shell
conda env create -f environment.yml
conda activate seeing
```

3. Set env path.

```shell
export PYTHONPATH=$PYTHONPATH:path_to_proj/
```

### Training ###

1. Download the data from the above google drive. 

2. Put action_*.pickle under dataloader. 

3. Edit yaml files
```shell
root_dir: path_to_data/
save_dir: path_to_save_the_weights/
```

4. To train:

    ```shell
    python scripts/trainer.py --experiment='knob_turning_0113' --task='knob_turning'
    ```

### Testing

1. Follow [this](https://github.com/facebookresearch/fairo/tree/main/polymetis) to install polymetis for robot control.

2. Edit yaml files
```shell
model_weights: path_to_pretrained_models/
gripper_weights: path_to_pretrained_gripper_predictor/
```

3. Edit GripperController
```shell
close configuration: [here](https://github.com/AGI-Labs/manipulate-by-seeing/blob/1d1f91d8bee98fa7fee399da3993efe0a71c7671/inference/robot_setup.py#L76)
open configuration: [here](https://github.com/AGI-Labs/manipulate-by-seeing/blob/1d1f91d8bee98fa7fee399da3993efe0a71c7671/inference/robot_setup.py#L79)
```

4. This command to test on Franka Emika.
```shell
python inference/run.py --task='pick_and_place'
```

### Acknowledgement

The controller of dynamixel motors is borrowed from [Vikash Kumar](https://vikashplus.github.io/)

If you find our paper or code useful, please cite our papers:

```
@article{Wang_2023_cvpr, 
author = {Wang, Jianren and Dasari, Sudeep and Kumar, Mohan and Tulsiani, Shubham and Gupta, Abhinav}, 
journal = {arxiv}, 
title = {{Learning to Manipulate by Learning to See}}, 
year = {2023} 
}
```
