# Gradient Decomposition and Alignment for Incremental Object Detection

### Install

```bash
conda create -n gda python=3.8 -y

conda activate gda

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda==11.6 mkl==2024.0 -c pytorch -c nvidia -y

pip install -U openmim

mim install mmengine==0.7.3 mmcv==2.0.0

pip install -r requirements.txt

pip install -v -e .
```

### Dataset Prepare

#### DIOR

Assume you are under DIOR directory, then  `mkdir ImageSets/Main` and `mv trainval.txt Main/` 

#### DOTA

Use [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) to crop the original images into patches of 1024x1024 with an overlap of 256 pixels (i.e., `gap=768`), and save under directory `trainsplit` and `testsplit`, respectively.  After cropping, remove empty and invalid (bbox_size=0) annotations.

DOTA directory now should be like

```
DOTA
├── test
│   ├── images
│   ├── labelTxt
│   └── testset.txt
├── testsplit
│   ├── images
│   └── labelTxt
├── train
│   ├── images
│   └── labelTxt
└── trainsplit
    ├── images
    └── labelTxt
```

#### Link Data

Link your DIOR and DOTA dataset to `/data/my_code/dataset`, and the final structure should be like

```
/data/my_code/dataset
├── DIOR
│   ├── Annotations
│   ├── ImageSets
│   └── JPEGImages
├── DOTA
│   └── test
│		 ├── labelTxt
│		 └── testset.txt
└── DOTA_xml
     ├── Annotations
     ├── ImageSets
     └── JPEGImages
```

#### Split Data for IL

Execute the split script below to segment dataset in accordance to different incremental settings(task 0 excluded).

```bash
. scripts/split.sh
```

### Run

We provide scripts for training.

```bash
# first step(task0)
. scripts/run_first_step.sh

# Single-increment
. scripts/run_SI.sh

# Multi-increment
. scripts/run_MI.sh
```