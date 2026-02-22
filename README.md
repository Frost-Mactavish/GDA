from os import mkdir

### Install
You can follow the steps to prepare the environment:
```
conda create -n CLOD python=3.8 -y

source activate CLOD

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install tqdm

pip install -U openmim

mim install mmengine==0.7.3

mim install mmcv==2.0.0

# cd GDA-COLD

mkdir cache
mkdir temp_cheakpoints
pip install -v -e .
chmod -R 777 ./tools/dist_train.sh
```

### Dataset Prepare
Download the VOC2007 dataset and note the root directory path of your VOC2007 folder.
Open replace.py and set the variable: new_string = "your_VOC2007_directory_path"
```python
# Run
python replace.py
```
Next, configure the pattern variable in pascal_voc_split.py to either '10+10' or '5+5', depending on your desired experimental setting. Then, execute the script:
```python
# run
python pascol_voc_split.py
```

### Download Model Weights
You can download our provided model weight for Task0 from [Link](https://drive.google.com/drive/folders/120KLcvSG1Lz81Cf-KIhT7RqrbDSFapiO?usp=sharing), and moving the model weights you have downloaded to the ./base folder.

### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.

# you can train VOC(10-10) [Task0, Task1]
# This is Task1 for VOC(10-10)
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/faster-rcnn-increase/voc/10-10/gad-iod/10+10_gda.py 1

# you can train VOC(5-5) [Task0, Task1, Task2, Task3]
# This is Task1 for VOC(10-10)
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/faster-rcnn-increase/voc/5-5/gda-iod/5+5_gda.py 1
# This is Task2 for VOC(10-10)
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/faster-rcnn-increase/voc/5-5/gda-iod/10+5_gda.py 1
# This is Task3 for VOC(10-10)
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/faster-rcnn-increase/voc/5-5/gda-iod/15+5_gda.py 1
```

### Citation
```
@inproceedings{gda_iod,
 author = {W. Luo and S. Zhang and D. Cheng and Y. Xing and G. Liang and P. Wang and Y. Zhang},
 title  = {Gradient Decomposition and Alignment for Incremental Object Detection},
 year   = {2025},
 booktitle = {ICCV}
}
```
