conda create -n eiod python=3.8 -y

source activate eiod

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 mkl==2024.0 -c pytorch -c nvidia

pip install -U openmim

mim install mmengine mmcv==2.0.0

pip install -v -e .