#!/bin/bash

python tools/train.py configs/inc/DIOR/19+1.py
python tools/train.py configs/inc/DIOR/15+5.py
python tools/train.py configs/inc/DIOR/10+10.py
python tools/train.py configs/inc/DIOR/5+15.py

python tools/train.py configs/inc/DOTA/14+1.py
python tools/train.py configs/inc/DOTA/10+5.py
python tools/train.py configs/inc/DOTA/8+7.py
python tools/train.py configs/inc/DOTA/5+10.py

python tools/train.py configs/inc/VOC/19+1.py
python tools/train.py configs/inc/VOC/15+5.py
python tools/train.py configs/inc/VOC/10+10.py
python tools/train.py configs/inc/VOC/5+15.py