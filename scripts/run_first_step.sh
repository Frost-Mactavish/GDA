#!/bin/bash

python tools/train.py configs/base/DIOR/5.py
python tools/train.py configs/base/DIOR/10.py
python tools/train.py configs/base/DIOR/15.py
python tools/train.py configs/base/DIOR/19.py

python tools/train.py configs/base/DOTA/5.py
python tools/train.py configs/base/DOTA/8.py
python tools/train.py configs/base/DOTA/10.py
python tools/train.py configs/base/DOTA/14.py