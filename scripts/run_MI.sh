#!/bin/bash

# ----------------------- DIOR -----------------------
for s in {1..2}; do
    python tools/train.py configs/inc/DIOR/10+5_$s.py
done

for s in {1..3}; do
    python tools/train.py configs/inc/DIOR/5+5_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/DIOR/10+2_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/DIOR/15+1_$s.py
done

# ----------------------- DOTA -----------------------
for s in {1..2}; do
    python tools/train.py configs/inc/DOTA/5+5_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/DOTA/10+1_$s.py
done

# ----------------------- VOC -----------------------
for s in {1..2}; do
    python tools/train.py configs/inc/VOC/10+5_$s.py
done

for s in {1..3}; do
    python tools/train.py configs/inc/VOC/5+5_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/VOC/10+2_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/VOC/15+1_$s.py
done