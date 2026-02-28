#!/bin/bash

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

for s in {1..2}; do
    python tools/train.py configs/inc/DOTA/5+5_$s.py
done

for s in {1..5}; do
    python tools/train.py configs/inc/DOTA/10+1_$s.py
done