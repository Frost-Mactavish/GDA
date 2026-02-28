#!/bin/bash

alias split='python tools/dataset_split.py'

split -d DIOR -p 19+1
split -d DIOR -p 15+5
split -d DIOR -p 10+10
split -d DIOR -p 5+15

split -d DIOR -p 15+1
split -d DIOR -p 10+2
split -d DIOR -p 5+5
split -d DIOR -p 10+5


split -d DOTA -p 14+1
split -d DOTA -p 10+5
split -d DOTA -p 8+7
split -d DOTA -p 5+10

split -d DOTA -p 10+1
split -d DOTA -p 5+5