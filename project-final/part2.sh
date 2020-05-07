#!/bin/bash
#$ -cwd
export CUDA_VISIBLE_DEVICES="$1"
python3 -u part2.py
