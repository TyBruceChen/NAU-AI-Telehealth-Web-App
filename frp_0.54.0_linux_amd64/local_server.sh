#!/bin/bash

#initialize conda for bash shell
CONDA_PATH="/home/piko/miniconda3" #installation location of conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate web_3_12
python3 /home/piko/Documents/Flask/app.py
