#!/bin/bash

set -e

# Navigate to the directory
cd
cd ./models/rade_gs/RaDe-GS

# Initialize Conda and activate the environment
# (Hook is required to allow 'conda activate' inside a script)
eval "$(conda shell.bash hook)"
conda activate radegs


# Execute the existing combined script
bash run_all.sh

# hyperparams are saved in each saved model dir as cfg_args

cd