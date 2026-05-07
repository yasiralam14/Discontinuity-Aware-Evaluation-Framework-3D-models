#!/bin/bash

set -e

# Navigate to the directory
cd
cd ./models/3dcs/convex-splatting

# Initialize Conda and activate the environment
# (Hook is required to allow 'conda activate' inside a script)
eval "$(micromamba shell hook --shell bash)"
micromamba activate convex_splatting

# Execute the existing combined script
bash run_all.sh

# hyperparams are saved in each saved model dir as cfg_args
cd 