#!/bin/bash

set -e

# Navigate to the directory
cd
cd ./triangle_splatting2/triangle-splatting2

# Initialize Conda and activate the environment
# (Hook is required to allow 'conda activate' inside a script)
eval "$(micromamba shell hook --shell bash)"
micromamba activate triangle-splatting2


# Execute the existing combined script
bash run_all.sh

# hyperparams are saved in each saved model dir as cfg_args

cd