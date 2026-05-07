#!/bin/bash

set -e

# Navigate to the directory
cd
cd ./radfoam/radfoam

# Initialize Conda and activate the environment
# (Hook is required to allow 'conda activate' inside a script)
eval "$(conda shell.bash hook)"
conda activate cuda_env


# Execute the existing combined script
bash run_all.sh

cp ./configs/*.yaml ./output/

cd