#!/bin/bash

set -e

# Navigate to the directory
cd
cd ./QGS/QGS

# Initialize Conda and activate the environment
# (Hook is required to allow 'conda activate' inside a script)
eval "$(conda shell.bash hook)"
conda activate QGS

# Execute the existing combined script
bash run_all.sh

cp ./config/*.yaml /home/salam4/trained_models/QGS_models/

cd