#!/bin/bash

# Usage: ./run_batch.sh /path/to/root_dir

# 1. Get the root directory from the command line argument
ROOT_DIR="/home/salam4/renders/models"
# 2. Set the path to your python script (CHANGE THIS)
PYTHON_SCRIPT="multi_metrics.py"

# Check if ROOT_DIR was provided
if [ -z "$ROOT_DIR" ]; then
    echo "Error: Please provide a root directory."
    echo "Usage: $0 /path/to/root_dir"
    exit 1
fi

# Iterate through every item in the root directory
for parent_dir in "$ROOT_DIR"/*; do
    # Only process if it is a directory
    if [ -d "$parent_dir" ]; then
        echo "------------------------------------------------"
        echo "Processing group: $parent_dir"
        
        # Run the python script
        # "$parent_dir"/* expands to all subfolders inside that directory
        python "$PYTHON_SCRIPT" -m "$parent_dir"/*
    fi
done