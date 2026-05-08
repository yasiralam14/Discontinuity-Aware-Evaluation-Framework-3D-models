#!/bin/bash

# Define the list of scenes

set -e

cd
cd ./models/linprim/linear-splatting

eval "$(conda shell.bash hook)"
conda activate linear_splatting

bash ./render_all.sh

cd 

scenes=(
    "Barn" "Caterpillar"
    "Courthouse" "Ignatius" 
    "Meetingroom" "Truck"
)
# scenes=(
#     "Barn" "bicycle" "bonsai" "Caterpillar" "counter"
#     "Courthouse" "flowers" "garden" "Ignatius" "kitchen"
#     "Meetingroom" "room" "stump" "treehill" "Truck"
# )

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Processing: $scene"

    # Define source and destination paths
    src="/home/salam4/trained_models/linprim/${scene}/test/ours_30000"
    dest="/home/salam4/renders/models/linprim/${scene}/test/${scene}"
    

    # Create destination directory (mkdir -p creates parent dirs and ignores if exists)
    mkdir -p "$dest/gt" "$dest/renders" "$dest/masks"

    # Copy gt and renders directories
    # Check if source exists before copying to avoid errors
    cp -a "$src/gt/." "$dest/gt/"
    cp -a "$src/renders/." "$dest/renders/"
    cp /home/salam4/trained_models/linprim/${scene}/cfg_args /home/salam4/renders/models/linprim/${scene}
    python /home/salam4/renders/run_laplacian_and_save_masks.py --input_dir "$dest/gt" --output_dir "$dest/masks"
done

eval "$(conda shell.bash hook)"
conda activate ht_hvae

cd

cd ./metrics

python multi_metrics.py -m /home/salam4/renders/models/linprim/*


