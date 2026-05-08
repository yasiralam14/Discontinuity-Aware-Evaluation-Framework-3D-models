set -e

cd

eval "$(conda shell.bash hook)"

cd 

conda activate ht_hvae

scenes=(
    "Barn" "bicycle" "bonsai" "Caterpillar" "counter"
    "Courthouse" "flowers" "garden" "Ignatius" "kitchen"
    "Meetingroom" "room" "stump" "treehill" "Truck"
)

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Processing: $scene"

    # Define source and destination paths
    src="/home/salam4/trained_models/3dgs/${scene}/test/ours_30000"
    dest="/home/salam4/renders/models/3dgs/${scene}/test/${scene}"

    # Create destination directory (mkdir -p creates parent dirs and ignores if exists)
    mkdir -p "$dest/gt" "$dest/renders" "$dest/masks"

    # Copy gt and renders directories
    # Check if source exists before copying to avoid errors
    cp -a "$src/gt/." "$dest/gt/"
    cp -a "$src/renders/." "$dest/renders/"
    cp /home/salam4/trained_models/3dgs/${scene}/cfg_args /home/salam4/renders/models/3dgs/${scene}
    python /home/salam4/renders/run_laplacian_and_save_masks.py --input_dir "$dest/gt" --output_dir "$dest/masks"
done



cd

cd ./metrics

python multi_metrics.py -m /home/salam4/renders/models/3dgs/*