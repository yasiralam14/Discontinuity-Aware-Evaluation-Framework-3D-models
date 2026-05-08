#!/bin/bash
cd ~/models/radfoam/radfoam

eval "$(conda shell.bash hook)"
conda activate ht_hvae


scenes=(
  "Barn" "bicycle" "bonsai" "Caterpillar" "counter"
  "Courthouse" "flowers" "garden" "Ignatius" "kitchen"
  "Meetingroom" "room" "stump" "treehill" "Truck"
)

for scene in "${scenes[@]}"; do
  echo "Processing: $scene"

  pattern="/home/salam4/models/radfoam/radfoam/output/${scene}"*
  matches=( $pattern )

  # keep only directories
  dirs=()
  for m in "${matches[@]}"; do
    [ -d "$m" ] && dirs+=("$m")
  done

  src="${dirs[0]}/test"
  dest="/home/salam4/renders/models/radfoam/${scene}/test/${scene}"

  mkdir -p "$dest/gt" "$dest/renders" "$dest/masks"
  
  if [ -d "$src/gt" ]; then
    cp -a "$src/gt/." "$dest/gt/"
  else
    echo "Warning: missing $src/gt"
  fi

  if [ -d "$src/renders" ]; then
    cp -a "$src/renders/." "$dest/renders/"
  else
    echo "Warning: missing $src/renders"
  fi
  python /home/salam4/renders/run_laplacian_and_save_masks.py --input_dir "$dest/gt" --output_dir "$dest/masks"
done

echo "Done."
cd

cp ~/models/radfoam/radfoam/configs/*.yaml /home/salam4/renders/models/radfoam/


cd

cd ./metrics

python metrics.py -m /home/salam4/renders/models/radfoam/*

