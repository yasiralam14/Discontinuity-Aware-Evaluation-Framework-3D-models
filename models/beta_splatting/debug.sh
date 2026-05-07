#!/bin/bash

# Stop the script if any command fails
set -e

# --- CONFIGURATION ---
# Force script to use GPU 1
export CUDA_VISIBLE_DEVICES=1
MODEL_ROOT="/home/salam4/trained_models/beta_splatting"
TnT_DATA_ROOT="/home/salam4/3d_datasets/tnt/TNT_GOF/TrainingSet"
# ---------------------

TnT_SCENES=("Barn" "Caterpillar" "Courthouse" "Ignatius" "Meetingroom" "Truck")

# Uncomment these if you need to activate conda inside the script
# eval "$(conda shell.bash hook)"
# conda activate beta_splatting

for SCENE in "${TnT_SCENES[@]}"; do
  DATA_PATH="${TnT_DATA_ROOT}/${SCENE}"
  MODEL_PATH="${MODEL_ROOT}/${SCENE}"

  mkdir -p "${MODEL_PATH}"

  echo ""
  echo "========================================================"
  echo "STARTING SCENE: ${SCENE} on GPU 1"
  echo "Model Path: ${MODEL_PATH}"
  echo "========================================================"

  # Running without log redirection so you see errors/prompts immediately
  python train.py \
    -s "${DATA_PATH}" \
    --model_path "${MODEL_PATH}"

  echo "FINISHED SCENE: ${SCENE}"
done

echo "All scenes done."