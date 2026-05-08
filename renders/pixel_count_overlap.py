import torch
import json
import argparse
from pathlib import Path

def calculate_region_fractions(input_root: str, output_json: str):
    root = Path(input_root)
    results = {}

    if not root.is_dir():
        raise RuntimeError(f"Input root directory does not exist: {input_root}")

    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        pt_files = list(scene_dir.glob("*.pt"))
        
        if not pt_files:
            continue

        # Accumulators for the 4 regions (channels 0, 1, 2, 3)
        fractions = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        for pt_file in pt_files:
            mask = torch.load(pt_file, weights_only=True)  # Shape: (4, H, W)
            _, h, w = mask.shape
            total_pixels = h * w

            for channel_idx in range(4):
                region = channel_idx + 1
                active_pixels = mask[channel_idx].sum().item()
                fractions[region] += (active_pixels / total_pixels)

        # Average over all files in the scene
        num_files = len(pt_files)
        results[scene_name] = {
            f"Region {region}": fractions[region] / num_files 
            for region in range(1, 5)
        }
        
        print(f"Processed {scene_name}: {num_files} files")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results successfully saved to {output_json}")


calculate_region_fractions('/home/salam4/renders/multi_masks', 'pixel_count_overlap.json')