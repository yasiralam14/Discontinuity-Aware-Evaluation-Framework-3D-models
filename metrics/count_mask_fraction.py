import json
import numpy as np
from pathlib import Path
from PIL import Image

def compute_mask_fractions(root_dir, output_path):
    root_path = Path(root_dir)
    results = {}

    for scene_dir in root_path.iterdir():
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        masks_dir = scene_dir / "test" / scene_name / "masks"
        
        if not masks_dir.exists():
            continue

        fractions_sum = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        image_count = 0

        # Iterate through all images in the masks directory
        for img_path in masks_dir.glob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                continue

            try:
                img_arr = np.array(Image.open(img_path))
                total_pixels = img_arr.size
                
                if total_pixels == 0:
                    continue

                # Calculate the fraction of each value for the current image
                for val in [1, 2, 3, 4]:
                    fractions_sum[val] += np.sum(img_arr == val) / total_pixels
                    
                image_count += 1
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

        # Average the fractions across all images in the scene directory
        if image_count > 0:
            results[scene_name] = {
                f"value {val}": fractions_sum[val] / image_count 
                for val in [1, 2, 3, 4]
            }

    # Write the dictionary to a JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

# Usage example:
compute_mask_fractions('/home/salam4/renders/models/2dgs', '/home/salam4/renders/pixel_counts.json')