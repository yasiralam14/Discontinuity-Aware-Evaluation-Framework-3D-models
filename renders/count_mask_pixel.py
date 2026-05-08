import os
import numpy as np
from PIL import Image

def calculate_average_mask_percentages(base_dir):
    for scene_name in os.listdir(base_dir):
        scene_path = os.path.join(base_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue
        
        masks_dir = os.path.join(scene_path, "test", scene_name, "masks")
        if not os.path.exists(masks_dir):
            continue

        print(f"Scene: {scene_name}")
        
        total_pixels_scene = 0
        value_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        for img_file in os.listdir(masks_dir):
            if not img_file.lower().endswith('.png'):
                continue
                
            img_path = os.path.join(masks_dir, img_file)
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                
                if len(img_array.shape) > 2:
                    img_array = img_array[:, :, 0]
                    
                total_pixels_scene += img_array.size
                
                for val in range(5):
                    value_counts[val] += np.sum(img_array == val)
                    
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")

        if total_pixels_scene > 0:
            for val in range(5):
                percentage = (value_counts[val] / total_pixels_scene) * 100
                print(f"  Average Value {val}: {percentage:.2f}%")
        else:
            print("  No valid mask images found to process.")

# Example usage:
calculate_average_mask_percentages("/home/salam4/renders/models/3dgs")