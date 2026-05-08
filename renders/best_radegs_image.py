import os
import json

def find_best_improvement(base_dir_path, scene_name):
    # Updated paths based on the correct directory structure
    path_3dgs = os.path.join(base_dir_path, "radegs", scene_name, "multi_per_view.json")
    path_masked = os.path.join(base_dir_path, "radegs_masked", scene_name, "multi_per_view.json")

    with open(path_3dgs, 'r') as f:
        data_3dgs = json.load(f)[scene_name]
        
    with open(path_masked, 'r') as f:
        data_masked = json.load(f)[scene_name]

    metrics_to_check = {
        "Masked4 SSIM": 1,   # Higher is better: improvement = masked - 3dgs
        "Masked4 PSNR": 1,   # Higher is better: improvement = masked - 3dgs
        "Masked4 LPIPS": -1  # Lower is better: improvement = 3dgs - masked
    }

    best_images = {
        "Masked4 SSIM": {"image_name": None, "improvement": float('-inf')},
        "Masked4 PSNR": {"image_name": None, "improvement": float('-inf')},
        "Masked4 LPIPS": {"image_name": None, "improvement": float('-inf')}
    }

    for metric, direction in metrics_to_check.items():
        base_vals = data_3dgs.get(metric, {})
        masked_vals = data_masked.get(metric, {})

        for img_name, m_val in masked_vals.items():
            if img_name in base_vals:
                b_val = base_vals[img_name]
                improvement = direction * (m_val - b_val)

                if improvement > best_images[metric]["improvement"]:
                    best_images[metric]["improvement"] = improvement
                    best_images[metric]["image_name"] = img_name

    return best_images

# Example usage:
result = find_best_improvement("/home/salam4/renders/models", "Courthouse")
print(result)