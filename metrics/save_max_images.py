import os
import json
import cv2
import numpy as np
import argparse

def find_max_diff_file(data, metric_base):
    """
    Finds the filename with the maximum absolute difference between 
    the base metric and its masked version.
    
    Returns:
        tuple: (filename, base_value, masked_value, difference)
    """
    masked_metric = f"Masked {metric_base}"
    
    if metric_base not in data or masked_metric not in data:
        return None, 0, 0, 0

    max_diff = -1.0
    target_file = None
    best_val = 0.0
    best_masked_val = 0.0

    for filename, value in data[metric_base].items():
        if filename in data[masked_metric]:
            masked_value = data[masked_metric][filename]
            diff = value - masked_value
            if metric_base == "LPIPS":
                diff = masked_value - value    
            if diff > max_diff:
                max_diff = diff
                target_file = filename
                best_val = value
                best_masked_val = masked_value
                
    return target_file, best_val, best_masked_val, max_diff

def create_side_by_side(src_root, category, scene, filename, metric_name, 
                        val, masked_val, diff, dest_dir):
    """
    Loads GT and Render images, concatenates them, adds a caption, and saves.
    """
    if not filename:
        return

    # Construct paths
    base_path = os.path.join(src_root, category, scene, "test", scene)
    gt_path = os.path.join(base_path, "gt", filename)
    render_path = os.path.join(base_path, "renders", filename)

    # Load images
    img_gt = cv2.imread(gt_path)
    img_render = cv2.imread(render_path)

    if img_gt is None or img_render is None:
        print(f"Warning: Could not load images for {category}/{scene}/{filename}")
        return

    # Resize if dimensions differ
    if img_gt.shape != img_render.shape:
        img_render = cv2.resize(img_render, (img_gt.shape[1], img_gt.shape[0]))

    # Combine images side by side
    combined = np.hstack((img_gt, img_render))

    # --- Add Caption ---
    # Prepare text
    caption = f"{metric_name}: {val:.4f} | Masked: {masked_val:.4f} | Diff: {diff:.4f}"
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255) # White
    bg_color = (0, 0, 0)         # Black background
    
    # Calculate text size to position it
    (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, font_thickness)
    
    # Position: Top-left corner with some padding
    x_pos = 10
    y_pos = 30
    padding = 5

    # Draw black rectangle background for readability
    cv2.rectangle(combined, 
                  (x_pos - padding, y_pos - text_h - padding), 
                  (x_pos + text_w + padding, y_pos + baseline + padding), 
                  bg_color, 
                  -1) # -1 fills the rectangle

    # Draw text
    cv2.putText(combined, caption, (x_pos, y_pos), font, font_scale, text_color, font_thickness)

    # -------------------

    # Construct output name: category_scene_metric.png
    out_name = f"{category}_{scene}_{metric_name.lower()}_{filename}.png"
    out_path = os.path.join(dest_dir, out_name)

    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")

def process_directories(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 1. Iterate through each sub dir (Category)
    for category in os.listdir(src_dir):
        cat_path = os.path.join(src_dir, category)
        if not os.path.isdir(cat_path):
            continue

        # 2. Iterate through each sub sub dir (Scene)
        for scene in os.listdir(cat_path):
            scene_path = os.path.join(cat_path, scene)
            if not os.path.isdir(scene_path):
                continue

            json_path = os.path.join(scene_path, "per_view.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        full_data = json.load(f)
                        # Extract inner dictionary based on scene name
                        metrics_data = full_data.get(scene, full_data.get(list(full_data.keys())[0]))

                        # 3. Find files for max difference in metrics
                        metrics_to_check = ["SSIM", "PSNR", "LPIPS"]
                        
                        for metric in metrics_to_check:
                            # Unpack all 4 values returned by the updated function
                            target_file, val, masked_val, diff = find_max_diff_file(metrics_data, metric)
                            
                            # 4. Create and save side-by-side image with caption
                            create_side_by_side(
                                src_dir, category, scene, target_file, metric, 
                                val, masked_val, diff, dest_dir
                            )

                    except Exception as e:
                        print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    SOURCE_DIR = "/home/salam4/renders/models"
    DEST_DIR = "/home/salam4/renders/max_images"
    
    process_directories(SOURCE_DIR, DEST_DIR)