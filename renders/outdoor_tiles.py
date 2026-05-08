import os
import math
from pathlib import Path
from PIL import Image

def create_dynamic_grid(root_dir, scenes, output_path, cols=3, padding=20, bg_color=(255, 255, 255)):
    """
    Reads ground truth images and stitches them into a grid.
    Automatically centers rows that are not completely full.
    """
    root_path = Path(root_dir)
    images = []
    target_size = None

    print(f"Scanning directory: {root_path}")

    # Step 1: Collect the first image from each scene
    for scene in scenes:
        gt_dir = root_path / scene / 'test' / scene / 'gt'
        
        if not gt_dir.exists():
            print(f"Warning: Directory not found -> {gt_dir}")
            continue

        valid_exts = {'.png', '.jpg', '.jpeg'}
        image_files = sorted([f for f in gt_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
        
        if not image_files:
            continue

        try:
            img = Image.open(image_files[0]).convert('RGB')
            if target_size is None:
                target_size = img.size 
            else:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            images.append(img)
            print(f"Loaded: {image_files[0].name} from {scene}")
        except Exception as e:
            print(f"Error loading image from {scene}: {e}")

    if not images:
        print("Error: No valid images found.")
        return

    # Step 2: Calculate Grid Dimensions
    total_images = len(images)
    rows = math.ceil(total_images / cols)
    cell_w, cell_h = target_size

    # Total canvas size
    grid_w = (cols * cell_w) + ((cols - 1) * padding)
    grid_h = (rows * cell_h) + ((rows - 1) * padding)

    grid_canvas = Image.new('RGB', (grid_w, grid_h), color=bg_color)

    # Step 3: Paste images with row-centering logic
    for r in range(rows):
        # Calculate how many images are in this specific row
        start_idx = r * cols
        end_idx = min(start_idx + cols, total_images)
        items_in_row = end_idx - start_idx
        
        # Calculate the starting X position to center the row
        row_width = (items_in_row * cell_w) + ((items_in_row - 1) * padding)
        start_x = (grid_w - row_width) // 2
        
        for c in range(items_in_row):
            img_idx = start_idx + c
            pos_x = start_x + c * (cell_w + padding)
            pos_y = r * (cell_h + padding)
            
            grid_canvas.paste(images[img_idx], (pos_x, pos_y))

    # Step 4: Save
    grid_canvas.save(output_path)
    print(f"\nSuccess! Grid saved to: {output_path}")

# ==========================================
# Execution
# ==========================================

SCENES_TO_PROCESS = ['bicycle', 'flowers', 'garden', 'stump', 'treehill']
DIRECTORY_TO_SCAN = "/home/salam4/renders/models/masked_beta_splatting" 
OUTPUT_FILE = "./outdoor_scenes_grid.png"

# Change COLUMNS to 5 for a horizontal strip, or keep it at 3 for a tiled grid.
COLUMNS = 3 

create_dynamic_grid(
    root_dir=DIRECTORY_TO_SCAN, 
    scenes=SCENES_TO_PROCESS,
    output_path=OUTPUT_FILE, 
    cols=COLUMNS,
    padding=15,               
    bg_color=(255, 255, 255)        
)