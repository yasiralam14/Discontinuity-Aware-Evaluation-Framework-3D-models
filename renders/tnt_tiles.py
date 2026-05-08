import os
from pathlib import Path
from PIL import Image

def create_scene_grid(root_dir, output_path, padding=20, bg_color=(255, 255, 255)):
    """
    Reads the first ground truth image from specific scene directories and 
    stitches them into a 2x3 grid.
    
    Args:
        root_dir (str): The main directory containing the scene folders.
        output_path (str): Where to save the final stitched image.
        padding (int): The thickness of the boundary between images in pixels.
        bg_color (tuple): The RGB color of the boundary (default is white).
    """
    scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
    root_path = Path(root_dir)
    
    images = []
    target_size = None

    print(f"Scanning directory: {root_path}")

    # Step 1: Collect the first image from each scene's specific directory
    for scene in scenes:
        # Construct the path: <root>/<scene>/test/<scene>/gt/
        gt_dir = root_path / scene / 'test' / scene / 'gt'
        
        if not gt_dir.exists():
            print(f"Warning: Directory not found -> {gt_dir}")
            images.append(None)
            continue

        # Find all image files and sort them to reliably get the "first" one
        valid_exts = {'.png', '.jpg', '.jpeg'}
        image_files = sorted([f for f in gt_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
        
        if not image_files:
            print(f"Warning: No images found in -> {gt_dir}")
            images.append(None)
            continue

        # Open the first image
        try:
            img = Image.open(image_files[0]).convert('RGB')
            # Set the baseline size using the first successfully loaded image
            if target_size is None:
                target_size = img.size 
            else:
                # Resize subsequent images to match the first one to ensure the grid aligns perfectly
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            images.append(img)
            print(f"Loaded: {image_files[0].name} from {scene}")
        except Exception as e:
            print(f"Error loading image from {scene}: {e}")
            images.append(None)

    # Step 2: Create the grid canvas
    if target_size is None:
        print("Error: No valid images were found. Exiting.")
        return

    cell_w, cell_h = target_size
    cols = 3
    rows = 2

    # Calculate total dimensions including padding between tiles
    grid_w = (cols * cell_w) + ((cols - 1) * padding)
    grid_h = (rows * cell_h) + ((rows - 1) * padding)

    # Create a blank background canvas
    grid_canvas = Image.new('RGB', (grid_w, grid_h), color=bg_color)

    # Step 3: Paste each image into its correct tile position
    for idx, img in enumerate(images):
        if img is None:
            continue # Leave the tile blank (background color) if image is missing
            
        row = idx // cols
        col = idx % cols

        # Calculate x and y offsets
        pos_x = col * (cell_w + padding)
        pos_y = row * (cell_h + padding)

        grid_canvas.paste(img, (pos_x, pos_y))

    # Step 4: Save the final image
    grid_canvas.save(output_path)
    print(f"\nSuccess! Grid saved to: {output_path}")

# ==========================================
# Execution
# ==========================================

# Replace these with your actual paths
DIRECTORY_TO_SCAN = "/home/salam4/renders/models/masked_beta_splatting" 
OUTPUT_FILE = "./tnt_scenes_grid.png"

create_scene_grid(
    root_dir=DIRECTORY_TO_SCAN, 
    output_path=OUTPUT_FILE, 
    padding=15,               # Boundary thickness
    bg_color=(255, 255, 255)        # Boundary color (Black)
)