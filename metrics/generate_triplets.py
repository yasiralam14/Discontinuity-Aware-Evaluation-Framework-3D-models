from pathlib import Path
import numpy as np
from PIL import Image

def generate_triplets(source_root: str, target_root: str):
    src_path = Path(source_root)
    tgt_path = Path(target_root)

    # Iterate through model directories
    for model_dir in src_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Iterate through scene directories
        for scene_dir in model_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            gt_dir = scene_dir / "test" /scene_dir.name/ "gt"
            renders_dir = scene_dir / "test" /scene_dir.name/ "renders"
            masks_dir = scene_dir / "test" /scene_dir.name/"masks"

            # Skip if the expected directory structure is missing
            if not (gt_dir.exists() and renders_dir.exists() and masks_dir.exists()):
                continue

            # Create target directory structure
            out_dir = tgt_path / model_dir.name / scene_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Process each ground truth image
            for gt_file in gt_dir.iterdir():
                if not gt_file.is_file():
                    continue

                # Match files by exact filename
                render_file = renders_dir / gt_file.name
                mask_file = masks_dir / gt_file.name

                if not (render_file.exists() and mask_file.exists()):
                    print(f"Missing matching render or mask for {gt_file.name}")
                    continue

                # Load images and ensure GT and Render are RGB
                gt_img = np.array(Image.open(gt_file).convert("RGB"))
                render_img = np.array(Image.open(render_file).convert("RGB"))
                
                # Load mask (assuming single channel)
                mask_np = np.array(Image.open(mask_file))

                # Create binary mask where value == 4, and scale to 255 for visualization
                binary_mask = (mask_np == 4).astype(np.uint8) * 255
                
                # Convert 1-channel mask to 3-channel RGB to allow concatenation
                vis_mask_rgb = np.stack((binary_mask,) * 3, axis=-1)

                # Concatenate images horizontally (width-wise)
                triplet = np.concatenate((gt_img, render_img, vis_mask_rgb), axis=1)

                # Save the resulting triplet
                out_file_path = out_dir / gt_file.name
                Image.fromarray(triplet).save(out_file_path)

# Execution
source_directory = "/home/salam4/renders/models"
target_directory = "/home/salam4/renders/triplets"

generate_triplets(source_directory, target_directory)