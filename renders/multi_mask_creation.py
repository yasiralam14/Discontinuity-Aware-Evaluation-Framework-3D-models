import argparse
from pathlib import Path

# Import the function from your first file
from multi_mask_save import process_dir_save_tensor

def process_root_dir(input_root: str, output_root: str, tau: float = 0.03, device: str = "cpu"):
    in_root = Path(input_root)
    out_root = Path(output_root)

    if not in_root.is_dir():
        raise RuntimeError(f"Input root directory does not exist: {input_root}")

    for scene_dir in in_root.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        gt_dir = scene_dir / "test" / scene_name / "gt"
        
        if not gt_dir.exists() or not gt_dir.is_dir():
            print(f"Skipping {scene_name}: GT directory not found at {gt_dir}")
            continue

        scene_out_dir = out_root / scene_name
        print(f"Processing scene: {scene_name} ...")
        
        # Call the imported function with the correct paths
        process_dir_save_tensor(str(gt_dir), str(scene_out_dir), tau, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process scenes into 3D Laplacian mask tensors.")
    
    parser.add_argument("--input_root", type=str, required=True, help="Root directory containing scene folders")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to save output scene folders")
    parser.add_argument("--tau", type=float, default=0.03, help="Threshold value (default: 0.03)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu' or 'cuda')")

    args = parser.parse_args()

    process_root_dir(
        input_root=args.input_root, 
        output_root=args.output_root, 
        tau=args.tau, 
        device=args.device
    )