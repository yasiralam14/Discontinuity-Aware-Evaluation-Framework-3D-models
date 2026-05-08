import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import Tensor
import os
from pathlib import Path
import argparse  

def laplacian_pyramid_th(image: Tensor, tau: float) -> Tensor:
    batch_size, _, h, w = image.shape
    levels = 4

    pyr = [image]
    for _ in range(levels):
        pyr.append(F.interpolate(pyr[-1], scale_factor=0.5, mode="bilinear", align_corners=False))

    all_diff_mask = torch.zeros((batch_size, 1, h, w), device=image.device, dtype=torch.float32)

    for l in range(levels, 0, -1):
        region_id = levels - l + 1

        up = F.interpolate(pyr[l], size=pyr[l - 1].shape[-2:], mode="bilinear", align_corners=False)
        diff = torch.abs(up - pyr[l - 1]).mean(dim=1, keepdim=True)
        diff = F.interpolate(diff, size=(h, w), mode="bilinear", align_corners=False)

        diff_mask = (diff > tau).float()

        if l == levels:
            all_diff_mask = diff_mask * region_id
        else:
            all_diff_mask = torch.where(diff_mask == 1, torch.tensor(region_id, device=image.device).float(), all_diff_mask)

    return all_diff_mask



def process_dir_save_png(input_dir: str, output_dir: str, tau: float = 0.03, device: str = "cpu"):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}

    img_paths = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix in exts])
    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    for p in img_paths:
        pil_img = Image.open(p).convert("RGB")
        tensor = TF.to_tensor(pil_img).unsqueeze(0).to(device)  

        with torch.no_grad():
            masks = laplacian_pyramid_th(tensor, tau)               
            mask_id = masks[0, 0].detach().cpu().to(torch.uint8)    

        out_path = out_dir / f"{p.stem}.png"
        Image.fromarray(mask_id.numpy(), mode="L").save(out_path)

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images into Laplacian masks.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output PNGs")
    
    parser.add_argument("--tau", type=float, default=0.05, help="Threshold value (default: 0.05)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu' or 'cuda')")

    args = parser.parse_args()

    process_dir_save_png(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        tau=args.tau, 
        device=args.device
    )
