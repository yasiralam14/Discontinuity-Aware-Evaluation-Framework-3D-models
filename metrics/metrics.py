#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from image_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from image_utils import psnr
from argparse import ArgumentParser
from nvs.nvs_metrics import nvs_eval
import numpy as np
import traceback


def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(mask_dir / fname)
        mask = (np.array(mask) == 4).astype(np.uint8)

        # Create tensors
        r_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        g_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        m_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()


        # Append to lists
        renders.append(r_tensor)
        gts.append(g_tensor)
        masks.append(m_tensor)
        image_names.append(fname)

    return renders, gts, masks, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                mask_dir = method_dir / "masks"
                renders, gts, masks ,image_names = readImages(renders_dir, gt_dir,mask_dir)

                ssims = []
                psnrs = []
                lpipss = []
                masked_ssims = []
                masked_psnrs = []
                masked_lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    masked_ssim, masked_psnr, masked_lpips = nvs_eval(renders[idx], gts[idx], masks[idx])
                    masked_ssims.append(masked_ssim)
                    masked_psnrs.append(masked_psnr)
                    masked_lpipss.append(masked_lpips)
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  Masked SSIM : {:>12.7f}".format(torch.tensor(masked_ssims).mean(), ".5"))
                print("  Masked PSNR : {:>12.7f}".format(torch.tensor(masked_psnrs).mean(), ".5"))
                print("  Masked LPIPS: {:>12.7f}".format(torch.tensor(masked_lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "Masked SSIM": torch.tensor(masked_ssims).mean().item(),
                                                        "Masked PSNR": torch.tensor(masked_psnrs).mean().item(),
                                                        "Masked LPIPS": torch.tensor(masked_lpipss).mean().item(),
                                                        })
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "Masked SSIM": {name: ssim for ssim, name in zip(torch.tensor(masked_ssims).tolist(), image_names)},
                                                            "Masked PSNR": {name: psnr for psnr, name in zip(torch.tensor(masked_psnrs).tolist(), image_names)},
                                                            "Masked LPIPS": {name: lp for lp, name in zip(torch.tensor(masked_lpipss).tolist(), image_names)},
                                                            })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            # Optional: This prints the line number where it failed
            traceback.print_exc()
            
    flat_results = {}
    
    for scene_path, methods_dict in full_dict.items():
        scene_name = Path(scene_path).name
        # Extract the single method's metrics
        if methods_dict:
            metrics = next(iter(methods_dict.values()))
            flat_results[scene_name] = metrics

    # 2. Compute Averages over all scenes
    if flat_results:
        avg_metrics = {}
        # Use keys from the first scene to determine what to average
        metric_keys = next(iter(flat_results.values())).keys()
        
        for key in metric_keys:
            # Collect values for this metric across all scenes
            values = [d[key] for d in flat_results.values() if isinstance(d[key], (int, float))]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Add average to the end
        flat_results['average'] = avg_metrics
    if model_paths:
        output_dir = Path(model_paths[0]).parent 
        
        with open(output_dir / "all_scenes_results.json", 'w') as fp:
            json.dump(flat_results, fp, indent=True)
            

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)