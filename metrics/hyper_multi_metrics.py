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
    masks1 = []
    masks2 = []
    masks3 = []
    masks4 = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(mask_dir / fname)
        mask1 = (np.array(mask) == 1).astype(np.uint8)
        mask2 = (np.array(mask) == 2).astype(np.uint8)
        mask3 = (np.array(mask) == 3).astype(np.uint8)
        mask4 = (np.array(mask) == 4).astype(np.uint8)

        # Create tensors
        r_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        g_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        m1_tensor = torch.from_numpy(mask1).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
        m2_tensor = torch.from_numpy(mask2).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
        m3_tensor = torch.from_numpy(mask3).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
        m4_tensor = torch.from_numpy(mask4).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()


        # Append to lists
        renders.append(r_tensor)
        gts.append(g_tensor)
        masks1.append(m1_tensor)
        masks2.append(m2_tensor)
        masks3.append(m3_tensor)
        masks4.append(m4_tensor)
        image_names.append(fname)

    return renders, gts, masks1, masks2, masks3, masks4,  image_names

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

            test_dir = Path(scene_dir)

            method = "ours"
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            gt_dir = test_dir/ "gt"
            renders_dir = test_dir / "renders"
            mask_dir = test_dir / "masks"
            renders, gts, masks1, masks2, masks3, masks4 ,image_names = readImages(renders_dir, gt_dir,mask_dir)

            ssims = []
            psnrs = []
            lpipss = []
            masked1_ssims = []
            masked1_psnrs = []
            masked1_lpipss = []
            masked2_ssims = []
            masked2_psnrs = []
            masked2_lpipss = []
            masked3_ssims = []
            masked3_psnrs = []
            masked3_lpipss = []
            masked4_ssims = []
            masked4_psnrs = []
            masked4_lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                masked1_ssim, masked1_psnr, masked1_lpips = nvs_eval(renders[idx], gts[idx], masks1[idx])
                masked2_ssim, masked2_psnr, masked2_lpips = nvs_eval(renders[idx], gts[idx], masks2[idx])
                masked3_ssim, masked3_psnr, masked3_lpips = nvs_eval(renders[idx], gts[idx], masks3[idx])
                masked4_ssim, masked4_psnr, masked4_lpips = nvs_eval(renders[idx], gts[idx], masks4[idx])
                masked1_ssims.append(masked1_ssim)
                masked1_psnrs.append(masked1_psnr)
                masked1_lpipss.append(masked1_lpips)
                masked2_ssims.append(masked2_ssim)
                masked2_psnrs.append(masked2_psnr)
                masked2_lpipss.append(masked2_lpips)
                masked3_ssims.append(masked3_ssim)
                masked3_psnrs.append(masked3_psnr)
                masked3_lpipss.append(masked3_lpips)
                masked4_ssims.append(masked4_ssim)
                masked4_psnrs.append(masked4_psnr)
                masked4_lpipss.append(masked4_lpips)
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))



                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "Masked1 SSIM": torch.tensor(masked1_ssims).mean().item(),
                                                        "Masked1 PSNR": torch.tensor(masked1_psnrs).mean().item(),
                                                        "Masked1 LPIPS": torch.tensor(masked1_lpipss).mean().item(),
                                                        "Masked2 SSIM": torch.tensor(masked2_ssims).mean().item(),
                                                        "Masked2 PSNR": torch.tensor(masked2_psnrs).mean().item(),
                                                        "Masked2 LPIPS": torch.tensor(masked2_lpipss).mean().item(),
                                                        "Masked3 SSIM": torch.tensor(masked3_ssims).mean().item(),
                                                        "Masked3 PSNR": torch.tensor(masked3_psnrs).mean().item(),
                                                        "Masked3 LPIPS": torch.tensor(masked3_lpipss).mean().item(),
                                                        "Masked4 SSIM": torch.tensor(masked4_ssims).mean().item(),
                                                        "Masked4 PSNR": torch.tensor(masked4_psnrs).mean().item(),
                                                        "Masked4 LPIPS": torch.tensor(masked4_lpipss).mean().item(),
                                                        })
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "Masked1 SSIM": {name: ssim for ssim, name in zip(torch.tensor(masked1_ssims).tolist(), image_names)},
                                                            "Masked1 PSNR": {name: psnr for psnr, name in zip(torch.tensor(masked1_psnrs).tolist(), image_names)},
                                                            "Masked1 LPIPS": {name: lp for lp, name in zip(torch.tensor(masked1_lpipss).tolist(), image_names)},
                                                            "Masked2 SSIM": {name: ssim for ssim, name in zip(torch.tensor(masked2_ssims).tolist(), image_names)},
                                                            "Masked2 PSNR": {name: psnr for psnr, name in zip(torch.tensor(masked2_psnrs).tolist(), image_names)},
                                                            "Masked2 LPIPS": {name: lp for lp, name in zip(torch.tensor(masked2_lpipss).tolist(), image_names)},
                                                            "Masked3 SSIM": {name: ssim for ssim, name in zip(torch.tensor(masked3_ssims).tolist(), image_names)},
                                                            "Masked3 PSNR": {name: psnr for psnr, name in zip(torch.tensor(masked3_psnrs).tolist(), image_names)},
                                                            "Masked3 LPIPS": {name: lp for lp, name in zip(torch.tensor(masked3_lpipss).tolist(), image_names)},
                                                            "Masked4 SSIM": {name: ssim for ssim, name in zip(torch.tensor(masked4_ssims).tolist(), image_names)},
                                                            "Masked4 PSNR": {name: psnr for psnr, name in zip(torch.tensor(masked4_psnrs).tolist(), image_names)},
                                                            "Masked4 LPIPS": {name: lp for lp, name in zip(torch.tensor(masked4_lpipss).tolist(), image_names)},
                                                            })

            with open(scene_dir + "/multi_results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/multi_per_view.json", 'w') as fp:
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
        
        with open(output_dir / "multi_all_scenes_results.json", 'w') as fp:
            json.dump(flat_results, fp, indent=True)
            

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)