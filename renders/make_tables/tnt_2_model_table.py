import os
import json

def format_metrics(metrics):
    if not metrics or metrics.get("count", 0) == 0:
        return "- / - / -"
    return f"{metrics['SSIM']:0.4f} / {metrics['PSNR']:.2f} / {metrics['LPIPS']:0.4f}"

def generate_latex_table(base_dir):
    # Specify the two models you want to compare side-by-side
    target_models = ["3dgs", "2dgs"] 
    
    target_scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    
    regions = {
        "Standard": "",
        "Region 1": "Masked1 ",
        "Region 2": "Masked2 ",
        "Region 3": "Masked3 ",
        "Region 4": "Masked4 "
    }
    
    model_averages = {m: {r: {"SSIM": 0.0, "PSNR": 0.0, "LPIPS": 0.0, "count": 0} for r in regions} for m in target_models}
    
    # Data Gathering
    for model in target_models:
        model_path = os.path.join(base_dir, model)
        if not os.path.exists(model_path):
            continue
            
        actual_dirs = {d.lower(): d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))}
        
        for scene in target_scenes:
            scene_lower = scene.lower()
            if scene_lower in actual_dirs:
                json_path = os.path.join(model_path, actual_dirs[scene_lower], "multi_results.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        res = json.load(f)
                        if res and isinstance(res, dict):
                            scene_data = res[list(res.keys())[0]]
                            
                            for region_name, prefix in regions.items():
                                ssim_key = f"{prefix}SSIM".strip()
                                psnr_key = f"{prefix}PSNR".strip()
                                lpips_key = f"{prefix}LPIPS".strip()
                                
                                if ssim_key in scene_data:
                                    model_averages[model][region_name]["SSIM"] += scene_data[ssim_key]
                                    model_averages[model][region_name]["PSNR"] += scene_data[psnr_key]
                                    model_averages[model][region_name]["LPIPS"] += scene_data[lpips_key]
                                    model_averages[model][region_name]["count"] += 1

    # Averaging
    for model in target_models:
        for region_name in regions:
            count = model_averages[model][region_name]["count"]
            if count > 0:
                for m_key in ["SSIM", "PSNR", "LPIPS"]:
                    model_averages[model][region_name][m_key] /= count

    # LaTeX Generation
    col_format = "c|cc"
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Averaged results on TnT Scenes: Side-by-side comparison 3DGS vs 2DGS.}\\vspace{-10pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_format}}}",
        "  \\hline"
    ]
    
    # Header Row
    model_names = [m.replace('_', '\\_') for m in target_models]
    header = ["\\textbf{Regions}"] + [f"\\textbf{{{m}}}" for m in model_names]
    lines.append("    " + " & ".join(header) + " \\\\")
    
    # Metric Sub-header
    metrics_sub = [""] + ["\\small SSIM / PSNR / LPIPS"] * len(target_models)
    lines.append("    " + " & ".join(metrics_sub) + " \\\\")
    lines.append("  \\hline")
    
    # Data Rows
    for region_name in regions.keys():
        row = [f"\\textbf{{{region_name}}}"]
        for model in target_models:
            row.append(format_metrics(model_averages[model][region_name]))
        lines.append("    " + " & ".join(row) + " \\\\")
            
    lines.extend([
        "  \\hline",
        "  \\end{tabular}",
        "  }",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    TARGET_DIR = "/home/salam4/renders/models" 
    OUTPUT_FILE = "3dgs_2dgs.txt"
    
    if os.path.exists(TARGET_DIR):
        latex_output = generate_latex_table(TARGET_DIR)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(latex_output)
            
        print(f"Success! The LaTeX table has been written to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print(f"Error: The directory '{TARGET_DIR}' does not exist.")