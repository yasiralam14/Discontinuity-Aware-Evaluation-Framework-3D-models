import os
import json

def format_metrics(metrics):
    if not metrics or metrics.get("count", 0) == 0:
        return "- / - / -"
    return f"{metrics['SSIM']:0.4f} / {metrics['PSNR']:.2f} / {metrics['LPIPS']:0.4f}"

def generate_latex_table(base_dir):
    # Exclude models with "masked" in the name
    models = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "masked" not in d.lower()])
    target_scenes = ["bonsai", "counter", "kitchen", "room"]
    
    # Define mapping from row name to JSON key prefix
    regions = {
        "Standard": "",
        "Region 1": "Masked1 ",
        "Region 2": "Masked2 ",
        "Region 3": "Masked3 ",
        "Region 4": "Masked4 "
    }
    
    # Initialize data structure to hold averages
    model_averages = {m: {r: {"SSIM": 0.0, "PSNR": 0.0, "LPIPS": 0.0, "count": 0} for r in regions} for m in models}
    
    for model in models:
        model_path = os.path.join(base_dir, model)
        actual_dirs = {d.lower(): d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))}
        
        for scene in target_scenes:
            scene_lower = scene.lower()
            if scene_lower in actual_dirs:
                actual_scene_dir = actual_dirs[scene_lower]
                json_path = os.path.join(model_path, actual_scene_dir, "multi_results.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        res = json.load(f)
                        
                        # Dynamically grab the inner dictionary (e.g., {"Barn": {...}} -> {...})
                        if res and isinstance(res, dict):
                            scene_key = list(res.keys())[0]
                            scene_data = res[scene_key]
                            
                            # Accumulate data for each region
                            for region_name, prefix in regions.items():
                                ssim_key = f"{prefix}SSIM".strip() if prefix == "" else f"{prefix}SSIM"
                                psnr_key = f"{prefix}PSNR".strip() if prefix == "" else f"{prefix}PSNR"
                                lpips_key = f"{prefix}LPIPS".strip() if prefix == "" else f"{prefix}LPIPS"
                                
                                if ssim_key in scene_data:
                                    model_averages[model][region_name]["SSIM"] += scene_data[ssim_key]
                                    model_averages[model][region_name]["PSNR"] += scene_data[psnr_key]
                                    model_averages[model][region_name]["LPIPS"] += scene_data[lpips_key]
                                    model_averages[model][region_name]["count"] += 1

    # Calculate averages for this model
    for model in models:
        for region_name in regions:
            count = model_averages[model][region_name]["count"]
            if count > 0:
                model_averages[model][region_name]["SSIM"] /= count
                model_averages[model][region_name]["PSNR"] /= count
                model_averages[model][region_name]["LPIPS"] /= count

    # Determine max columns based on chunk size (set to 4)
    max_chunk_size = min(4, len(models)) if models else 4
    col_format = "c|" + "||".join(["c"] * max_chunk_size)
    
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption{Averaged quantitative results across MP360 Indoor scenes by region. Evaluated on SSIM, PSNR, and LPIPS.}\\vspace{-10pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\label{tab:region_results}",
        f"  \\begin{{tabular}}{{{col_format}}}"
    ]
    
    # Process models in chunks of 4
    for i in range(0, len(models), 4):
        chunk = models[i:i+4]
        padding_needed = max_chunk_size - len(chunk)
        
        lines.append("  \\hline")
        
        # Metrics Header Row
        metrics_row = [""] + ["\\textbf{SSIM} / \\textbf{PSNR} / \\textbf{LPIPS}"] * len(chunk) + [""] * padding_needed
        lines.append("    " + " & ".join(metrics_row) + " \\\\")
        
        lines.append("  \\hline")
        
        # Models Header Row
        model_names = [m.replace('_', '\\_') for m in chunk]
        model_row = ["\\textbf{Regions}"] + [f"\\textbf{{{m}}}" for m in model_names] + [""] * padding_needed
        lines.append("    " + " & ".join(model_row) + " \\\\")
        
        lines.append("  \\hline")
        
        # Region Data Rows
        for region_name in regions.keys():
            row = [f"\\textbf{{{region_name}}}"]
            for model in chunk:
                row.append(format_metrics(model_averages[model][region_name]))
            row += [""] * padding_needed
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
    OUTPUT_FILE = "indoor_region_table.txt"
    
    if os.path.exists(TARGET_DIR):
        latex_output = generate_latex_table(TARGET_DIR)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(latex_output)
            
        print(f"Success! The LaTeX table has been written to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print(f"Error: The directory '{TARGET_DIR}' does not exist.")