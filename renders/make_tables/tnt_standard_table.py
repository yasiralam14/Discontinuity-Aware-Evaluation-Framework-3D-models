import os
import json

def format_metrics(metrics):
    if not metrics:
        return "- / - / -"
    return f"{metrics['SSIM']:0.3f} / {metrics['PSNR']:.2f} / {metrics['LPIPS']:0.3f}"

def generate_latex_table(base_dir):
    # Modified to exclude models with "masked" in the name
    models = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "masked" not in d.lower()])
    target_scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    data = {m: {} for m in models}
    
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
                        if scene in res:
                            data[model][scene] = {
                                "SSIM": res[scene].get("SSIM", 0.0),
                                "PSNR": res[scene].get("PSNR", 0.0),
                                "LPIPS": res[scene].get("LPIPS", 0.0)
                            }
        
        # Calculate the average metrics for the model
        valid_scenes = [data[model][s] for s in target_scenes if s in data[model]]
        if valid_scenes:
            data[model]["Average"] = {
                "SSIM": sum(s["SSIM"] for s in valid_scenes) / len(valid_scenes),
                "PSNR": sum(s["PSNR"] for s in valid_scenes) / len(valid_scenes),
                "LPIPS": sum(s["LPIPS"] for s in valid_scenes) / len(valid_scenes)
            }

    # Modified max columns based on chunk size (up to 4)
    max_chunk_size = min(4, len(models)) if models else 4
    col_format = "c|" + "||".join(["c"] * max_chunk_size)
    
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption{Quantitative results on the \\textbf{TnT} dataset. Evaluated on SSIM, PSNR, and LPIPS.}\\vspace{-10pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\label{tab:tnt_results}",
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
        model_row = ["\\textbf{Scenes}"] + [f"\\textbf{{{m}}}" for m in model_names] + [""] * padding_needed
        lines.append("    " + " & ".join(model_row) + " \\\\")
        
        lines.append("  \\hline")
        
        # Scene Data Rows
        for scene in target_scenes:
            row = [f"\\textbf{{{scene}}}"]
            for model in chunk:
                row.append(format_metrics(data[model].get(scene)))
            row += [""] * padding_needed
            lines.append("    " + " & ".join(row) + " \\\\")
            
        # Average Data Row
        lines.append("  \\hline")
        avg_row = ["\\textbf{Average}"]
        for model in chunk:
            avg_row.append(format_metrics(data[model].get("Average")))
        avg_row += [""] * padding_needed
        lines.append("    " + " & ".join(avg_row) + " \\\\")
            
    lines.extend([
        "  \\hline",
        "  \\end{tabular}",
        "  }",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    # ==========================================
    # UPDATE THESE VARIABLES BEFORE RUNNING
    # ==========================================
    TARGET_DIR = "/home/salam4/renders/models" 
    OUTPUT_FILE = "tnt_standard_table.txt"
    
    if os.path.exists(TARGET_DIR):
        latex_output = generate_latex_table(TARGET_DIR)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(latex_output)
            
        print(f"Success! The LaTeX table has been written to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print(f"Error: The directory '{TARGET_DIR}' does not exist. Please update the TARGET_DIR variable.")