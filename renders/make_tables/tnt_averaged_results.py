import os
import json

def generate_model_centric_table(base_dir):
    models = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and "masked" not in d.lower()])
    target_scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    
    results = []

    for model in models:
        model_path = os.path.join(base_dir, model)
        actual_dirs = {d.lower(): d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))}
        
        metrics = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "count": 0}
        
        for scene in target_scenes:
            scene_lower = scene.lower()
            if scene_lower in actual_dirs:
                json_path = os.path.join(model_path, actual_dirs[scene_lower], "multi_results.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        
                        if data and isinstance(data, dict):
                            scene_key = list(data.keys())[0]
                            scene_data = data[scene_key]
                            
                            if "PSNR" in scene_data and "SSIM" in scene_data and "LPIPS" in scene_data:
                                metrics["PSNR"] += scene_data["PSNR"]
                                metrics["SSIM"] += scene_data["SSIM"]
                                metrics["LPIPS"] += scene_data["LPIPS"]
                                metrics["count"] += 1

    
        if metrics["count"] > 0:
            avg_psnr = metrics["PSNR"] / metrics["count"]
            avg_ssim = metrics["SSIM"] / metrics["count"]
            avg_lpips = metrics["LPIPS"] / metrics["count"]
            results.append((model.replace('_', '\\_'), avg_psnr, avg_ssim, avg_lpips))
        else:
            results.append((model.replace('_', '\\_'), None, None, None))

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Average quantitative results across TnT scenes per model.}",
        "  \\label{tab:model_results}",
        "  \\begin{tabular}{l|ccc}",
        "  \\hline",
        "  \\textbf{Model} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{LPIPS} \\\\",
        "  \\hline"
    ]
    
    for model_name, psnr, ssim, lpips in results:
        if psnr is not None:
            lines.append(f"  {model_name} & {psnr:.2f} & {ssim:.4f} & {lpips:.4f} \\\\")
        else:
            lines.append(f"  {model_name} & - & - & - \\\\")
            
    lines.extend([
        "  \\hline",
        "  \\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    TARGET_DIR = "/home/salam4/renders/models" 
    OUTPUT_FILE = "tnt_model_table.txt"
    
    if os.path.exists(TARGET_DIR):
        latex_output = generate_model_centric_table(TARGET_DIR)
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(latex_output)
            
        print(f"Success! The LaTeX table has been written to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print(f"Error: The directory '{TARGET_DIR}' does not exist.")