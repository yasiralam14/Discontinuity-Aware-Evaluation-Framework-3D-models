import os
import json
import csv

def compile_results_to_csv(root_dir, output_file='/home/salam4/renders/models/consolidated_results.csv'):
    # List to hold all processed rows
    all_data = []
    
    # Set to collect all possible metric names (SSIM, PSNR, etc.) for CSV headers
    metric_headers = set()

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "all_scenes_results.json" in filenames:
            json_path = os.path.join(dirpath, "all_scenes_results.json")
            
            # Use the directory name as the Model Name
            model_name = os.path.basename(dirpath)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                # Iterate through each scene (Barn, bicycle, average, etc.)
                for scene_name, metrics in data.items():
                    row = {
                        "Model": model_name,
                        "Scene": scene_name
                    }
                    
                    # Add metrics to the row and track headers
                    for metric, value in metrics.items():
                        row[metric] = value
                        metric_headers.add(metric)
                    
                    all_data.append(row)
                    
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    # specific order for headers: Model, Scene, then alphabetical metrics
    fieldnames = ["Model", "Scene"] + sorted(list(metric_headers))

    # Write to CSV
    if all_data:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Successfully created {output_file} with {len(all_data)} rows.")
    else:
        print("No 'all_scenes_results.json' files found.")

# --- Usage ---
# Replace 'path/to/your/root_dir' with the actual path to your folder
root_directory = r'/home/salam4/renders/models' 
compile_results_to_csv(root_directory)