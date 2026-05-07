import os
import json
import pandas as pd

# 1. SETUP: Define Scene Lists
tnt_scenes = [
    'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'
]
mp360_scenes = [
    'bicycle', 'flowers', 'garden', 'stump', 'treehill', 'bonsai', 
    'counter', 'kitchen', 'room'
]

def get_dataset(scene_name):
    # Case-insensitive matching just in case
    scene_lower = scene_name.lower()
    if scene_name in tnt_scenes or scene_lower in [s.lower() for s in tnt_scenes]:
        return 'tnt'
    elif scene_name in mp360_scenes or scene_lower in [s.lower() for s in mp360_scenes]:
        return 'mp360'
    else:
        return 'unknown'

def process_metrics(root_dir):
    all_records = []
    
    # Standard metrics we want to track
    target_metrics = ['PSNR', 'SSIM', 'LPIPS']

    # 2. ITERATE: Loop through Model directories
    # Check if directory exists first
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return

    models = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for model_name in models:
        model_path = os.path.join(root_dir, model_name)
        
        # 3. ITERATE: Loop through Scene directories inside Model
        scenes = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for scene_name in scenes:
            scene_path = os.path.join(model_path, scene_name)
            dataset = get_dataset(scene_name)
            
            # 4. LOCATE: Check for per_view.json
            json_file_path = os.path.join(scene_path, 'per_view.json')
            
            if not os.path.exists(json_file_path):
                continue # Skip if json is missing
                
            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                
                # The JSON structure starts with the Scene Name key
                # e.g., { "Barn": { ... } }
                if scene_name in data:
                    scene_data = data[scene_name]
                else:
                    # Fallback: if folder name is 'barn' but json key is 'Barn'
                    # keys() returns a list, we take the first one if it matches roughly
                    keys = list(data.keys())
                    if len(keys) > 0:
                        scene_data = data[keys[0]]
                    else:
                        continue

                # 5. PARSE: Match Standard and Masked metrics
                for metric in target_metrics:
                    masked_metric_key = f"Masked {metric}"
                    
                    # Ensure both standard and masked keys exist in the JSON
                    if metric in scene_data and masked_metric_key in scene_data:
                        standard_files = scene_data[metric]
                        masked_files = scene_data[masked_metric_key]
                        
                        # Iterate over filenames in the standard metric
                        for filename, std_val in standard_files.items():
                            
                            # Get corresponding masked value
                            mask_val = masked_files.get(filename, None)
                            
                            if mask_val is not None:
                                # Calculate Difference based on metric type
                                if metric == 'LPIPS':
                                    diff = mask_val - std_val
                                else:
                                    diff = std_val - mask_val
                                
                                # Add to records
                                all_records.append({
                                    'Model': model_name,
                                    'Dataset': dataset,
                                    'Scene': scene_name,
                                    'filename': f"{model_name}_{scene_name}_{filename}",
                                    'Metric': metric,
                                    'standard_value': std_val,
                                    'Masked_Value': mask_val,
                                    'Difference': diff
                                })
                                
            except Exception as e:
                print(f"Error processing {json_file_path}: {e}")

    # 6. EXPORT: Create DataFrame and Save
    df = pd.DataFrame(all_records)
    
    # Define column order
    cols = ['Model', 'Dataset', 'Scene', 'filename', 'Metric', 
            'standard_value', 'Masked_Value', 'Difference']
    
    if not df.empty:
        df = df[cols]
        output_filename = '/home/salam4/renders/models/consolidated_per_view.csv'
        df.to_csv(output_filename, index=False)
        print(f"Success! Processed {len(df)} rows. Saved to '{output_filename}'")
    else:
        print("No matching data found.")

# --- RUN THE FUNCTION ---
# Replace this path with your actual root directory path
root_directory = '/home/salam4/renders/models/' 
process_metrics(root_directory)