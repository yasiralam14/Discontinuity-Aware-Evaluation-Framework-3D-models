import os
import json
import csv

def compile_metrics_to_csv(input_root, output_file):
    tnt_scenes = {'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'}
    metrics = ['SSIM', 'PSNR', 'LPIPS']
    
    headers = [
        'Model', 'Dataset', 'Scene', 'Metric', 
        'Standard Value', 'Masked1 Value', 'Masked2 Value', 
        'Masked3 Value', 'Masked4 Value', 
        'Difference1', 'Difference2', 'Difference3', 'Difference4'
    ]

    csv_rows = []

    # Iterate through model directories
    for model_name in os.listdir(input_root):
        model_dir = os.path.join(input_root, model_name)
        if not os.path.isdir(model_dir):
            print(f"skipped {model_dir}")
            continue
            
        # Iterate through scene directories
        for scene_name in os.listdir(model_dir):
            scene_dir = os.path.join(model_dir, scene_name)
            if not os.path.isdir(scene_dir):
                print(f"skipped {scene_dir}")
                continue
                
            json_path = os.path.join(scene_dir, 'multi_results.json')
            if not os.path.exists(json_path):
                print(f"skipped {json_path}")
                continue

            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"skipped could not open {json_path}")
                    continue
            
            if not data:
                print(f"empty{json_path}")
                continue
                
            # Extract the inner dictionary (e.g., the "Barn" object)
            scene_data = list(data.values())[0]
            dataset_name = 'TnT' if scene_name in tnt_scenes else 'MP360'

            for metric in metrics:
                if metric not in scene_data:
                    continue
                    
                std_val = scene_data.get(metric, 0.0)
                m1 = scene_data.get(f'Masked1 {metric}', 0.0)
                m2 = scene_data.get(f'Masked2 {metric}', 0.0)
                m3 = scene_data.get(f'Masked3 {metric}', 0.0)
                m4 = scene_data.get(f'Masked4 {metric}', 0.0)

                if metric == 'LPIPS':
                    d1, d2, d3, d4 = (m1 - std_val), (m2 - std_val), (m3 - std_val), (m4 - std_val)
                else:
                    d1, d2, d3, d4 = (std_val - m1), (std_val - m2), (std_val - m3), (std_val - m4)

                csv_rows.append([
                    model_name, dataset_name, scene_name, metric,
                    std_val, m1, m2, m3, m4,
                    d1, d2, d3, d4
                ])

    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

# Usage example:
compile_metrics_to_csv('/home/salam4/renders/models', '/home/salam4/renders/models/grouped_multi.csv')