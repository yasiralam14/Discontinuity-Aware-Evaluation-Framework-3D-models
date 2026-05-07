import os
import json
import csv

def compile_per_view_metrics_to_csv(input_root, output_file):
    tnt_scenes = {'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'}
    metrics = ['SSIM', 'PSNR', 'LPIPS']
    
    headers = [
        'Model', 'Dataset', 'Scene', 'file_name', 'Metric', 
        'Standard Value', 'Masked1 Value', 'Masked2 Value', 
        'Masked3 Value', 'Masked4 Value', 
        'Difference1', 'Difference2', 'Difference3', 'Difference4'
    ]

    csv_rows = []

    for model_name in os.listdir(input_root):
        model_dir = os.path.join(input_root, model_name)
        if not os.path.isdir(model_dir):
            print(f"Skipping: '{model_dir}' is not a directory.")
            continue
            
        for scene_name in os.listdir(model_dir):
            scene_dir = os.path.join(model_dir, scene_name)
            if not os.path.isdir(scene_dir):
                print(f"Skipping: '{scene_dir}' is not a directory.")
                continue
                
            json_path = os.path.join(scene_dir, 'multi_per_view.json')
            if not os.path.exists(json_path):
                print(f"Skipping: 'multi_per_view.json' not found in '{scene_dir}'.")
                continue

            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping: Failed to decode JSON in '{json_path}'.")
                    continue
            
            if not data:
                print(f"Skipping: JSON data is empty in '{json_path}'.")
                continue
                
            scene_data = list(data.values())[0]
            dataset_name = 'TnT' if scene_name in tnt_scenes else 'MP360'

            for metric in metrics:
                if metric not in scene_data:
                    print(f"Skipping metric: '{metric}' not found in '{json_path}'.")
                    continue
                    
                for img_file, std_val in scene_data[metric].items():
                    constructed_file_name = f"{model_name}_{scene_name}_{img_file}"
                    
                    m1 = scene_data.get(f'Masked1 {metric}', {}).get(img_file, 0.0)
                    m2 = scene_data.get(f'Masked2 {metric}', {}).get(img_file, 0.0)
                    m3 = scene_data.get(f'Masked3 {metric}', {}).get(img_file, 0.0)
                    m4 = scene_data.get(f'Masked4 {metric}', {}).get(img_file, 0.0)

                    if metric == 'LPIPS':
                        d1, d2, d3, d4 = (m1 - std_val), (m2 - std_val), (m3 - std_val), (m4 - std_val)
                    else:
                        d1, d2, d3, d4 = (std_val - m1), (std_val - m2), (std_val - m3), (std_val - m4)

                    csv_rows.append([
                        model_name, dataset_name, scene_name, constructed_file_name, metric,
                        std_val, m1, m2, m3, m4,
                        d1, d2, d3, d4
                    ])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

# Usage
compile_per_view_metrics_to_csv('/home/salam4/renders/models', '/home/salam4/renders/models/multi_per_view.csv')