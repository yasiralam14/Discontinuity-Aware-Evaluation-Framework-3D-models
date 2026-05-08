import os
from PIL import Image

def get_image_dimensions(base_dir):
    for model in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path): 
            continue
        
        for scene in os.listdir(model_path):
            scene_path = os.path.join(model_path, scene)
            if not os.path.isdir(scene_path): 
                continue
            
            # Target path: .../model_name/scene_name/test/scene_name/gt
            gt_path = os.path.join(scene_path, 'test', scene, 'gt')
            if not os.path.isdir(gt_path): 
                continue
            
            # Find and process the first valid image
            for file in os.listdir(gt_path):
                file_path = os.path.join(gt_path, file)
                if os.path.isfile(file_path):
                    try:
                        with Image.open(file_path) as img:
                            print(f"{model} {scene} {img.width}x{img.height}")
                            break
                    except Exception:
                        continue

# Example Usage:
get_image_dimensions('/home/salam4/renders/models')