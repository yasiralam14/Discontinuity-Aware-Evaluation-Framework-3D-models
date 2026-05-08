from PIL import Image

def visualize_scene_pairs(image_paths, output_path="scenes_grid.png"):
    if len(image_paths) != 6:
        raise ValueError("Exactly 6 image paths are required (3 scenes x 2 images).")

    images = [Image.open(p) for p in image_paths]
    
    # Determine the target size to ensure uniform tiles (using the max dimensions found)
    target_width = max(img.width for img in images)
    target_height = max(img.height for img in images)
    
    # Resize all images to the target size
    images = [img.resize((target_width, target_height)) for img in images]
    
    # Group images by scene (pairs of 2)
    scenes = [(images[0], images[1]), (images[2], images[3]), (images[4], images[5])]
    
    scene_columns = []
    
    # Stack each scene's pair vertically
    for img1, img2 in scenes:
        col_img = Image.new('RGB', (target_width, target_height * 2))
        col_img.paste(img1, (0, 0))
        col_img.paste(img2, (0, target_height))
        
        scene_columns.append(col_img)

    # Stack the 3 scene columns horizontally
    final_width = target_width * 3
    final_height = target_height * 2
    
    final_img = Image.new('RGB', (final_width, final_height))
    
    x_offset = 0
    for col in scene_columns:
        final_img.paste(col, (x_offset, 0))
        x_offset += target_width

    final_img.save(output_path)
    return output_path

# Example usage:
paths = [
    "/home/salam4/renders/models/radegs/bicycle/test/bicycle/renders/_DSC8768.png", 
    "/home/salam4/renders/models/radegs_masked/bicycle/test/bicycle/renders/_DSC8768.png", 
    "/home/salam4/renders/models/radegs/Ignatius/test/Ignatius/renders/000209.png", 
    "/home/salam4/renders/models/radegs_masked/Ignatius/test/Ignatius/renders/000209.png", 
    "/home/salam4/renders/models/radegs/kitchen/test/kitchen/renders/DSCF0656.png", 
    "/home/salam4/renders/models/radegs_masked/kitchen/test/kitchen/renders/DSCF0656.png"
]

visualize_scene_pairs(paths, "radegs_scenes.png")