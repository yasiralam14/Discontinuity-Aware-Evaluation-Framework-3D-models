import numpy as np
from PIL import Image

def visualize_mask_regions(image_path, output_path="regions_combined.png", border_size=10):
    img = Image.open(image_path)
    img_array = np.array(img)

    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]

    region_images = []
    
    for val in range(1, 5):
        mask = (img_array != val).astype(np.uint8) * 255
        region_images.append(Image.fromarray(mask, mode='L'))

    width, height = img.size
    
    # Add border width to the total width calculation (3 boundaries for 4 images)
    total_width = (width * 4) + (border_size * 3)

    # Create image with a gray background (color=128) to act as the visible boundary
    combined_img = Image.new('L', (total_width, height), color=128)

    for i, region_img in enumerate(region_images):
        x_offset = i * (width + border_size)
        combined_img.paste(region_img, (x_offset, 0))

    combined_img.save(output_path)
    return output_path

# Example usage:
visualize_mask_regions("/home/salam4/renders/models/2dgs/Barn/test/Barn/masks/000001.png", "Barn_mask.png")