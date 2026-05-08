import numpy as np
from PIL import Image

def scale_png_to_255(input_path, output_path):
    # Load image and convert to numpy array
    img_array = np.array(Image.open(input_path))
    print(img_array.shape)
    
    # Scale values from 0-4 to 0-255
    # (0 -> 0, 1 -> 63, 2 -> 127, 3 -> 191, 4 -> 255)
    scaled_array = (img_array / 4.0 * 255).astype(np.uint8)
    
    # Convert back to PIL Image and save
    Image.fromarray(scaled_array).save(output_path)
    
path1 = "/home/salam4/renders/test_dir/masks/DSCF5857.png"

if __name__ == "__main__":
    scale_png_to_255(path1, "/home/salam4/renders/test_dir/2dgs_0.05_DSCF5857.png")