import torch
from torchvision.utils import save_image

def create_and_save_test_images():
    # 1. Solid Blue Image
    blue_image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    blue_image[:, 2, :, :] = 1.0  

    # 2. Four Squares Image
    squares_image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    
    # Top-Right: White 
    squares_image[:, :, 0:16, 16:32] = 1.0
    # Bottom-Left: Red 
    squares_image[:, 0, 16:32, 0:16] = 1.0  
    # Bottom-Right: Purple 
    squares_image[:, 0, 16:32, 16:32] = 1.0  
    squares_image[:, 2, 16:32, 16:32] = 1.0  

    # Save locally
    save_image(blue_image, "blue_image.png")
    save_image(squares_image, "squares_image.png")

    return blue_image, squares_image

blue_img, squares_img = create_and_save_test_images()