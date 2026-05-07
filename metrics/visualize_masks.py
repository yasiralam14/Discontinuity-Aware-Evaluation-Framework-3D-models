import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import ListedColormap

def save_laplacian_mask_visualization(mask_path: str, output_path: str):
    # Load the image array 
    mask = np.array(Image.open(mask_path))

    plt.figure(figsize=(10, 8))
    
    # Plot with grayscale colormap
    hex_colors = ['#022b3a', '#1f7a8c', '#bfdbf7', '#e1e5f2', '#ffffff']
    cmap_distinct = ListedColormap(hex_colors)
    img_plot = plt.imshow(mask, cmap=cmap_distinct)
    
    plt.title("Laplacian Pyramid Mask Visualization")
    plt.axis('off')
    
    # Save the figure and close it to free up memory
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# Usage:
save_laplacian_mask_visualization('/home/salam4/renders/models/2dgs/Barn/test/Barn/masks/000017.png', './test_visualization4.png')