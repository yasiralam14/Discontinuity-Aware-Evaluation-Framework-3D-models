from PIL import Image

def stack_images_horizontally(image_paths, output_path):
    # Open all images
    images = [Image.open(path) for path in image_paths]
    
    # Calculate the total width and the maximum height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the calculated dimensions
    merged_image = Image.new('RGB', (total_width, max_height))
    
    # Paste each image side-by-side
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
        
    # Save the final image
    merged_image.save(output_path)

# Example usage:
paths = ["/home/salam4/renders/models/2dgs/bicycle/test/bicycle/gt/_DSC8679.png","/home/salam4/renders/models/2dgs/flowers/test/flowers/gt/_DSC9040.png","/home/salam4/renders/models/2dgs/garden/test/garden/gt/DSC07956.png","/home/salam4/renders/models/2dgs/stump/test/stump/gt/_DSC9213.png","/home/salam4/renders/models/2dgs/treehill/test/treehill/gt/_DSC8922.png"]
stack_images_horizontally(paths, "outdoor.jpg")