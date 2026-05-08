from PIL import Image

def combine_images_side_by_side(path1, path2, output_path="combined.png"):
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    new_img = Image.new('RGB', (total_width, max_height))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    new_img.save(output_path)
    return output_path

# Example usage:
combine_images_side_by_side("/home/salam4/renders/models/radegs/Courthouse/test/Courthouse/renders/000777.png", "/home/salam4/renders/models/radegs_masked/Courthouse/test/Courthouse/renders/000777.png", "radegs_Courthouse.png")