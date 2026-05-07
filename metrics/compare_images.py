import os
import numpy as np
from PIL import Image

def compare_directories(dir1, dir2):
    subdirs = ['gt', 'renders', 'masks']
    all_match = True

    for subdir in subdirs:
        path1 = os.path.join(dir1, subdir)
        path2 = os.path.join(dir2, subdir)
        
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"Directory missing: {subdir} in one or both paths.")
            all_match = False
            continue

        filenames = os.listdir(path1)
        for filename in filenames:
            file1 = os.path.join(path1, filename)
            file2 = os.path.join(path2, filename)

            if not os.path.exists(file2):
                print(f"Missing file in second directory: {subdir}/{filename}")
                all_match = False
                continue

            # Load images and convert to numpy arrays
            img1 = np.array(Image.open(file1))
            img2 = np.array(Image.open(file2))

            # Compare shapes and pixel values
            if img1.shape != img2.shape or not np.array_equal(img1, img2):
                print(f"Mismatch found in: {subdir}/{filename}")
                all_match = False

    if all_match:
        print("All corresponding images are exactly the same.")
    else:
        print("Differences were found.")

if __name__ == "__main__":
    dir1_path = "/home/salam4/trained_models/radegs_masked/counter_2/test/ours_30000"
    dir2_path = "/home/salam4/renders/models/radegs/counter/test/counter"
    
    compare_directories(dir1_path, dir2_path)