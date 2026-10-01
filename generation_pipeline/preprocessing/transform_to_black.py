import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

def is_grayscale(img):
    """Check if image is already grayscale."""
    if img.mode in ("L", "1"):
        return True
    if img.mode == "RGB":
        return all(r == g == b for r, g, b in img.getdata())
    return False

def process_image(filename):
    """Convert a single image to grayscale and save."""
    try:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return None  # Skip non-image files

        source_path = os.path.join(source_folder, filename)
        img = Image.open(source_path)

        if is_grayscale(img):
            return f"Skipped grayscale: {filename}"

        img_bw = img.convert('L')

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_black{ext}"
        target_path = os.path.join(target_folder, new_filename)

        img_bw.save(target_path)

        return f"Saved black-and-white: {new_filename}"
    except Exception as e:
        return f"Error with {filename}: {e}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert images to black-and-white and save a copy")
    parser.add_argument("--dataset_folder", required=True, help="Path to the main repo/folder")
    args = parser.parse_args()

    source_folder = os.path.join(args.dataset_folder, 'images')
    target_folder = os.path.join(args.dataset_folder, 'images_black')
    os.makedirs(target_folder,exist_ok=True)

    files = os.listdir(source_folder)

    print(f"Found {len(files)} files in {source_folder}")
    processes = cpu_count()

    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(process_image, files), total=len(files), desc="Processing"):
            if result:
                print(result)
