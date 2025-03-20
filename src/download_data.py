import os
import io
from PIL import Image
import pandas as pd
from datasets import load_dataset

def save_images_from_parquet(parquet_path, target_folder):
    """
    Load parquet file using datasets, read each row's BYTES and FILENAME, 
    and save the image to the target folder with the name FILENAME.
    
    Args:
        parquet_path: Path to the input parquet file
        target_folder: Path to the target folder to save images
    """
    # Load the parquet file
    dataset = load_dataset('loooooong/Any2anyTryon_vitonhd_test', data_files=parquet_path)['train']
    
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Process each row
    for row in dataset:
        try:
            # Read image bytes and filename
            image = row['BYTES']
            filename = row['FILENAME']
            # Open image using PIL
            # image = Image.open(io.BytesIO(image_bytes))
            
            # Save image to target folder
            image.save(os.path.join(target_folder, filename))
        except Exception as e:
            print(f"Failed to process row with filename {row['FILENAME']}: {e}")

if __name__ == "__main__":
    parquet_path = "data/vitonhd_test.parquet"
    target_folder = "data/zalando-hd-resized/test/image_synthesis"
    
    save_images_from_parquet(parquet_path, target_folder)
