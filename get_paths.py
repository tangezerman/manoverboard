import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os 

classes = ['cross over', 'drill through', 'one foot above second',
           'sit', 'two feet off the ground', 'two feet on first']
base_path_str = "."
base_path = Path(base_path_str)

splits = ["train", "test", "val"]


def get_paths_to_csv(base_path, output_csv="dataset_paths.csv", mode=["images", "labels"]):
    """
    Get file paths for images and labels and write them to a CSV file with debug information.
    """
    data_rows = []

    print(f"Searching in base path: {base_path.absolute()}")

    for split in splits:
        # Updated paths to look directly in split directories
        img_dir = base_path / split / "images"  # Changed order to split/images
        label_dir = base_path / split / "labels"  # Changed order to split/labels

        print(f"\nChecking split: {split}")
        print(f"Image directory exists: {img_dir.exists()}")
        print(f"Image directory path: {img_dir}")
        print(f"Label directory exists: {label_dir.exists()}")
        print(f"Label directory path: {label_dir}")

        # Get all image files
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            current_files = list(img_dir.glob(f'*{ext}'))
            img_files.extend(current_files)
            print(f"Found {len(current_files)} files with extension {ext}")

        print(f"Total images found in {split}: {len(img_files)}")

        for img_path in sorted(img_files):
            label_path = label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                data_rows.append({
                    'split': split,
                    'image_path': str(img_path),
                    'label_path': str(label_path)
                })
            else:
                print(f"Missing label file for image: {img_path}")

    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)

    print(f"\nSummary:")
    print(f"Successfully wrote {len(data_rows)} paths to {output_csv}")
    return df


# get_paths_to_csv(base_path=base_path)



def pixel_dist(base_path, splits=['train', 'test', 'val'], save_dir='plots'):
    df = pd.read_csv("dataset_paths.csv")
    values = {}
    
    for split in tqdm(splits, desc="Processing splits"):
        b_hist = np.zeros(256)
        g_hist = np.zeros(256)
        r_hist = np.zeros(256)
        split_df = df[df['split'] == split]
        
        for image in tqdm(split_df["image_path"], desc=f"Processing {split} images", leave=False):
            # Read image in BGR format (OpenCV default)
            img = cv2.imread(image)
            
            # Calculate histograms keeping BGR order
            b_hist += np.histogram(img[:,:,0], bins=256, range=(0,256))[0]  # Blue channel
            g_hist += np.histogram(img[:,:,1], bins=256, range=(0,256))[0]  # Green channel
            r_hist += np.histogram(img[:,:,2], bins=256, range=(0,256))[0]  # Red channel
            
        # Average the histograms
        num_images = len(split_df)
        b_hist = b_hist / num_images
        g_hist = g_hist / num_images
        r_hist = r_hist / num_images
        
        # Store distributions with BGR channel labels
        values[split] = {
            'blue': b_hist,     # Channel 0
            'green': g_hist,    # Channel 1
            'red': r_hist       # Channel 2
        }
        
        plt.figure(figsize=(12, 6))
        x = np.arange(256)
        
        # Plot in BGR order to match OpenCV's format
        plt.plot(x, b_hist, color='blue', label='Blue')
        plt.plot(x, g_hist, color='green', label='Green')
        plt.plot(x, r_hist, color='red', label='Red')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Pixel Value')
        plt.ylabel('Average Frequency')
        plt.title(f'Average Pixel Distribution (BGR) - {split} split')
        plt.legend()
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{os.path.join(save_dir, split)}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return values



pixel_dist(base_path,  save_dir='distribution_plots')
    
# get_paths_to_csv(base_path=base_path)

