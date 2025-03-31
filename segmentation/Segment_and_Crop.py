import os
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import parse_csv_annotations
from GrabCut import segment_with_grabcut

# === CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data","PennFudanPed", "results_rst.csv")
IMG_DIR = os.path.join(BASE_DIR, "data", "PennFudanPed", "PNGImages")
MASK_DIR = os.path.join(BASE_DIR, "output", "segmented_masks")
CROP_DIR = os.path.join(BASE_DIR, "output", "crops")

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# === Load Annotations ===
annotations = parse_csv_annotations(CSV_PATH)

# === Main Loop ===
for image_file, boxes in annotations.items():
    image_path = os.path.join(IMG_DIR, image_file)
    if not os.path.exists(image_path):
        print(f"Missing image: {image_file}")
        continue
    
    image_name = image_file.replace(".png", "") 
    pil_image = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        mask, segmented = segment_with_grabcut(pil_image, box)
        if mask is None or np.sum(mask) == 0:
            print(f"{image_name} box {i+1} → Empty mask, skipped.")
            continue
            
        # Save mask
        mask_path = os.path.join(MASK_DIR, f"{image_name}_box{i+1}_mask.png")
        cv2.imwrite(mask_path, mask)

        # Save cropped image of the segment
        cropped_rgb = segmented[y1:y2, x1:x2]
        crop_path = os.path.join(CROP_DIR, f"{image_name}_box{i+1}.png")
        cv2.imwrite(crop_path, cropped_rgb)

    print(f" Processed: {image_name} ({len(boxes)} boxes)")
