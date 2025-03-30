import os
import numpy as np
import cv2

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CROP_DIR = os.path.join(BASE_DIR, "output", "crops")
MASK_DIR = os.path.join(BASE_DIR, "output", "segmented_masks")
# Clothes descriptor
#OUT_DIR = os.path.join(BASE_DIR, "output", "cluster_clothes")
# Body size descriptor
OUT_DIR = os.path.join(BASE_DIR, "output", "cluster_body_size")

os.makedirs(OUT_DIR, exist_ok=True)

def extract_descriptors(image, mask):
    """
    Extracts interpretable features from a cropped person image:
    - height, width, aspect ratio
    """
    h, w = image.shape[:2]
    aspect_ratio = h / w if w > 0 else 0
    mask_area = np.count_nonzero(mask)
    # # Clothes descriptor
    # mean_color = np.mean(image.reshape(-1, 3), axis=0)  # BGR
    # mean_r, mean_g, mean_b = mean_color[::-1]  # convert to RGB
    # return [h, w, aspect_ratio, mean_r, mean_g, mean_b]
    
    # Body size descriptor
    area = h * w
    extent = mask_area / area if area else 0
    solidity = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area else 0

    return [h, w, aspect_ratio, mask_area, extent, solidity]

# === Loop through all crops ===
features = []
identifiers = []

for filename in sorted(os.listdir(CROP_DIR)):
    if not filename.endswith(".png") or "_mask" in filename:
        continue
    crop_path = os.path.join(CROP_DIR, filename)
    base_name = filename.replace(".png", "")
    mask_filename = f"{base_name}_mask.png"
    mask_path = os.path.join(MASK_DIR, mask_filename)
    
    img = cv2.imread(crop_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f" Failed to load {filename} or its mask")
        continue

    descriptors = extract_descriptors(img,mask)
    features.append(descriptors)
    identifiers.append(base_name)

# === Save outputs ===
np.save(os.path.join(OUT_DIR, "features.npy"), np.array(features))
np.save(os.path.join(OUT_DIR, "identifiers.npy"), np.array(identifiers))
print(f" Saved descriptors for {len(features)} instances.")
