import os
import cv2
import numpy as np
import random
from collections import defaultdict

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "PennFudanPed", "PNGImages")
MASK_DIR = os.path.join(BASE_DIR, "output", "segmented_masks")
# Clothes descriptor
# LABELS_PATH = os.path.join(BASE_DIR, "output", "cluster_labels.npy")
# Body size descriptor
LABELS_PATH = os.path.join(BASE_DIR, "output", "cluster_labels_bodySize.npy")

# Clothes descriptor
# OUT_DIR = os.path.join(BASE_DIR, "output", "visualClotheDescriptor")
# Body size descriptor
OUT_DIR = os.path.join(BASE_DIR, "output", "visualBodySizeDescriptor")

os.makedirs(OUT_DIR, exist_ok=True)

# === Load cluster labels ===
cluster_data = np.load(LABELS_PATH, allow_pickle=True)
cluster_map = {row[0]: int(row[1]) for row in cluster_data}

# === Assign random colors to each cluster ===
unique_clusters = sorted(set(cluster_map.values()))
cluster_colors = {
    cid: (
        random.randint(60, 255),
        random.randint(60, 255),
        random.randint(60, 255)
    )
    for cid in unique_clusters
}

# === Group instances by image ===
image_instances = defaultdict(list)
for instance_id in cluster_map:
    image_name = instance_id.split("_box")[0]
    image_instances[image_name].append(instance_id)

# === Main Loop ===
for image_name in sorted(image_instances.keys()):
    img_path = os.path.join(IMG_DIR, f"{image_name}.png")
    if not os.path.exists(img_path):
        print(f"Missing image: {image_name}")
        continue

    image = cv2.imread(img_path)
    overlay = image.copy()

    for instance_id in image_instances[image_name]:
        mask_filename = instance_id if instance_id.endswith("_mask") else f"{instance_id}_mask"
        mask_path = os.path.join(MASK_DIR, f"{mask_filename}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or np.count_nonzero(mask) == 0:
            continue

        cluster_id = cluster_map[instance_id]
        color = cluster_colors[cluster_id]

        # Draw filled semi-transparent mask
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

        # Optional: Draw contour for clean boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Get mask bounding box for label placement
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
        x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))

        label = f"Person {cluster_id}"
        cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join(OUT_DIR, f"{image_name}_clustered.png")
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")
