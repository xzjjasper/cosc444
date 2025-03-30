import os
import cv2
import numpy as np

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PRED_DIR = os.path.join(BASE_DIR, "output", "segmented_masks")
GT_DIR = os.path.join(BASE_DIR, "data", "PennFudanPed", "PedMasksConverted")

def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    return intersection / union if union > 0 else 0

ious = []

for filename in sorted(os.listdir(PRED_DIR)):
    if not filename.endswith("_mask.png"):
        continue

    pred_path = os.path.join(PRED_DIR, filename)
    gt_path = os.path.join(GT_DIR, filename)

    if not os.path.exists(gt_path):
        print(f"No ground truth for {filename}")
        continue

    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    iou = compute_iou(pred_mask, gt_mask)
    ious.append(iou)
    print(f"{filename}: IoU = {iou:.4f}")

# Summary
if ious:
    print(f"\nAverage IoU over {len(ious)} instances: {np.mean(ious):.4f}")
else:
    print("No IoUs computed.")
