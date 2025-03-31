import os
import cv2
import numpy as np

# === CONFIG ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT_MASK_DIR = os.path.join(BASE_DIR, "data", "PennFudanPed", "PedMasks")
OUT_DIR = os.path.join(BASE_DIR, "data", "PennFudanPed", "PedMasksConverted")
os.makedirs(OUT_DIR, exist_ok=True)

def convert_to_instance_masks():
    for filename in os.listdir(GT_MASK_DIR):
        if not filename.endswith(".png"):
            continue

        path = os.path.join(GT_MASK_DIR, filename)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Ignore background (0), extract unique labels (1,2,...)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]

        base_name = filename.replace("_mask.png", "").replace(".png", "")
        for idx, instance_id in enumerate(instance_ids, start=1):
            instance_mask = (mask == instance_id).astype(np.uint8) * 255
            out_path = os.path.join(OUT_DIR, f"{base_name}_box{idx}_mask.png")
            cv2.imwrite(out_path, instance_mask)
            print(f"Saved {out_path}")

    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_instance_masks()