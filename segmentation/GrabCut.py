import cv2
import numpy as np
from PIL import Image


def refine_mask(mask):
    """
    Applies morphological operations to refine the binary mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Morphological Closing (fills small holes inside the person)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Morphological Opening (removes small noise)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened

def segment_with_grabcut(pil_image, box, iter_count=5):
    """
    Apply GrabCut + refinement segmentation to extract foreground object from image.

    Parameters:
        pil_image: PIL Image object (RGB)
        box: (x1, y1, x2, y2) bounding box
        iter_count: number of GrabCut iterations

    Returns:
        mask (binary mask as uint8)
        segmented_img (numpy array of segmented RGB image)
    """
    img = np.array(pil_image)
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    x1, y1, x2, y2 = box
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        refined = refine_mask(binary_mask * 255)
        segmented = cv2.bitwise_and(img, img, mask=refined)
        return refined, segmented    
    except Exception as e:
        print(f"GrabCut failed at box: {box}")
        return None, None
