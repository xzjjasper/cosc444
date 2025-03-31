import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_slic(image, num_segments=100, compactness=10, sigma=1, convert_to_lab=True):
    """
    Apply SLIC algorithm for superpixel segmentation
    
    Parameters:
        image: Input image (BGR or RGB format)
        num_segments: Desired number of superpixels
        compactness: Compactness parameter, controls regularity of superpixels, higher is more regular
        sigma: Standard deviation for Gaussian pre-processing
        convert_to_lab: Whether to convert image to LAB color space (SLIC usually works better in LAB space)
    
    Returns:
        segmented_image: Segmented image, each superpixel region represented by its mean color
        labels: Superpixel labels for each pixel
    """
    # Ensure image is in BGR format (OpenCV default)
    if len(image.shape) == 3 and image.shape[2] == 3:
        if convert_to_lab:
            # Convert to LAB color space to improve segmentation quality
            image_for_slic = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            image_for_slic = image.copy()
    else:
        # If grayscale, convert to 3 channels
        image_for_slic = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if convert_to_lab:
            image_for_slic = cv2.cvtColor(image_for_slic, cv2.COLOR_BGR2LAB)
    
    # Create SLIC object
    slic = cv2.ximgproc.createSuperpixelSLIC(
        image_for_slic, 
        algorithm=cv2.ximgproc.SLIC, 
        region_size=int(np.sqrt(image.shape[0]*image.shape[1]/num_segments)),
        ruler=compactness
    )
    
    # Number of iterations
    slic.iterate(10)
    
    # Get labels
    labels = slic.getLabels()
    
    # Generate segmented image
    segmented_image = np.zeros_like(image)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Calculate average color for each superpixel region
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask.fill(0)
        mask[labels == label] = 1
        
        # Calculate mean color for this region
        if len(image.shape) == 3:  # Color image
            mean_color = [
                np.mean(image[:,:,0][labels == label]),
                np.mean(image[:,:,1][labels == label]),
                np.mean(image[:,:,2][labels == label])
            ]
            segmented_image[:,:,0][labels == label] = mean_color[0]
            segmented_image[:,:,1][labels == label] = mean_color[1]
            segmented_image[:,:,2][labels == label] = mean_color[2]
        else:  # Grayscale image
            mean_color = np.mean(image[labels == label])
            segmented_image[labels == label] = mean_color
    
    return segmented_image, labels

def visualize_superpixels(image, labels, save_path=None):
    """
    Visualize superpixel segmentation results
    
    Parameters:
        image: Original image
        labels: Segmentation labels
        save_path: Path to save results, if None no saving
    """
    # Create superpixel boundary image
    contour_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Get superpixel boundaries
    dx = np.diff(labels, axis=1)
    dy = np.diff(labels, axis=0)
    
    # Horizontal boundaries
    contour_h = np.zeros_like(dx, dtype=np.bool_)
    contour_h[dx != 0] = True
    
    # Vertical boundaries
    contour_v = np.zeros_like(dy, dtype=np.bool_)
    contour_v[dy != 0] = True
    
    # Merge boundaries
    contours = np.zeros_like(labels, dtype=np.bool_)
    contours[:-1, :] |= contour_v
    contours[:, :-1] |= contour_h
    
    # Draw boundaries
    if len(image.shape) == 3:  # Color image
        contour_mask[contours] = [0, 0, 255]  # Red boundaries
    else:  # Grayscale image
        contour_mask[contours] = 255
    
    # Overlay display
    overlay = cv2.addWeighted(image, 0.7, contour_mask, 0.3, 0)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if len(overlay.shape) == 3:
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(overlay, cmap='gray')
    plt.title(f'Superpixel Segmentation ({len(np.unique(labels))} regions)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
def preprocess_image_with_slic(image, num_segments=100, feature_extraction=False):
    """
    Preprocess image using SLIC superpixels
    
    Parameters:
        image: Input image
        num_segments: Desired number of superpixels
        feature_extraction: Whether to extract features (return superpixel features instead of image)
    
    Returns:
        If feature_extraction=False:
            Returns preprocessed image
        If feature_extraction=True:
            Returns superpixel features (average color, position, size, etc. for each superpixel)
    """
    # Apply SLIC algorithm
    segmented_image, labels = apply_slic(image, num_segments=num_segments)
    
    if not feature_extraction:
        return segmented_image
    
    # Extract superpixel features
    features = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Get mask for current superpixel
        mask = (labels == label)
        num_pixels = np.sum(mask)
        
        # Calculate position features (center coordinates of superpixel)
        y_indices, x_indices = np.where(mask)
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)
        
        # Calculate color features (average color of superpixel region)
        if len(image.shape) == 3:  # Color image
            color_feature = [
                np.mean(image[:,:,0][mask]),
                np.mean(image[:,:,1][mask]),
                np.mean(image[:,:,2][mask])
            ]
        else:  # Grayscale image
            color_feature = [np.mean(image[mask])]
        
        # Calculate shape features (area and aspect ratio)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        area = num_pixels
        aspect_ratio = width / height if height > 0 else 0
        
        # Combine features
        feature = {
            'label': label,
            'center': (center_x, center_y),
            'color': color_feature,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'bbox': (min_x, min_y, max_x, max_y),
            'mask': mask
        }
        
        features.append(feature)
    
    return features 