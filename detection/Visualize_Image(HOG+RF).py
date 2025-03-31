import os
import cv2
import numpy as np
import joblib
import csv
import Sliding as sd
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

# Allow duplicate lib loading (if needed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Parameters for detection
size = (64, 128)  # Detection window size
step_size = (9, 9)  # Sliding window step size
downscale = 1.25  # Pyramid downscale factor

# Folder with images and output CSV file name (updated to reflect Random Forest)
images_folder = "PNGImages"
output_csv = "Annotations (HOG+RF).csv"

# Load the trained model (if it's a tuple, extract the first element)
loaded_data = joblib.load('models/models_RF.dat')
model = loaded_data[0] if isinstance(loaded_data, tuple) else loaded_data

# List to hold CSV rows and initialize annotation counter
annotations = []
annotation_id = 1

# Loop over each file in the images folder
for image_file in os.listdir(images_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Resize image (adjust size as needed)
        image = cv2.resize(image, (400, 256))

        # List to store detections for this image
        detections = []
        scale = 0

        # Loop over the image pyramid
        for im_scaled in pyramid_gaussian(image, downscale=downscale):
            # Break if the image is smaller than the detection window
            if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
                break

            # Slide a window across the current pyramid level
            for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
                # Ensure the window has the desired size
                if window.shape[0] != size[1] or window.shape[1] != size[0]:
                    continue

                # Convert window from float64 [0,1] to uint8 [0,255]
                window_uint8 = cv2.convertScaleAbs(window, alpha=255.0)
                # Convert to grayscale for HOG feature extraction
                window_gray = cv2.cvtColor(window_uint8, cv2.COLOR_BGR2GRAY)

                # Compute HOG features
                fd = hog(window_gray, orientations=9, pixels_per_cell=(8, 8),
                         visualize=False, cells_per_block=(3, 3))
                fd = fd.reshape(1, -1)

                # Predict using the Random Forest classifier
                pred = model.predict(fd)
                # Use predict_proba to get the probability for class 1 (positive)
                score = model.predict_proba(fd)[0, 1]

                # If detection is positive and probability is high enough, store it
                if pred == 1 and score > 0.5:
                    detections.append((
                        int(x * (downscale ** scale)),
                        int(y * (downscale ** scale)),
                        score,
                        int(size[0] * (downscale ** scale)),
                        int(size[1] * (downscale ** scale))
                    ))
            scale += 1

        # If detections were made, apply non-maxima suppression
        if detections:
            # Create rectangles from detections: [x, y, x+w, y+h]
            rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
            # Extract scores for each detection
            scores = np.array([score for (x, y, score, w, h) in detections])
            # Apply non-max suppression with overlap threshold 0.1
            picks = non_max_suppression(rects, probs=scores, overlapThresh=0.1)

            # For each final detection, append a row to annotations list
            for (x1, y1, x2, y2) in picks:
                annotations.append([image_file, annotation_id, x1, y1, x2, y2])
                annotation_id += 1

# Write annotations to CSV
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image_name", "annotation_id", "x_min", "y_min", "x_max", "y_max"])
    writer.writerows(annotations)

print(f"Detection complete. CSV annotations saved to '{output_csv}'.")

# Folder where images are stored and CSV file name
images_folder = "PNGImages"
csv_file = output_csv

# Dictionary to hold annotations grouped by image name
annotations_by_image = {}

# Read CSV and group annotations
with open(csv_file, mode='r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_name = row["image_name"]
        # Parse bounding box coordinates and annotation id
        annotation_id = row["annotation_id"]
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])

        if image_name not in annotations_by_image:
            annotations_by_image[image_name] = []
        annotations_by_image[image_name].append((annotation_id, x_min, y_min, x_max, y_max))

# Iterate over each image and add annotations
for image_name, boxes in annotations_by_image.items():
    image_path = os.path.join(images_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_name}.")
        continue

    # Draw each bounding box and add annotation id text
    for (annotation_id, x_min, y_min, x_max, y_max) in boxes:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{annotation_id}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the annotated image
    cv2.imshow(f"Annotations - {image_name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()