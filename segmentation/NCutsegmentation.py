import os
import cv2
import numpy as np
import csv
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from skimage import data, segmentation, color
from skimage import graph
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



# First lets load images, and their annotated labels


current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
dataset_path = os.path.join(project_root, 'data', 'PennFudanPed', 'PNGImages')

dataset_path_annotated = os.path.join(project_root, 'detection', 'Annotations (HOG+RF).csv')


#dataset_path_annotated = '/Users/yerdanamaulenbay/Documents/COSC 444/Project/InstanceSegmentation_COSC444/detection/Annotations (HOG+RF).csv'
# 'InstanceSegmentation_COSC444/data/PennFudanPed/PNGImages/FudanPed00001.png'  # debug: sample relative path

# here I am reading each image, and saving it in a dictionary
# keys are image names, values are the images shown as matrices
images = {}

for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        image_path = os.path.join(dataset_path, filename)
        images[filename] = cv2.imread(image_path, cv2.IMREAD_COLOR)

#Now lets read the annotations:

annotations = {}
with open(dataset_path_annotated, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['image_name']
        annotation_id = int(row["annotation_id"])
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])

        if image_name not in annotations:
            annotations[image_name] = []
        # append the annotation details as a tuple
        annotations[image_name].append((annotation_id, x_min, y_min, x_max, y_max))


# Now I need to apply the annotations to my images, so it produces a cropped image

cropped_images = {}
# structure: annotated_name: [id, x_min, y_min, x_max, y_max]
# structure: image_name: [image]

for image_name, img in images.items():
    if image_name in annotations:
        crops =[]
        for annotation in annotations[image_name]:
            annotation_id, x_min, y_min, x_max, y_max = annotation
            crop = img[y_min:y_max, x_min:x_max]
            crops.append(crop)
        cropped_images[image_name] = crops
    else:
        print(f"No annotations found for {image_name}")

# Now I will apply Super Iterative Linear Clustering -- it is basically a method that uses knn from sklearn (you can think of it as debuffed NCut)

first_image = cropped_images['FudanPed00001.png'][0]
first_not_cropped_image = images['FudanPed00001.png']

labelsone = segmentation.slic(first_not_cropped_image, slic_zero=True)
superpixels = color.label2rgb(labelsone, first_not_cropped_image, kind='avg')



# Now we are bulding a graph, where nodes represent pixels, and edges show the affinity between them
Ngraph = graph.rag_mean_color(first_not_cropped_image, labelsone, mode='similarity')
ncut_labels = graph.cut_normalized(labelsone, Ngraph)
# here I am essentially assigning each pixel a color of its centroid (average color basically)
ncut_result = color.label2rgb(ncut_labels, first_not_cropped_image, kind='avg')

#display results
# figure(num=None, figsize=(18, 14), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(1,2,1)
# plt.imshow(cv2.cvtColor(ncut_result, cv2.COLOR_BGR2RGB))
# plt.title("Ncuts")


# plt.subplot(1,2,2)
# plt.imshow(cv2.cvtColor(superpixels, cv2.COLOR_BGR2RGB))
# plt.title("superpixels")
# plt.xticks([]), plt.yticks([])
# plt.show()

ncut_images = []

for image_name, img in cropped_images.items():
    labels = segmentation.slic(img[0], compactness=0.1)
    superpixels = color.label2rgb(labels, img[0], kind='avg')

    #Now we are building a graph based on superpixels that have their pixels colors averaged out to the centroids

    Ngraph = graph.rag_mean_color(img[0], labels, mode='similarity')
    ncut_labels = graph.cut_normalized(labels, Ngraph)
    ncuts_results = color.label2rgb(ncut_labels, img[0], kind='overlay')
    #saving the result into ncut_results array
    ncut_images.append(ncuts_results)

len(ncut_images)

import matplotlib.pyplot as plt
import cv2

# Number of images to display
num_images = 7

# Create a figure with an appropriate size (2 columns, num_images rows)
plt.figure(figsize=(14, num_images * 4))

# traversing over items so that we can print out about 12 output images which is ideal for efficiency
for i, (image_name, img_list) in enumerate(cropped_images.items()):
    if i >= num_images:
        break
    # since each list in cropped_image has only one image, we get that image with img_list[0]
    original_img = img_list[0]
    # Here I assumed that the order will be same
    ncut_img = ncut_images[i]
    
    # Left subplot: original (cropped) image
    plt.subplot(num_images, 2, 2 * i + 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{image_name} Original")
    plt.xticks([]), plt.yticks([])

    # Right subplot: Ncut segmented image
    ncut_img_8u = (ncut_img * 255).astype(np.uint8)
    plt.subplot(num_images, 2, 2 * i + 2)
    plt.imshow(cv2.cvtColor(ncut_img_8u, cv2.COLOR_BGR2RGB))
    plt.title(f"{image_name} Ncut")
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

