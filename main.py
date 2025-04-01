#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from pathlib import Path

def display_detection_results(image_path, boxes, scores):
    """
    Display detection results
    
    Args:
    image_path: image path
    boxes: detection boxes [N, 4] format [x1, y1, x2, y2]
    scores: detection scores [N]
    """
    # read image
    img = plt.imread(image_path)
    
    # create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # display original image + detection boxes
    ax1.imshow(img)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='red', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'{score:.2f}', 
                color='red', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    ax1.set_title('Detection Results')
    ax1.axis('off')
    
    ax2.imshow(np.zeros_like(img))  # temporary display black image
    ax2.set_title('Segmentation Results (TODO)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # start MATLAB engine
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    
    # set MATLAB working directory to dpm_jasper/dpm
    dpm_path = os.path.join('detection', 'dpm_jasper', 'dpm')
    eng.cd(dpm_path)
    print(f"Set MATLAB working directory to: {dpm_path}")
    
    # Get current MATLAB directory for debugging
    current_dir = eng.pwd(nargout=1)
    print(f"MATLAB current directory: {current_dir}")
    
    # Run startup.m to ensure all paths are correctly set
    try:
        print("Running startup.m...")
        eng.eval("startup", nargout=0)
        print("MATLAB paths initialized successfully")
    except Exception as e:
        print(f"Warning: Could not run startup.m: {str(e)}")
        print("Continuing anyway, but detection might fail if paths are not set correctly")
    
    # set model path
    model_path = os.path.join('INRIA/inriaperson_final.mat')
    
    print("DPM detection program started")
    print("Please input image path (input 'q' to exit):")
    
    while True:
        # get user input
        image_path = input().strip()
        
        # check if exit
        if image_path.lower() == 'q':
            break
            
        # check if file exists
        if not os.path.exists(image_path):
            print(f"Error: file {image_path} does not exist")
            continue
            
        try:
            # call MATLAB function for detection
            print(f"Processing image: {image_path}")
            boxes, scores = eng.detect_single_image(image_path, model_path, nargout=2)
            
            # convert to numpy arrays and handle format
            boxes = np.array(boxes)
            scores = np.array(scores).flatten()  # flatten to 1D array
            
            # boxes is Nx4 matrix of bounding boxes [x1, y1, x2, y2]
            # scores is N vector of detection scores

            # display results
            display_detection_results(image_path, boxes, scores)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue
        
        print("\nPlease input the next image path (input 'q' to exit):")
    
    # close MATLAB engine
    print("Closing MATLAB engine...")
    eng.quit()
    print("Program exited")

if __name__ == "__main__":
    main()
