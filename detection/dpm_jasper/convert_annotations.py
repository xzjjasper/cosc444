#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
import pandas as pd


def parse_annotation_file(file_path):
    """
    Parse a single Penn-Fudan-Ped annotation file
    
    Args:
    file_path: Path to the annotation file
    
    Returns:
    List of dictionaries containing parsed annotations
    """
    annotations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Extract image filename
        image_filename = None
        for line in lines:
            if line.startswith('Image filename'):
                image_filename = line.split(':')[1].strip().strip('"')
                break
        
        if not image_filename:
            print(f"Warning: Could not find image filename in {file_path}")
            return annotations
        
        # Extract image size
        image_size = None
        for line in lines:
            if line.startswith('Image size'):
                size_match = re.search(r'(\d+)\s+x\s+(\d+)', line)
                if size_match:
                    image_size = (int(size_match.group(1)), int(size_match.group(2)))
                break
        
        # Find all object annotations
        current_object = {}
        for line in lines:
            line = line.strip()
            
            # Start of new object
            if line.startswith('# Details for pedestrian'):
                if current_object:
                    annotations.append(current_object)
                current_object = {'image_path': image_filename}
            
            # Parse bounding box
            if line.startswith('Bounding box'):
                bbox_match = re.search(r'\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)', line)
                if bbox_match:
                    current_object.update({
                        'x1': int(bbox_match.group(1)),
                        'y1': int(bbox_match.group(2)),
                        'x2': int(bbox_match.group(3)),
                        'y2': int(bbox_match.group(4)),
                        'class': 'person'  # Penn-Fudan-Ped only contains person annotations
                    })
        
        # Add the last object if exists
        if current_object:
            annotations.append(current_object)
    
    return annotations


def convert_annotations_folder(input_dir, output_file):
    """
    Convert all annotation files in a folder to a single CSV file
    
    Args:
    input_dir: Directory containing annotation files
    output_file: Path to save the output CSV file
    """
    # Get all annotation files
    annotation_files = list(Path(input_dir).glob('*.txt'))
    print(f"Found {len(annotation_files)} annotation files")
    
    # Process all files
    all_annotations = []
    for file_path in annotation_files:
        try:
            annotations = parse_annotation_file(file_path)
            all_annotations.extend(annotations)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully converted {len(all_annotations)} annotations to {output_file}")
        print(f"CSV columns: {', '.join(df.columns)}")
    else:
        print("No annotations were found")


def main():
    parser = argparse.ArgumentParser(description='Convert Penn-Fudan-Ped annotation files to CSV format')
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing annotation files')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert annotations
    convert_annotations_folder(args.input_dir, args.output)


if __name__ == "__main__":
    main() 