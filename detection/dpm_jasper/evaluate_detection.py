#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    
    Args:
    box1, box2: bounding boxes in [x1, y1, x2, y2] format
    
    Returns:
    IoU value in range [0, 1]
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is intersection
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def rename_columns(df, column_mapping):
    """
    Rename DataFrame columns based on mapping
    
    Args:
    df: DataFrame to rename columns
    column_mapping: dictionary mapping target names to source names

    Returns:
    DataFrame with renamed columns
    """
    # Create a new mapping containing only columns that exist in df
    valid_mapping = {}
    for target, source in column_mapping.items():
        if source in df.columns:
            valid_mapping[source] = target
    
    # Rename existing columns
    if valid_mapping:
        return df.rename(columns=valid_mapping)
    return df


def calculate_pascal_voc_ap(recall, precision):
    """
    Calculate AP using PASCAL VOC method (post-2010)
    
    Args:
    recall: array of recall values
    precision: array of precision values
    
    Returns:
    AP value
    """
    # Ensure arrays are not empty
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    
    # Add sentinel values to simplify calculation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Calculate precision envelope (non-decreasing)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    
    # Calculate area: width of each recall interval times highest precision in interval
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    
    return ap


def evaluate_detections_pascal_voc(gt_file, results_file, iou_threshold=0.5, score_threshold=None, 
                                  gt_column_mapping=None, results_column_mapping=None,
                                  class_column='class', ignore_difficult=False,
                                  generate_plots=False, output_dir=None):
    """
    Evaluate object detection results according to PASCAL VOC standard
    
    Args:
    gt_file: path to CSV file containing ground truth annotations
    results_file: path to CSV file containing detection results
    iou_threshold: IoU threshold above which a detection is considered correct
    score_threshold: optional confidence threshold, detections below this will be ignored
    gt_column_mapping: mapping for ground truth CSV column names
    results_column_mapping: mapping for results CSV column names
    class_column: name of the class column for per-class AP calculation
    ignore_difficult: whether to ignore samples marked as difficult
    generate_plots: whether to generate PR curve plots
    output_dir: directory to save plots
    
    Returns:
    Dictionary containing evaluation metrics
    """
    # Read CSV files
    try:
        gt_df = pd.read_csv(gt_file)
        results_df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(gt_df)} ground truth annotations and {len(results_df)} detection results")
    
    # Check and print column names
    print("Original ground truth column names:", list(gt_df.columns))
    print("Original results column names:", list(results_df.columns))
    
    # Apply column name mapping if provided
    if gt_column_mapping:
        gt_df = rename_columns(gt_df, gt_column_mapping)
        print("Ground truth columns after mapping:", list(gt_df.columns))
    
    if results_column_mapping:
        results_df = rename_columns(results_df, results_column_mapping)
        print("Results columns after mapping:", list(results_df.columns))
    
    # Check CSV file format
    required_gt_columns = ['image_path', 'x1', 'y1', 'x2', 'y2']
    required_results_columns = ['image_path', 'x1', 'y1', 'x2', 'y2', 'score']
    
    # Add default class if class column is missing
    if class_column not in gt_df.columns:
        print(f"Warning: Class column '{class_column}' not found in ground truth, adding default class 'default'")
        gt_df[class_column] = 'default'
    
    if class_column not in results_df.columns:
        print(f"Warning: Class column '{class_column}' not found in results, adding default class 'default'")
        results_df[class_column] = 'default'
    
    # Add class column to required lists
    required_gt_columns.append(class_column)
    required_results_columns.append(class_column)
    
    if not all(col in gt_df.columns for col in required_gt_columns):
        print(f"Ground truth file must contain columns: {required_gt_columns}")
        print(f"Actual columns: {list(gt_df.columns)}")
        sys.exit(1)
        
    if not all(col in results_df.columns for col in required_results_columns):
        print(f"Results file must contain columns: {required_results_columns}")
        print(f"Actual columns: {list(results_df.columns)}")
        sys.exit(1)
    
    # Extract filename from image path
    gt_df['filename'] = gt_df['image_path'].apply(lambda x: os.path.basename(x))
    results_df['filename'] = results_df['image_path'].apply(lambda x: os.path.basename(x))
    
    # Filter detections below confidence threshold if specified
    if score_threshold is not None:
        old_count = len(results_df)
        results_df = results_df[results_df['score'] >= score_threshold]
        print(f"Applied confidence threshold {score_threshold}, filtered out {old_count - len(results_df)} detections")
    
    # Get all classes
    all_classes = sorted(set(gt_df[class_column].unique()) | set(results_df[class_column].unique()))
    print(f"Evaluating classes: {all_classes}")
    
    # Initialize result storage
    class_metrics = {}
    cumulative_precision = {}
    cumulative_recall = {}
    
    # Evaluate each class separately
    for class_name in all_classes:
        print(f"\nEvaluating class: {class_name}")
        
        # Filter data for current class
        class_gt_df = gt_df[gt_df[class_column] == class_name]
        class_results_df = results_df[results_df[class_column] == class_name]
        
        # Sort detections by confidence
        class_results_df = class_results_df.sort_values(by='score', ascending=False)
        
        # Statistics
        total_gt = len(class_gt_df)
        total_detected = len(class_results_df)
        
        if total_gt == 0:
            print(f"  Warning: No samples found for class '{class_name}' in ground truth")
            class_metrics[class_name] = {
                'AP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_gt': 0,
                'total_detected': total_detected
            }
            continue
        
        if total_detected == 0:
            print(f"  Warning: No detections found for class '{class_name}' in results")
            class_metrics[class_name] = {
                'AP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': total_gt,
                'total_gt': total_gt,
                'total_detected': 0
            }
            continue
        
        print(f"  Found {total_gt} ground truth objects, detected {total_detected} results")
        
        # Initialize cumulative variables
        tp = np.zeros(total_detected)
        fp = np.zeros(total_detected)
        
        # Track matched ground truth objects, grouped by image
        gt_matched = {}
        for _, row in class_gt_df.iterrows():
            img_filename = row['filename']
            if img_filename not in gt_matched:
                gt_matched[img_filename] = []
            gt_matched[img_filename].append(False)
        
        # Process each detection
        for detection_idx, (_, detection) in enumerate(class_results_df.iterrows()):
            img_filename = detection['filename']
            
            # Detection box
            det_box = [
                detection['x1'],
                detection['y1'],
                detection['x2'],
                detection['y2']
            ]
            
            # If no ground truth for this class in image, mark as false positive
            if img_filename not in gt_matched or len(gt_matched[img_filename]) == 0:
                fp[detection_idx] = 1
                continue
            
            # Get ground truth for current image and class
            img_gt_df = class_gt_df[class_gt_df['filename'] == img_filename]
            
            # Find best matching ground truth
            max_iou = -float('inf')
            max_idx = -1
            
            for gt_idx, (_, gt_row) in enumerate(img_gt_df.iterrows()):
                # Skip already matched objects
                if gt_matched[img_filename][gt_idx]:
                    continue
                
                # Ground truth box
                gt_box = [
                    gt_row['x1'],
                    gt_row['y1'],
                    gt_row['x2'],
                    gt_row['y2']
                ]
                
                # Calculate IoU
                iou = calculate_iou(det_box, gt_box)
                
                # Update best match
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
            
            # Check if best match exceeds threshold
            if max_iou >= iou_threshold:
                # Mark as matched
                gt_matched[img_filename][max_idx] = True
                tp[detection_idx] = 1
            else:
                fp[detection_idx] = 1
        
        # Calculate cumulative TP and FP
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / total_gt
        
        # Store for plotting
        cumulative_precision[class_name] = precision
        cumulative_recall[class_name] = recall
        
        # Calculate AP using PASCAL VOC method
        ap = calculate_pascal_voc_ap(recall, precision)
        
        # Calculate final precision, recall and F1 score
        if len(precision) > 0:
            final_precision = precision[-1]
        else:
            final_precision = 0.0
            
        if len(recall) > 0:
            final_recall = recall[-1]
        else:
            final_recall = 0.0
            
        if final_precision + final_recall > 0:
            f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall)
        else:
            f1_score = 0.0
        
        # Calculate false negatives (undetected ground truth objects)
        fn = total_gt - cumulative_tp[-1] if len(cumulative_tp) > 0 else total_gt
        
        # Store class metrics
        class_metrics[class_name] = {
            'AP': ap,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': f1_score,
            'true_positives': int(cumulative_tp[-1]) if len(cumulative_tp) > 0 else 0,
            'false_positives': int(cumulative_fp[-1]) if len(cumulative_fp) > 0 else 0,
            'false_negatives': int(fn),
            'total_gt': total_gt,
            'total_detected': total_detected
        }
        
        # Print current class metrics
        print(f"  AP: {ap:.4f}")
        print(f"  Precision: {final_precision:.4f}")
        print(f"  Recall: {final_recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}")
        print(f"  True Positives: {class_metrics[class_name]['true_positives']}")
        print(f"  False Positives: {class_metrics[class_name]['false_positives']}")
        print(f"  False Negatives: {class_metrics[class_name]['false_negatives']}")
    
    # Calculate mAP (mean of AP across all classes)
    ap_values = [metrics['AP'] for metrics in class_metrics.values()]
    mAP = sum(ap_values) / len(ap_values) if ap_values else 0.0
    
    # Calculate overall statistics across all classes
    all_tp = sum(metrics['true_positives'] for metrics in class_metrics.values())
    all_fp = sum(metrics['false_positives'] for metrics in class_metrics.values())
    all_fn = sum(metrics['false_negatives'] for metrics in class_metrics.values())
    
    all_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0.0
    
    # Generate PR curve plots if requested
    if generate_plots and output_dir:
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot PR curve for each class
        for class_name in all_classes:
            if class_name in cumulative_precision and class_name in cumulative_recall:
                plt.figure(figsize=(10, 8))
                plt.plot(cumulative_recall[class_name], cumulative_precision[class_name], marker='.', label=f'{class_name} PR curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {class_name} (AP={class_metrics[class_name]["AP"]:.4f})')
                plt.grid(True)
                plt.legend()
                plt.savefig(plot_dir / f'pr_curve_{class_name}.png')
                plt.close()
        
        # Plot all classes on one graph
        plt.figure(figsize=(12, 10))
        for class_name in all_classes:
            if class_name in cumulative_precision and class_name in cumulative_recall:
                plt.plot(cumulative_recall[class_name], cumulative_precision[class_name], marker='.', label=f'{class_name} (AP={class_metrics[class_name]["AP"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - All Classes (mAP={mAP:.4f})')
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_dir / 'pr_curve_all_classes.png')
        plt.close()
    
    # Return complete evaluation results
    evaluation_results = {
        'mAP': mAP,
        'overall_precision': all_precision,
        'overall_recall': all_recall,
        'overall_f1': all_f1,
        'total_true_positives': all_tp,
        'total_false_positives': all_fp,
        'total_false_negatives': all_fn,
        'per_class': class_metrics
    }
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection results according to PASCAL VOC standard')
    
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth CSV file')
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to detection results CSV file')
    
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold, default 0.5')
    
    parser.add_argument('--score', type=float, default=None,
                        help='Confidence threshold, detections below this will be ignored')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results, default print to console')
    
    parser.add_argument('--auto-map', action='store_true',
                        help='Automatically try to map common column name differences')
    
    parser.add_argument('--class-column', type=str, default='class',
                        help='Name of the class column for per-class AP calculation')
    
    parser.add_argument('--plots', action='store_true',
                        help='Generate PR curve plots')
    
    parser.add_argument('--plots-dir', type=str, default='evaluation_plots',
                        help='Directory to save PR curve plots')
    
    args = parser.parse_args()
    
    print(f"Evaluating detection results according to PASCAL VOC standard...")
    print(f"Ground truth file: {args.gt}")
    print(f"Results file: {args.results}")
    print(f"IoU threshold: {args.iou}")
    print(f"Confidence threshold: {args.score if args.score is not None else 'Not applied'}")
    print(f"Class column name: {args.class_column}")
    
    # Predefined column name mappings
    gt_column_mapping = {
        'image_path': 'file_name',
        'x1': 'x_min',
        'y1': 'y_min',
        'x2': 'x_max',
        'y2': 'y_max'
    }
    
    results_column_mapping = {}
    
    # Check if auto-mapping is needed
    if not args.auto_map:
        gt_column_mapping = {}
        results_column_mapping = {}
    
    # Read one line to auto-detect possible column name mappings
    try:
        gt_df_peek = pd.read_csv(args.gt, nrows=1)
        
        # Check ground truth file column names
        if 'file_name' in gt_df_peek.columns and 'image_path' not in gt_df_peek.columns:
            gt_column_mapping['image_path'] = 'file_name'
        
        if all(col in gt_df_peek.columns for col in ['x_min', 'y_min', 'x_max', 'y_max']):
            gt_column_mapping.update({
                'x1': 'x_min',
                'y1': 'y_min',
                'x2': 'x_max',
                'y2': 'y_max'
            })
        
        print("Auto-detected ground truth column name mappings:", gt_column_mapping)
    except Exception as e:
        print(f"Could not auto-detect ground truth column names: {e}")
    
    # Run evaluation
    metrics = evaluate_detections_pascal_voc(
        args.gt, 
        args.results, 
        args.iou, 
        args.score,
        gt_column_mapping,
        results_column_mapping,
        class_column=args.class_column,
        generate_plots=args.plots,
        output_dir=args.plots_dir if args.plots else None
    )
    
    # Prepare output results
    output_str = "\n===== PASCAL VOC Detection Evaluation Results =====\n"
    output_str += f"Mean Average Precision (mAP): {metrics['mAP']:.4f}\n"
    output_str += f"Overall Precision: {metrics['overall_precision']:.4f}\n"
    output_str += f"Overall Recall: {metrics['overall_recall']:.4f}\n"
    output_str += f"Overall F1 Score: {metrics['overall_f1']:.4f}\n"
    output_str += f"Total True Positives (TP): {metrics['total_true_positives']}\n"
    output_str += f"Total False Positives (FP): {metrics['total_false_positives']}\n"
    output_str += f"Total False Negatives (FN): {metrics['total_false_negatives']}\n\n"
    
    # Add detailed results for each class
    output_str += "Per-class Detailed Results:\n"
    output_str += "-" * 80 + "\n"
    output_str += f"{'Class':<15} {'AP':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6}\n"
    output_str += "-" * 80 + "\n"
    
    for class_name, class_metric in metrics['per_class'].items():
        output_str += f"{class_name:<15} {class_metric['AP']:<10.4f} {class_metric['precision']:<10.4f} "
        output_str += f"{class_metric['recall']:<10.4f} {class_metric['f1_score']:<10.4f} "
        output_str += f"{class_metric['true_positives']:<6} {class_metric['false_positives']:<6} {class_metric['false_negatives']:<6}\n"
    
    # Output or save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"Results saved to: {args.output}")
    else:
        print(output_str)
    
    # Generate PR curve plots
    if args.plots:
        print(f"PR curve plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main() 