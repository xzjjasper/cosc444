import os
import subprocess

def run_step(description, command):
    print(f"\n=== {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed.\n")
        exit(1)
    else:
        print(f"[DONE] {description}\n")

if __name__ == "__main__":
    print("Running Full Instance Segmentation Pipeline")

    # 1. Segment people using GrabCut and save masks + crops
    run_step("1. Segmenting with GrabCut", "python segmentation/Segment_and_Crop.py")

    # 2. Extract body size descriptors from each cropped person
    run_step("2. Extracting Descriptors", "python segmentation/extractDescriptor.py")

    # 3. Cluster instances based on visual descriptors
    run_step("3. Clustering Instances", "python segmentation/clusterInstances.py")

    # 4. Visualize the segmented and clustered results
    run_step("4. Visualizing Clusters", "python clusterVisual.py")

    # 5. Evaluate segmentation using IoU against ground truth
    run_step("5. Evaluating Segmentation (IoU)", "python IOUEvaluation.py")

    print("Pipeline complete!")
