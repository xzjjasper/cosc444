import csv
from collections import defaultdict

def parse_csv_annotations(csv_path, score_threshold=0.0):
    """
    Parses CSV with headers:
    image_path, x1, y1, x2, y2, score

    Returns:
        dict: image_name -> list of (x1, y1, x2, y2) tuples
    """
    annotations = defaultdict(list)
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            score = float(row.get("score", 1.0))
            if score < score_threshold:
                continue

            # Just filename (strip directory if present)
            image_name = row["image_path"].split("/")[-1].split("\\")[-1]

            box = tuple(map(lambda x: int(round(float(x))), (row["x1"], row["y1"], row["x2"], row["y2"])))
            annotations[image_name].append(box)
    return annotations
