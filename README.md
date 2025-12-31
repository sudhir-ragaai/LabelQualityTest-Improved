# LabelQualityTest-Improved

This repository contains an improved implementation of a Label Quality Test (LQT) tool designed to detect mislabeled bounding boxes in object detection datasets. It leverages **DINOv2** embeddings and a **Hybrid k-NN + Centroid** scoring algorithm to identify potential label errors with high precision.

## Key Features

-   **State-of-the-Art Embeddings**: Uses `facebook/dinov2-large` (or base) for robust visual feature extraction.
-   **Hybrid Scoring Algorithm**: Combines local consistency (k-NN) with global class coherence (Centroid Distance) to detect outliers.
-   **Improved k-NN Voting**:
    -   **Distance Weighting**: Closer neighbors have more influence.
    -   **Outlier Filtering**: Ignores neighbors that are statistically too far away.
    -   **Majority Thresholding**: Only flags items where a significant portion of neighbors disagree.
-   **Per-Class Normalization**: Normalizes scores within each class to account for varying class distributions (using Min-Max or Percentile-based normalization).
-   **Caching**: Caches extracted features to speed up subsequent runs.

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy torch transformers scikit-learn tqdm Pillow
```

## Usage

The main script is `src/gen_mistake_score_detection.py`. You can run it directly or import the `calculate_mistake_scores` function.

### Running via Command Line

You can modify the `if __name__ == "__main__":` block in `src/gen_mistake_score_detection.py` or create a wrapper script.

```python
from src.gen_mistake_score_detection import calculate_mistake_scores

calculate_mistake_scores(
    input_csv="path/to/dataset.csv",
    output_csv="path/to/output_with_scores.csv",
    cache_path="cache/features.pkl",
    score_col='MistakeScore_KNN',
    k=15,                       # Number of neighbors
    metric='euclidean',         # Distance metric
    use_distance_weighting=True,
    use_hybrid=True             # Enable hybrid k-NN + Centroid score
)
```

### Input CSV Format

The input CSV must contain:
-   `LocalPath`: Absolute path to the image file.
-   `Annotations` (or similar column): JSON string or list of dictionaries containing:
    -   `Id`: Unique detection ID.
    -   `ClassId`: Integer class ID.
    -   `ClassName`: String class name.
    -   `BBox`: `[x, y, w, h]` (normalized 0-1, COCO format).

### Output

The script generates a new CSV with an added column (default `MistakeScore`) containing a dictionary of `{ann_id: score}` for each image.
-   **Score Range**: 0.0 (likely correct) to 1.0 (likely mistake).
