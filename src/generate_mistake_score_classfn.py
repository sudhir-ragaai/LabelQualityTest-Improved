import ast
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ClassificationMistakeScoreCalculator:
    def __init__(
        self, 
        df: pd.DataFrame, 
        label_col: str = "label_name", 
        embed_col: str = "ImageEmbedding", 
        output_col: str = "MistakeScore",
        k: int = 10,
        metric: str = 'euclidean',
        use_distance_weighting: bool = True,
        use_hybrid: bool = True,
        majority_threshold: float = 0.3,
        normalization_percentile: float = None,
        filter_outliers: bool = False,
        final_score_threshold: float = None,
    ):
        self.df = df
        self.label_col = label_col
        self.embed_col = embed_col
        self.output_col = output_col
        self.k = k
        self.metric = metric
        self.use_distance_weighting = use_distance_weighting
        self.use_hybrid = use_hybrid
        self.majority_threshold = majority_threshold
        self.normalization_percentile = normalization_percentile
        self.filter_outliers = filter_outliers
        self.final_score_threshold = final_score_threshold
        
        self._parse_embeddings()
        self._compute_scores()

    @staticmethod
    def _parse_label(label_val):
        """Return class name from dict-like {name: confidence} or raw string.
        Picks the key with highest confidence. Safely handles stringified dicts.
        """
        # If already a dict
        if isinstance(label_val, dict):
            if not label_val:
                return ""
            try:
                return max(label_val.items(), key=lambda kv: float(kv[1]))[0]
            except Exception:
                # Fallback to first key
                return next(iter(label_val.keys()), "")
        # Try to parse stringified dict
        if isinstance(label_val, str):
            s = label_val
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                try:
                    import json as _json
                    parsed = _json.loads(s)
                except Exception:
                    return s
            if isinstance(parsed, dict) and parsed:
                try:
                    return max(parsed.items(), key=lambda kv: float(kv[1]))[0]
                except Exception:
                    return next(iter(parsed.keys()), "")
            return str(parsed)
        # Other types -> string
        return str(label_val)

    def _parse_embeddings(self) -> None:
        parsed_embeddings = []
        for val in self.df[self.embed_col].tolist():
            if isinstance(val, list):
                parsed_embeddings.append(np.asarray(val, dtype=np.float32))
                continue
            if isinstance(val, str):
                try:
                    arr = ast.literal_eval(val)
                except Exception:
                    arr = []
            else:
                arr = []
            parsed_embeddings.append(np.asarray(arr, dtype=np.float32))
        self.df["__emb_vec"] = parsed_embeddings

    def calculate_knn_local_voting_score(self, embeddings, labels, k, metric='euclidean'):
        """
        Improved k-NN Local Voting Score with distance weighting and outlier filtering.
        The score is the weighted fraction of a point's k nearest neighbors that have a different label.
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Initialize NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
        nn.fit(embeddings)
        
        # Find neighbors (returns k+1 neighbors including self)
        distances, indices = nn.kneighbors(embeddings)
        
        # Filter self-reference: remove the first column (point itself)
        neighbor_indices = indices[:, 1:]  # Shape: (N_samples, k)
        neighbor_distances = distances[:, 1:]  # Shape: (N_samples, k)
        
        # Compute consensus scores
        scores = np.zeros(len(embeddings))
        
        for i in range(len(embeddings)):
            neighbor_labels = labels[neighbor_indices[i]]
            distances_i = neighbor_distances[i]
            
            # Filter outliers if enabled
            if self.filter_outliers:
                median_dist = np.median(distances_i)
                valid_mask = distances_i <= (2 * median_dist)
                
                if np.sum(valid_mask) == 0:
                    scores[i] = 0.0
                    continue
                
                neighbor_labels = neighbor_labels[valid_mask]
                distances_i = distances_i[valid_mask]
            
            # Count neighbors with different labels
            conflicts = (neighbor_labels != labels[i]).astype(float)
            
            if self.use_distance_weighting:
                # Soft distance weighting: use inverse square root for gentler weighting
                max_dist = np.max(distances_i) + 1e-6
                normalized_dists = distances_i / max_dist
                weights = 1.0 / (np.sqrt(normalized_dists) + 1e-6)
                weights /= np.sum(weights)  # Normalize weights
                scores[i] = np.sum(conflicts * weights)
            else:
                # Simple fraction (original method)
                scores[i] = np.sum(conflicts) / len(conflicts)
        
        return scores
    
    def _calculate_centroids(self, all_features, all_labels):
        """Calculate class centroids for hybrid approach."""
        class_embeddings = {}
        for idx, label in enumerate(all_labels):
            class_embeddings.setdefault(label, []).append(all_features[idx])
        
        centroids = {}
        for label, feats in class_embeddings.items():
            mat = np.vstack(feats)
            centroid = mat.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            centroids[label] = centroid
        return centroids
    
    def _calculate_centroid_distances(self, all_features, all_labels, centroids):
        """Calculate distance to own centroid vs other centroids."""
        scores = np.zeros(len(all_features))
        
        for idx, (feat, label) in enumerate(zip(all_features, all_labels)):
            own_centroid = centroids.get(label)
            if own_centroid is None:
                continue
            
            # Distance to own centroid
            dist_own = 1.0 - float(np.dot(feat, own_centroid))
            
            # Distance to nearest other centroid
            other_classes = [c for c in centroids.keys() if c != label]
            if other_classes:
                dist_others = [1.0 - float(np.dot(feat, centroids[c])) for c in other_classes]
                min_other = float(min(dist_others))
                # Higher score = more likely mistake (closer to other class than own)
                scores[idx] = dist_own - min_other
            else:
                scores[idx] = dist_own
        
        return scores

    def _compute_scores(self) -> None:
        """Calculate mistake scores using improved k-NN Local Voting Score."""
        # Collect all features and labels
        all_features = []
        all_labels = []
        
        for _, row in self.df.iterrows():
            vec = row["__emb_vec"]
            label = self._parse_label(row[self.label_col])
            
            if vec.size == 0:
                continue
            
            # Normalize embedding
            v = vec.astype(np.float32)
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            
            all_features.append(v_norm)
            all_labels.append(label)
        
        if not all_features:
            # No features found, assign empty scores
            self.df[self.output_col] = [{self._parse_label(row[self.label_col]): 0.0} for _, row in self.df.iterrows()]
            self.df.drop(columns=["__emb_vec"], inplace=True)
            return
        
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        # Calculate improved k-NN Local Voting Scores
        print(f"Computing improved k-NN Local Voting Scores (k={self.k}, metric={self.metric})...")
        print(f"  - Distance weighting: {self.use_distance_weighting}")
        print(f"  - Hybrid approach: {self.use_hybrid}")
        print(f"  - Majority threshold: {self.majority_threshold}")
        
        knn_scores = self.calculate_knn_local_voting_score(
            all_features, 
            all_labels, 
            k=self.k, 
            metric=self.metric
        )
        
        # Apply majority voting: only flag if >majority_threshold neighbors disagree
        if self.majority_threshold > 0:
            knn_scores = np.where(knn_scores > self.majority_threshold, knn_scores, 0.0)
        
        # Hybrid: combine with centroid distance
        if self.use_hybrid:
            print("  - Computing centroid distances...")
            centroids = self._calculate_centroids(all_features, all_labels)
            centroid_scores = self._calculate_centroid_distances(all_features, all_labels, centroids)
            
            # Normalize both scores to [0, 1] for combination
            knn_min, knn_max = knn_scores.min(), knn_scores.max()
            if knn_max > knn_min:
                knn_norm = (knn_scores - knn_min) / (knn_max - knn_min)
            else:
                knn_norm = knn_scores
            
            # Normalize centroid scores (they can be negative)
            cent_min, cent_max = centroid_scores.min(), centroid_scores.max()
            if cent_max > cent_min:
                cent_norm = (centroid_scores - cent_min) / (cent_max - cent_min)
            else:
                cent_norm = np.zeros_like(centroid_scores)
            
            # Weighted combination: 70% k-NN, 30% centroid
            raw_scores = 0.7 * knn_norm + 0.3 * cent_norm
        else:
            raw_scores = knn_scores
        
        # Group scores by class for per-class normalization
        class_to_scores = {}
        feature_idx = 0
        
        for _, row in self.df.iterrows():
            vec = row["__emb_vec"]
            label = self._parse_label(row[self.label_col])
            
            if vec.size == 0:
                continue
            
            score = float(raw_scores[feature_idx])
            class_to_scores.setdefault(label, []).append(score)
            feature_idx += 1
        
        # Build per-class normalization (percentile-based or min-max)
        class_minmax = {}
        for label, vals in class_to_scores.items():
            if not vals:
                class_minmax[label] = (0.0, 0.0)
            else:
                if self.normalization_percentile is not None:
                    threshold = np.percentile(vals, self.normalization_percentile)
                    vmin = float(np.min(vals))
                    vmax = float(threshold)
                else:
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                class_minmax[label] = (vmin, vmax)
        
        # Map scores back to per-row format
        norm_scores_dicts = []
        feature_idx = 0
        
        for _, row in self.df.iterrows():
            vec = row["__emb_vec"]
            label = self._parse_label(row[self.label_col])
            
            if vec.size == 0:
                norm_scores_dicts.append({label: 0.0})
                continue
            
            raw_score = raw_scores[feature_idx]
            vmin, vmax = class_minmax.get(label, (0.0, 0.0))
            
            if vmax == vmin:
                norm = 0.0
            else:
                norm = (raw_score - vmin) / (vmax - vmin)
                norm = max(0.0, min(1.0, norm))
            
            # Apply final score threshold if specified (post-processing filter)
            if self.final_score_threshold is not None:
                if norm < self.final_score_threshold:
                    norm = 0.0
            
            norm_scores_dicts.append({label: float(norm)})
            feature_idx += 1
        
        # Write as dict {label_name: score}
        self.df[self.output_col] = norm_scores_dicts
        self.df.drop(columns=["__emb_vec"], inplace=True)


def calculate_mistake_scores_classification(
    input_csv: str, 
    output_csv: str, 
    label_col: str = "label_name", 
    embed_col: str = "ImageEmbedding", 
    output_col: str = "MistakeScore",
    k: int = 10,
    metric: str = 'euclidean',
    use_distance_weighting: bool = True,
    use_hybrid: bool = True,
    majority_threshold: float = 0.3,
    normalization_percentile: float = None,
    filter_outliers: bool = False,
    final_score_threshold: float = None,
) -> None:
    """
    Calculate mistake scores using improved k-NN Local Voting Score algorithm for classification.
    
    Args:
        k: Number of nearest neighbors to check (default: 10).
        metric: Distance metric to use ('euclidean' or 'cosine', default: 'euclidean').
        use_distance_weighting: Weight neighbor votes by inverse distance (default: True).
        use_hybrid: Combine k-NN (70%) + centroid distance (30%) (default: True).
        majority_threshold: Require >threshold neighbors to disagree (default: 0.3).
        normalization_percentile: Use percentile for normalization (default: None = min-max).
        filter_outliers: Filter neighbors beyond 2x median distance (default: False).
        final_score_threshold: Post-processing threshold on normalized scores (default: None).
                              Only flag items with score >= threshold. Higher = more conservative.
    """
    df = pd.read_csv(input_csv)
    calc = ClassificationMistakeScoreCalculator(
        df, 
        label_col=label_col, 
        embed_col=embed_col, 
        output_col=output_col,
        k=k,
        metric=metric,
        use_distance_weighting=use_distance_weighting,
        use_hybrid=use_hybrid,
        majority_threshold=majority_threshold,
        normalization_percentile=normalization_percentile,
        filter_outliers=filter_outliers,
        final_score_threshold=final_score_threshold,
    )
    calc.df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV with {output_col} at {output_csv}")


if __name__ == "__main__":
    # input_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/new_chexpert_multiclass_4000.csv"
    # input_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/oasis_4000.csv"
    # input_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/nih_multiclass_4000.csv"
    input_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/rsna_4000.csv"
    output_csv = input_csv
    
    # Run all combinations
    configs = [
        # 1) DINOv2
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingDinov2", "output_col": "MistakeScoreDinov2_CleanGT"},
        {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingDinov2", "output_col": "MistakeScoreDinov2_KNN"},
        # 2) DINOv3
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingDinov3", "output_col": "MistakeScoreDinov3_CleanGT"},
        # {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingDinov3", "output_col": "MistakeScoreDinov3_NoisyGT"},
        # 3) CLIP
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingCLIP", "output_col": "MistakeScoreCLIP_CleanGT"},
        # {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingCLIP", "output_col": "MistakeScoreCLIP_NoisyGT"},
        # 4) EVA02
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingEva02", "output_col": "MistakeScoreEva02_CleanGT"},
        # {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingEva02", "output_col": "MistakeScoreEva02_NoisyGT"},
        # 5) BiomedCLIP
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingBiomedCLIP", "output_col": "MistakeScoreBiomedCLIP_CleanGT"},
        {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingBiomedCLIP", "output_col": "MistakeScoreBiomedCLIP_KNN"},
        # # 1) DINOv2 Hub
        # {"label_col": "label_name", "embed_col": "ImageEmbeddingDinov2Hub", "output_col": "MistakeScoreDinov2Hub_CleanGT"},
        # {"label_col": "NOISY_GT", "embed_col": "ImageEmbeddingDinov2Hub", "output_col": "MistakeScoreDinov2Hub_NoisyGT"},
    ]
    
    # Load the original CSV
    import pandas as pd
    from collections import Counter
    df = pd.read_csv(input_csv)
    

    for config in configs:
        print(f"Running: {config['output_col']}")
        calc = ClassificationMistakeScoreCalculator(
            df.copy(), 
            label_col=config["label_col"],
            embed_col=config["embed_col"],
            output_col=config["output_col"],
            # Medical dataset optimizations - aggressive for precision:
            k=10,  # Moderate k
            metric='euclidean',  # Cosine works better for normalized embeddings
            use_distance_weighting=True,
            use_hybrid=True,
            majority_threshold=0.3,  # Require >50% to disagree
            normalization_percentile=None,  # Flag top 10% per class
            filter_outliers=False,
            final_score_threshold=0.8,  # Very strict: only flag if score >= 0.8
        )
        # Add the new column to the main dataframe
        df[config["output_col"]] = calc.df[config["output_col"]]
    
    # Save the final CSV with all mistake scores    
    df.to_csv(output_csv, index=False)
    print(f"All mistake score calculations completed!")
    print(f"Final CSV saved to: {output_csv}")