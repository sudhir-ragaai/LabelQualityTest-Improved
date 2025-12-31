import argparse
import ast
import hashlib
import io
import json
import os
import pickle
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from google.cloud import storage
from requests.adapters import HTTPAdapter, Retry
from sklearn.neighbors import NearestNeighbors
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
from dotenv import load_dotenv

# Suppress harmless urllib3 connection pool warnings from Google auth library
warnings.filterwarnings("ignore", message=".*Connection pool is full.*", category=UserWarning)

# Load environment variables from .env file
# Try multiple locations: current working directory (most common), script dir, and parent dirs
env_loaded = False
env_paths = [
    os.path.join(os.getcwd(), ".env"),  # Current working directory (most common)
    os.path.expanduser("~/.env"),  # Home directory
]
# Also try walking up from current directory
current = os.getcwd()
for _ in range(5):  # Check up to 5 levels up
    env_paths.append(os.path.join(current, ".env"))
    parent = os.path.dirname(current)
    if parent == current:  # Reached root
        break
    current = parent

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        env_loaded = True
        break

# Fallback: default load_dotenv() behavior
if not env_loaded:
    load_dotenv()

class MistakeScoreCalculator:
    def __init__(
        self,
        df,
        annotation_col='Annotations',
        image_col='LocalPath',
        model_name='facebook/dinov2-large',
        score_col='MistakeScore',
        cache_path=None,
        k=5,
        metric='euclidean',
        debug_save_dir=None,
        debug_max_samples=3,
        crop_margin=0.1,
        use_distance_weighting=True,
        confidence_threshold=None,
        normalization_percentile=None,
        majority_threshold=0.3,
        filter_outliers=True,
    ):
        self.df = df
        self.annotation_col = annotation_col
        self.image_col = image_col
        self.model_name = model_name
        # Normalize score column name (users sometimes pass tuples or lists)
        if isinstance(score_col, (list, tuple)):
            if len(score_col) == 1:
                score_col = score_col[0]
            else:
                score_col = '_'.join(str(s) for s in score_col)
        self.score_col = str(score_col)
        self.cache_path = cache_path
        self.k = k  # Number of nearest neighbors
        self.metric = metric  # Distance metric: 'euclidean' or 'cosine'
        self.debug_save_dir = debug_save_dir
        self.debug_max_samples = debug_max_samples
        self._debug_saved = 0
        self.crop_margin = crop_margin  # Margin around bbox (0.1 = 10% padding)
        
        # New improvement parameters
        self.use_distance_weighting = use_distance_weighting  # Weight votes by inverse distance
        self.confidence_threshold = confidence_threshold  # Minimum conflict score to flag (None = no threshold)
        self.normalization_percentile = normalization_percentile  # Use percentile instead of min-max (80 = top 20%)
        self.filter_outliers = filter_outliers  # Filter neighbors beyond reasonable distance
        self.use_hybrid = True  # Combine k-NN with centroid distance
        self.majority_threshold = majority_threshold  # Require >30% neighbors to disagree (stricter voting, but not too aggressive)

        # I/O helpers
        self.gcs_client = self._init_gcs_client()
        self.http = self._init_http_session()

        # Device / precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = self.device == 'cuda'

        self._load_feature_extractor()
        self._precompute_detection_features()
        self._calculate_scores()

    # ----------------------------
    # Helper functions
    # ----------------------------
    @staticmethod
    def _parse_annotations(val):
        """Parse annotations into list of dicts with Id, ClassId, ClassName, BBox"""
        detections = []
        parsed = val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except:
                try:
                    parsed = ast.literal_eval(val)
                except:
                    parsed = []
        if isinstance(parsed, dict) and 'annotations' in parsed:
            detections = parsed['annotations'] or []
        elif isinstance(parsed, list):
            detections = parsed
        norm = []
        for idx, det in enumerate(detections):
            if not isinstance(det, dict):
                continue
            class_name = det.get('ClassName') or det.get('class') or det.get('label')
            class_id = det.get('ClassId')
            bbox = det.get('BBox')
            if class_name is None or class_id is None or bbox is None:
                continue
            norm.append({
                'Id': det.get('Id', idx),
                'ClassId': class_id,
                'ClassName': class_name,
                'BBox': bbox
            })
        return norm

    @staticmethod
    def _xywh_norm_to_xyxy_pixels(bbox, width, height):
        """Convert COCO normalized format (xywh_normalized) to pixel coordinates.
        xywh_normalized: x, y are top-left corner (normalized 0-1), w, h are width/height (normalized 0-1)
        """
        x, y, w, h = bbox
        x1 = max(0, int(x * width))
        y1 = max(0, int(y * height))
        x2 = min(width, int((x + w) * width))
        y2 = min(height, int((y + h) * height))
        return x1, y1, x2, y2

    def _init_http_session(self):
        sess = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retries)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        return sess

    def _init_gcs_client(self):
        try:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                return storage.Client.from_service_account_json(creds_path)
            return storage.Client()
        except Exception:
            return None

    @staticmethod
    def _parse_gcs_https(parsed):
        netloc_parts = parsed.netloc.split(".")
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if parsed.netloc == "storage.googleapis.com" and len(path_parts) == 2:
            return path_parts[0], path_parts[1]
        if len(netloc_parts) >= 4 and netloc_parts[-3:] == ["storage", "googleapis", "com"]:
            bucket = ".".join(netloc_parts[:-3])
            key = parsed.path.lstrip("/")
            return bucket, key
        return None, None

    def _open_image(self, path):
        path_str = str(path).strip()
        parsed = urlparse(path_str)

        if parsed.scheme == "gs":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return self._read_gcs(bucket, key)

        if parsed.scheme in ("http", "https"):
            is_gcs_https = (
                parsed.netloc.endswith("storage.googleapis.com")
                or ".storage.googleapis.com" in parsed.netloc
            )
            if is_gcs_https:
                bucket, key = self._parse_gcs_https(parsed)
                if bucket and key:
                    try:
                        return self._read_gcs(bucket, key)
                    except Exception:
                        pass
            resp = self.http.get(path_str, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))

        return Image.open(path_str)

    def _read_gcs(self, bucket: str, key: str) -> Image.Image:
        if self.gcs_client is None:
            url = f"https://storage.googleapis.com/{bucket}/{key}"
            resp = self.http.get(url, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        blob = self.gcs_client.bucket(bucket).blob(key)
        data = blob.download_as_bytes()
        return Image.open(io.BytesIO(data))

    def _load_feature_extractor(self):
        """Load DINOv2 model from Hugging Face and preprocessing"""
        from transformers import AutoImageProcessor, AutoModel

        # Use model_name directly as Hugging Face model ID, or default to DINOv2
        if '/' in self.model_name:
            # Already a full Hugging Face model ID
            hf_model_id = self.model_name
        else:
            # Default to DINOv2 large
            hf_model_id = 'facebook/dinov2-large'
        
        self.processor = AutoImageProcessor.from_pretrained(hf_model_id)
        self.model = AutoModel.from_pretrained(hf_model_id)
        self.model.eval().to(self.device)
        # Optional compile for speed on CUDA
        if self.use_amp and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile for faster inference")
            except Exception as exc:
                print(f"Warning: torch.compile failed: {exc}")

    def _embed_crop(self, image, bbox):
        w, h = image.size
        x1, y1, x2, y2 = self._xywh_norm_to_xyxy_pixels(bbox, w, h)
        
        # Add margin around bbox to preserve context and object quality
        if x2 > x1 and y2 > y1 and self.crop_margin > 0:
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            margin_x = int(bbox_w * self.crop_margin)
            margin_y = int(bbox_h * self.crop_margin)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
        
        # Ensure valid crop coordinates
        if x2 <= x1 or y2 <= y1:
            crop = image
        else:
            crop = image.crop((x1, y1, x2, y2))
        
        # Ensure crop is RGB and has valid dimensions
        crop_w, crop_h = crop.size
        
        # If crop is too small (less than 4x4), use full image to avoid processing issues
        if crop_w < 4 or crop_h < 4:
            crop = image
            crop_w, crop_h = crop.size
        
        # Ensure crop is RGB (convert if needed)
        if crop.mode != 'RGB':
            crop = crop.convert('RGB')
        
        # Ensure crop is a PIL Image (not numpy array)
        if not isinstance(crop, Image.Image):
            crop = Image.fromarray(crop).convert('RGB')
        
        # Double-check it's RGB after all operations
        if crop.mode != 'RGB':
            crop = crop.convert('RGB')
        
        # Resize while preserving aspect ratio - only if crop is too large
        # Let the processor handle final resizing for better quality
        # Only resize manually if crop is significantly larger than model input
        max_model_size = 518  # DINOv2 can handle up to 518px
        if crop_w > 0 and crop_h > 0:
            max_dim = max(crop_w, crop_h)
            # Only resize if crop is larger than model can handle efficiently
            # This preserves quality for smaller crops and avoids unnecessary upsampling
            if max_dim > max_model_size:
                aspect_ratio = crop_w / crop_h
                if aspect_ratio > 1:
                    # Wider than tall - resize width to max_model_size
                    new_w = max_model_size
                    new_h = int(max_model_size / aspect_ratio)
                else:
                    # Taller than wide or square - resize height to max_model_size
                    new_h = max_model_size
                    new_w = int(max_model_size * aspect_ratio)
                # Use LANCZOS for high-quality downsampling
                crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if (
            self.debug_save_dir
            and self._debug_saved < self.debug_max_samples
        ):
            try:
                os.makedirs(self.debug_save_dir, exist_ok=True)
                save_path = os.path.join(self.debug_save_dir, f"resized_{self._debug_saved}.jpg")
                crop.save(save_path)
                print(f"Saved resized bbox example {self._debug_saved + 1}/{self.debug_max_samples} to {save_path}")
                self._debug_saved += 1
            except Exception as exc:
                print(f"Warning: could not save resized example: {exc}")

        # Use Hugging Face processor - pass as PIL Image
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        else:
            autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)

        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**inputs)
            # Get embeddings - try pooler_output first, then last_hidden_state mean
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                # Use mean of last hidden state (CLS token or average pooling)
                feat = outputs.last_hidden_state.mean(dim=1)
        
        feat = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-12)
        return feat

    # ----------------------------
    # Feature caching
    # ----------------------------
    def _get_cache_key(self, image_path, ann_id):
        """Generate a unique cache key for a bbox feature using image path and annotation ID."""
        # Create a hash from image path, annotation ID, and model name
        key_str = f"{image_path}_{ann_id}_{self.model_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self):
        """Load feature cache from disk if available."""
        if self.cache_path is None or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} cached features from {self.cache_path}")
            return cache
        except Exception as e:
            print(f"Warning: Could not load cache from {self.cache_path}: {e}")
            return {}
    
    def _save_cache(self, cache):
        """Save feature cache to disk."""
        if self.cache_path is None:
            return
        try:
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Saved {len(cache)} features to cache: {self.cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache to {self.cache_path}: {e}")

    # ----------------------------
    # Precompute per-detection features
    # ----------------------------
    def _precompute_detection_features(self):
        """Extract features with caching support."""
        # Load existing cache
        cache = self._load_cache()
        cache_updated = False
        
        self.features_per_row = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
            ann_map = {}
            image_path = row.get(self.image_col)
            if not image_path:
                self.features_per_row.append(ann_map)
                continue
            
            # Parse annotations first
            annotations = self._parse_annotations(row.get(self.annotation_col, []))
            if not annotations:
                self.features_per_row.append(ann_map)
                continue
            
            # Check cache keys for all annotations first
            cache_keys = {}
            all_cached = True
            for ann in annotations:
                ann_id = ann['Id']
                cache_key = self._get_cache_key(image_path, ann_id)
                cache_keys[ann_id] = cache_key
                if cache_key not in cache:
                    all_cached = False
            
            # Only open image if at least one annotation is missing from cache
            image = None
            if not all_cached:
                try:
                    image = self._open_image(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                except:
                    # If image fetch fails, use cached features for what we have
                    image = None
            
            # Process each annotation
            for ann in annotations:
                ann_id = ann['Id']
                class_id = ann['ClassId']
                bbox = ann['BBox']
                cache_key = cache_keys[ann_id]
                
                # Check cache first
                if cache_key in cache:
                    feature = cache[cache_key]
                else:
                    # Extract feature if not in cache (requires image)
                    if image is None:
                        # Skip if image couldn't be loaded
                        continue
                    feature = self._embed_crop(image, bbox)
                    cache[cache_key] = feature
                    cache_updated = True
                
                ann_map[ann_id] = (class_id, feature)
            self.features_per_row.append(ann_map)
        
        # Save cache if updated
        if cache_updated:
            self._save_cache(cache)

    # ----------------------------
    # Improved k-NN Local Voting Score
    # ----------------------------
    def calculate_knn_local_voting_score(self, embeddings, labels, k, metric='euclidean'):
        """
        Improved k-NN Local Voting Score with distance weighting and outlier filtering.
        The score is the weighted fraction of a point's k nearest neighbors that have a different label.
        
        Args:
            embeddings: A NumPy array of feature embeddings (N_samples, D_features).
            labels: A NumPy array of class labels (N_samples,).
            k: The number of nearest neighbors to check (must be an integer >= 1).
            metric: The distance metric to use ('euclidean' or 'cosine').
        
        Returns:
            A NumPy array of scores (N_samples,) where higher scores indicate a higher probability of label error.
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Initialize NearestNeighbors
        # Use k+1 because the point itself will be the first neighbor
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
                # Only consider neighbors within 2x median distance
                median_dist = np.median(distances_i)
                valid_mask = distances_i <= (2 * median_dist)
                
                if np.sum(valid_mask) == 0:
                    # No valid neighbors, score is 0
                    scores[i] = 0.0
                    continue
                
                neighbor_labels = neighbor_labels[valid_mask]
                distances_i = distances_i[valid_mask]
            
            # Count neighbors with different labels
            conflicts = (neighbor_labels != labels[i]).astype(float)
            
            if self.use_distance_weighting:
                # Soft distance weighting: use inverse square root for gentler weighting
                # This gives closer neighbors more weight but not as extreme as pure inverse
                max_dist = np.max(distances_i) + 1e-6
                # Normalize distances to [0, 1] range
                normalized_dists = distances_i / max_dist
                # Use inverse square root for softer weighting
                weights = 1.0 / (np.sqrt(normalized_dists) + 1e-6)
                weights /= np.sum(weights)  # Normalize weights
                # Weighted conflict score
                scores[i] = np.sum(conflicts * weights)
            else:
                # Simple fraction (original method)
                scores[i] = np.sum(conflicts) / len(conflicts)
        
        return scores
    
    def _calculate_centroids(self, all_features, all_labels):
        """Calculate class centroids for hybrid approach."""
        class_embeddings = {}
        for idx, cls_id in enumerate(all_labels):
            class_embeddings.setdefault(cls_id, []).append(all_features[idx])
        
        centroids = {}
        for cls_id, feats in class_embeddings.items():
            mat = np.vstack(feats)
            centroid = mat.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            centroids[cls_id] = centroid
        return centroids
    
    def _calculate_centroid_distances(self, all_features, all_labels, centroids):
        """Calculate distance to own centroid vs other centroids."""
        scores = np.zeros(len(all_features))
        
        for idx, (feat, cls_id) in enumerate(zip(all_features, all_labels)):
            own_centroid = centroids.get(cls_id)
            if own_centroid is None:
                continue
            
            # Distance to own centroid
            dist_own = 1.0 - float(np.dot(feat, own_centroid))
            
            # Distance to nearest other centroid
            other_classes = [c for c in centroids.keys() if c != cls_id]
            if other_classes:
                dist_others = [1.0 - float(np.dot(feat, centroids[c])) for c in other_classes]
                min_other = float(min(dist_others))
                # Higher score = more likely mistake (closer to other class than own)
                scores[idx] = dist_own - min_other
            else:
                scores[idx] = dist_own
        
        return scores
    
    def _calculate_scores(self):
        """Calculate mistake scores using improved k-NN Local Voting Score."""
        # Collect all features and labels
        all_features = []
        all_labels = []
        feature_to_row_ann = []  # Map feature index to (row_idx, ann_id)
        
        for row_idx, ann_map in enumerate(self.features_per_row):
            for ann_id, (cls_id, feat) in ann_map.items():
                all_features.append(feat)
                all_labels.append(cls_id)
                feature_to_row_ann.append((row_idx, ann_id))
        
        if not all_features:
            # No features found, assign empty scores
            self.df[self.score_col] = [{} for _ in range(len(self.df))]
            return
        
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        # Adjust k based on data size: use smaller k if dataset is small
        n_samples = len(all_features)
        # k must be at least 1, and at most n_samples - 1 (since we exclude self)
        # For small datasets, use a smaller k (min 3, or 20% of data, whichever is smaller)
        if n_samples < self.k:
            adjusted_k = max(1, n_samples - 1)
        elif n_samples < 50:
            # For very small datasets, use min(3, 20% of data)
            adjusted_k = min(self.k, max(3, int(n_samples * 0.2)))
        else:
            adjusted_k = self.k
        
        # Calculate improved k-NN Local Voting Scores
        knn_scores = self.calculate_knn_local_voting_score(
            all_features, 
            all_labels, 
            k=adjusted_k, 
            metric=self.metric
        )
        
        # Apply majority voting: only flag if >majority_threshold neighbors disagree
        if self.majority_threshold > 0:
            knn_scores = np.where(knn_scores > self.majority_threshold, knn_scores, 0.0)
        
        # Hybrid: combine with centroid distance
        if self.use_hybrid:
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
        
        # Apply confidence threshold if specified
        if self.confidence_threshold is not None:
            print(f"  - Applying confidence threshold: {self.confidence_threshold}")
            raw_scores = np.where(raw_scores >= self.confidence_threshold, raw_scores, 0.0)
        
        # Group scores by class for per-class normalization
        class_to_scores = {}
        for idx, (row_idx, ann_id) in enumerate(feature_to_row_ann):
            cls_id = all_labels[idx]
            score = float(raw_scores[idx])
            class_to_scores.setdefault(cls_id, []).append(score)
        
        # Build per-class normalization (percentile-based or min-max)
        class_minmax = {}
        for cls_id, vals in class_to_scores.items():
            if not vals:
                class_minmax[cls_id] = (0.0, 0.0)
            else:
                if self.normalization_percentile is not None:
                    # Percentile-based: use top N% as threshold
                    threshold = np.percentile(vals, self.normalization_percentile)
                    vmin = float(np.min(vals))
                    vmax = float(threshold)
                else:
                    # Min-max normalization (original)
                    vmin = float(np.min(vals))
                    vmax = float(np.max(vals))
                class_minmax[cls_id] = (vmin, vmax)
        
        # Map scores back to per-image format
        norm_per_image = []
        current_idx = 0
        
        for row_idx, ann_map in enumerate(self.features_per_row):
            out = {}
            for ann_id in ann_map.keys():
                cls_id = all_labels[current_idx]
                raw_score = raw_scores[current_idx]
                vmin, vmax = class_minmax.get(cls_id, (0.0, 0.0))
                
                if vmax == vmin:
                    norm = 0.0
                else:
                    norm = (raw_score - vmin) / (vmax - vmin)
                    # Clamp to [0, 1]
                    norm = max(0.0, min(1.0, norm))
                
                out[str(ann_id)] = float(norm)
                current_idx += 1
            
            norm_per_image.append(out)
        
        # Assign id-wise per-class-normalized scores
        self.df[self.score_col] = norm_per_image

# ----------------------------
# Run full pipeline
# ----------------------------
def calculate_mistake_scores(
    input_csv,
    output_csv,
    annotation_col='AnnotationsV1_Noisy',
    image_col='LocalPath',
    model_name='facebook/dinov2-base',
    score_col='MistakeScore_NoisyA1',
    cache_path=None,
    k=5,
    metric='euclidean',
    crop_margin=0.1,
    use_distance_weighting=True,
    confidence_threshold=None,
    normalization_percentile=80,
    filter_outliers=True,
    majority_threshold=0.3,
):
    """
    Calculate mistake scores using improved k-NN Local Voting Score algorithm.
    """
    df = pd.read_csv(input_csv)
    # Resolve image column: prefer provided, else fall back to LocalPath or ImageUri
    resolved_image_col = image_col
    if resolved_image_col not in df.columns:
        if "LocalPath" in df.columns:
            resolved_image_col = "LocalPath"
            print(f"Info: image column '{image_col}' not found; using 'LocalPath'.")
        elif "ImageUri" in df.columns:
            resolved_image_col = "ImageUri"
            print(f"Info: image column '{image_col}' not found; using 'ImageUri'.")
        else:
            raise ValueError(f"Image column '{image_col}' not found and no fallback (LocalPath/ImageUri) present.")

    calc = MistakeScoreCalculator(
        df,
        annotation_col=annotation_col,
        image_col=resolved_image_col,
        model_name=model_name,
        score_col=score_col,
        cache_path=cache_path,
        k=k,
        metric=metric,
        crop_margin=crop_margin,
        use_distance_weighting=use_distance_weighting,
        confidence_threshold=confidence_threshold,
        normalization_percentile=normalization_percentile,
        filter_outliers=filter_outliers,
        majority_threshold=majority_threshold,
    )
    df = calc.df
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV with {score_col} column at {output_csv}")

def parse_args():
    parser = argparse.ArgumentParser(description="Compute mistake scores with k-NN voting and DINOv2 features.")
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-csv", help="Path to output CSV")
    parser.add_argument("--annotation-col", default="Annotations", help="Column with annotations")
    parser.add_argument("--image-col", default="ImageUri", help="Column with image paths")
    parser.add_argument("--model-name", default="facebook/dinov2-base", help="Hugging Face model id")
    parser.add_argument("--score-col", default="MistakeScore", help="Output score column name")
    parser.add_argument("--cache-path", default=None, help="Optional pickle cache for bbox features")
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean", help="Distance metric")
    parser.add_argument("--crop-margin", type=float, default=0.1, help="Padding fraction around bbox (e.g., 0.1)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache_path:
        cache_path = args.cache_path
    else:
        # Generate cache filename based on input CSV filename
        csv_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
        cache_path = os.path.join("cache", f"{csv_basename}_bbox_cache.pkl")
    calculate_mistake_scores(
        input_csv=args.input_csv,
        output_csv=args.output_csv or args.input_csv,
        annotation_col=args.annotation_col,
        image_col=args.image_col,
        model_name=args.model_name,
        score_col=args.score_col,
        cache_path=cache_path,
        k=args.k,
        metric=args.metric,
        crop_margin=args.crop_margin,
    )


if __name__ == "__main__":
    main()