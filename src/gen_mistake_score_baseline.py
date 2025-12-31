import pandas as pd
import numpy as np
import ast
import json
import torch
import pickle
import hashlib
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class MistakeScoreCalculator:
    def __init__(
        self,
        df,
        annotation_col='Annotations',
        image_col='LocalPath',
        model_name='facebook/dinov2-large',
        score_col='MistakeScore',
        cache_path=None,
    ):
        self.df = df
        self.annotation_col = annotation_col
        self.image_col = image_col
        self.model_name = model_name
        self.score_col = score_col
        self.cache_path = cache_path

        self._load_feature_extractor()
        self._precompute_detection_features()
        self._calculate_centroids()
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

    def _load_feature_extractor(self):
        """Load DINOv2 model from Hugging Face and preprocessing"""
        from transformers import AutoImageProcessor, AutoModel
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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

    def _embed_crop(self, image, bbox):
        w, h = image.size
        x1, y1, x2, y2 = self._xywh_norm_to_xyxy_pixels(bbox, w, h)
        
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
        
        # Use Hugging Face processor - pass as PIL Image
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
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
        key_str = f"{image_path}_{ann_id}_{self.model_name}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self):
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
        if self.cache_path is None:
            return
        try:
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
        cache = self._load_cache()
        cache_updated = False
        self.features_per_row = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting features"):
            ann_map = {}
            image_path = row.get(self.image_col)
            if not image_path:
                self.features_per_row.append(ann_map)
                continue
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception:
                self.features_per_row.append(ann_map)
                continue
            annotations = self._parse_annotations(row[self.annotation_col])
            for ann in annotations:
                ann_id = ann['Id']
                class_id = ann['ClassId']
                bbox = ann['BBox']
                cache_key = self._get_cache_key(image_path, ann_id)
                if cache_key in cache:
                    feature = cache[cache_key]
                else:
                    feature = self._embed_crop(image, bbox)
                    cache[cache_key] = feature
                    cache_updated = True
                ann_map[ann_id] = (class_id, feature)
            self.features_per_row.append(ann_map)
        if cache_updated:
            self._save_cache(cache)

    # ----------------------------
    # Centroids
    # ----------------------------
    def _calculate_centroids(self):
        class_embeddings = {}
        for ann_map in self.features_per_row:
            for _, (cls_id, feat) in ann_map.items():
                class_embeddings.setdefault(cls_id, []).append(feat)
        self.centroids = {}
        for cls_id, feats in class_embeddings.items():
            mat = np.vstack(feats)
            centroid = mat.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            self.centroids[cls_id] = centroid

    # ----------------------------
    # Mistake scores
    # ----------------------------
    def _calculate_scores(self):
        # First pass: compute raw scores and collect per-class values
        raw_per_image = []
        class_to_scores = {}
        for ann_map in self.features_per_row:
            img_scores = {}
            for ann_id, (cls_id, feat) in ann_map.items():
                own_centroid = self.centroids.get(cls_id)
                if own_centroid is None:
                    continue
                dist_own = 1.0 - float(np.dot(feat, own_centroid))
                other_classes = [c for c in self.centroids.keys() if c != cls_id]
                if other_classes:
                    dist_others = [1.0 - float(np.dot(feat, self.centroids[c])) for c in other_classes]
                    min_other = float(min(dist_others))
                    score = dist_own - min_other
                else:
                    score = dist_own
                score = float(score)
                img_scores[ann_id] = score
                class_to_scores.setdefault(cls_id, []).append(score)
            raw_per_image.append(img_scores)

        # Build per-class min/max for 0-1 scaling
        class_minmax = {}
        for cls_id, vals in class_to_scores.items():
            if not vals:
                class_minmax[cls_id] = (0.0, 0.0)
            else:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                class_minmax[cls_id] = (vmin, vmax)

        # Second pass: normalize each detection by its class distribution
        norm_per_image = []
        for i, img_scores in enumerate(raw_per_image):
            ann_map = self.features_per_row[i]
            out = {}
            for ann_id, raw in img_scores.items():
                cls_id = ann_map[ann_id][0]
                vmin, vmax = class_minmax.get(cls_id, (0.0, 0.0))
                if vmax == vmin:
                    norm = 0.0
                else:
                    norm = (raw - vmin) / (vmax - vmin)
                out[str(ann_id)] = float(norm)
            norm_per_image.append(out)

        # assign id-wise per-class-normalized scores
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
    score_col='MistakeScore_NoisyB',
    cache_path=None,
):
    df = pd.read_csv(input_csv)
    calc = MistakeScoreCalculator(
        df,
        annotation_col=annotation_col,
        image_col=image_col,
        model_name=model_name,
        score_col=score_col,
        cache_path=cache_path,
    )
    df = calc.df
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV with {score_col} column at {output_csv}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    input_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/coco_val.csv"
    output_csv = "/Users/ragaai_user/Desktop/LabelQualityTest/dataset/coco_val.csv"
    cache_file = "/Users/ragaai_user/Desktop/LabelQualityTest/cache/coco_val_features_baseline.pkl"
    calculate_mistake_scores(input_csv, output_csv, cache_path=cache_file)
