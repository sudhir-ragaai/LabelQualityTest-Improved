import argparse
import io
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import pandas as pd
import torch
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from google.cloud import storage
from dotenv import load_dotenv

# Suppress urllib3 connection pool warnings (harmless, from Google auth library)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

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


class EmbeddingGenerator:
    """Generate image embeddings using Hugging Face DINOv2 with GCS/HTTP/local support."""

    def __init__(self, hf_model_id: str = "facebook/dinov2-large", use_fast: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_model_id = hf_model_id
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id, use_fast=use_fast)
        self.model = AutoModel.from_pretrained(self.hf_model_id).to(self.device).eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Initialize GCS client with credentials from .env if available
        self.gcs_client = self._init_gcs_client()
        # Shared HTTP session with connection pooling + retries
        self.http = self._init_http_session()
    
    def _init_gcs_client(self):
        """Initialize GCS client using credentials from .env file or environment."""
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Clean up path (remove quotes if present, strip whitespace)
        if creds_path:
            creds_path = creds_path.strip().strip('"').strip("'")
        
        if creds_path and os.path.exists(creds_path):
            return storage.Client.from_service_account_json(creds_path)
        else:
            try:
                return storage.Client()
            except Exception as e:
                print(f"Warning: Could not initialize GCS client: {e}")
                print("GCS access will fall back to HTTP for public buckets")
                return None

    def _init_http_session(self):
        """Create a pooled HTTP session with retries to speed up repeated fetches."""
        sess = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=retries)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        return sess

    def _preprocess_image(self, image: Image.Image):
        """Preprocess a PIL Image for DINOv2."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs

    def _get_embedding(self, image: Image.Image):
        """Get normalized embedding for a single image (PIL Image object)."""
        data = self._preprocess_image(image)
        tensors = {k: v.to(self.device) for k, v in data.items()}
        use_amp = self.device.type == "cuda"
        if use_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16)
        else:
            # torch.cpu.amp.autocast is deprecated; use torch.amp.autocast with device_type="cpu"
            autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)
        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**tensors)
            feats = outputs.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats[0].tolist()

    def generate_embeddings_for_csv(
        self,
        csv_path: str,
        output_path: str,
        image_path_col: str = "ImageUri",
        output_embed_col: str = "ImageVectorsM1",
        num_workers: int = 16,
        cache_path: str | None = None,
    ):
        """
        Generate DINOv2 embeddings with parallel fetch + streaming:
        - Fetch images with a thread pool (bounded by num_workers).
        - Embed each image as soon as it finishes downloading to keep memory low.
        - Optional on-disk cache to skip re-embedding previously processed images.
        """
        df = pd.read_csv(csv_path)[:1000]
        if image_path_col not in df.columns:
            raise ValueError(f"Column '{image_path_col}' not found in CSV")

        total = len(df)
        embeddings = [None] * total

        # Load cache if provided
        cache = {}
        cache_updated = False
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached embeddings from {cache_path}")
            except Exception as e:
                print(f"Warning: could not read cache {cache_path}: {e}")

        embed_pbar = tqdm(total=total, desc="Generating embeddings", mininterval=0.5, miniters=1)

        paths = df[image_path_col].tolist()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {
                pool.submit(self._open_image, path): (idx, path)
                for idx, path in enumerate(paths)
                if path not in cache  # skip already cached
            }
            for future in as_completed(future_to_idx):
                idx, path = future_to_idx[future]
                try:
                    img = future.result()
                except Exception as e:
                    print(f"Warning: failed to fetch {path}: {e}")
                    embeddings[idx] = []
                    embed_pbar.update(1)
                    continue
                try:
                    emb = self._get_embedding(img)
                    embeddings[idx] = emb
                    if cache_path:
                        cache[path] = emb
                        cache_updated = True
                except Exception as e:
                    print(f"Warning: failed to process {path}: {e}")
                    embeddings[idx] = []
                embed_pbar.update(1)

        # Fill embeddings from cache for skipped items
        if cache:
            for idx, path in enumerate(paths):
                if embeddings[idx] is None and path in cache:
                    embeddings[idx] = cache[path]
                    embed_pbar.update(1)

        embed_pbar.close()

        df[output_embed_col] = embeddings
        df.to_csv(output_path, index=False)

        # Save cache if updated
        if cache_path and cache_updated:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True) if os.path.dirname(cache_path) else None
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
                print(f"Saved {len(cache)} cached embeddings to {cache_path}")
            except Exception as e:
                print(f"Warning: could not save cache {cache_path}: {e}")

        successful = sum(1 for e in embeddings if e)
        failed = len(df) - successful
        print(f"Completed: {successful} successful, {failed} failed")

    def _open_image(self, path):
        """Open image from local path, gs://, or HTTP(S) (incl. GCS HTTPS)."""
        path_str = str(path).strip()
        parsed = urlparse(path_str)

        # gs://bucket/key
        if parsed.scheme == "gs":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return self._read_gcs(bucket, key)

        # http(s):// (support GCS HTTPS and generic HTTP)
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
                        pass  # fall back to plain HTTP
            resp = self.http.get(path_str, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))

        # Local file
        return Image.open(path_str)

    def _read_gcs(self, bucket: str, key: str) -> Image.Image:
        """Read a GCS object into a PIL Image."""
        if self.gcs_client is None:
            # Fall back to HTTP if GCS client not available
            url = f"https://storage.googleapis.com/{bucket}/{key}"
            resp = self.http.get(url, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        
        blob = self.gcs_client.bucket(bucket).blob(key)
        data = blob.download_as_bytes()
        return Image.open(io.BytesIO(data))

    @staticmethod
    def _parse_gcs_https(parsed):
        """
        Parse HTTPS GCS URLs like:
        - https://storage.googleapis.com/bucket/key
        - https://bucket.storage.googleapis.com/key
        """
        netloc_parts = parsed.netloc.split(".")
        path_parts = parsed.path.lstrip("/").split("/", 1)

        # storage.googleapis.com/bucket/key
        if parsed.netloc == "storage.googleapis.com" and len(path_parts) == 2:
            bucket = path_parts[0]
            key = path_parts[1]
            return bucket, key

        # bucket.storage.googleapis.com/key
        if len(netloc_parts) >= 4 and netloc_parts[-3:] == ["storage", "googleapis", "com"]:
            bucket = ".".join(netloc_parts[:-3])
            key = parsed.path.lstrip("/")
            return bucket, key

        return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DINOv2 embeddings for a CSV of image paths."
    )
    parser.add_argument("--csv-path", required=True, help="Input CSV file path.")
    parser.add_argument(
        "--output-path",
        help="Output CSV file path. Defaults to the same as --csv-path (in-place).",
    )
    parser.add_argument(
        "--image-path-col",
        default="ImageUri",
        help="Name of the CSV column containing image paths.",
    )
    parser.add_argument(
        "--output-embed-col",
        default="ImageVectorsM1",
        help="Name of the CSV column to write embeddings into.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for fetching images (default: 8). Increase for faster I/O.",
    )
    parser.add_argument(
        "--cache-path",
        help="Optional pickle cache file path to reuse embeddings across runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = args.csv_path
    output_csv = args.output_path or args.csv_path
    generator = EmbeddingGenerator()
    generator.generate_embeddings_for_csv(
        input_csv,
        output_csv,
        image_path_col=args.image_path_col,
        output_embed_col=args.output_embed_col,
        num_workers=args.num_workers,
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main()