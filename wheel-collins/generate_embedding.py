import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from PIL import Image
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class EmbeddingGenerator:
    """Generate image embeddings using Hugging Face DINOv2 for local images."""

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
        image_path_col: str = "LocalPath",
        output_embed_col: str = "ImageVectorsM1",
        num_workers: int = 16,
        cache_path: str | None = None,
    ):
        """
        Generate DINOv2 embeddings for local images:
        - Load images with a thread pool (bounded by num_workers).
        - Embed each image as soon as it finishes loading to keep memory low.
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
                    print(f"Warning: failed to load {path}: {e}")
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
        """Open image from local file path."""
        path_str = str(path).strip()
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")
        return Image.open(path_str)


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
        default="LocalPath",
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
        help="Number of parallel workers for loading images (default: 8). Increase for faster I/O.",
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