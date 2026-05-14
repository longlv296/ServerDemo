"""
Pixel Art Preview Generator
============================
Converts source images in ServerDemo/image/[category]/ into:
  1. Color-quantized pixel art preview PNGs (64-col grid, 20 colors, at 2x scale)
  2. Per-image pixel array data JSON files  (_data.json)
  3. A master puzzles.json catalog at the repo root

Usage:
    python tools/generate_previews.py

Ported K-Means quantization logic from ImageToPuzzleConverter.kt
"""

import json
import os
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────
GRID_SIZE = 64          # number of columns in the pixel grid
PIXEL_SCALE = 2         # each grid cell → 2x2 px in preview PNG
DEFAULT_COLOR_COUNT = 12 # default palette size for coloring (stored in JSON)
MAX_KMEANS_ITER = 20
CATEGORIES = ["abstract", "aieditor", "aitools", "anime", "cyberpunk", "fantasy", "minimal", "nature", "space","icon"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IMAGE_DIR = REPO_ROOT / "image"
OUTPUT_JSON = REPO_ROOT / "puzzles.json"


# ── K-Means++ Initialization ────────────────────────────────────────────
def kmeans_pp_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means++ seeding — same logic as ImageToPuzzleConverter.kt"""
    n = data.shape[0]
    centroids = np.zeros((k, 3), dtype=np.float32)
    centroids[0] = data[rng.integers(n)]

    for i in range(1, k):
        # distance of every point to nearest existing centroid
        dists = np.min(
            np.sum((data[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
            axis=1,
        )
        total = dists.sum()
        if total == 0:
            centroids[i] = data[rng.integers(n)]
            continue
        probs = dists / total
        idx = rng.choice(n, p=probs)
        centroids[i] = data[idx]

    return centroids


def kmeans_quantize(pixels: np.ndarray, k: int, max_iter: int = MAX_KMEANS_ITER) -> np.ndarray:
    """
    K-Means color quantization.
    pixels: (N, 3) float32 RGB
    Returns: (k, 3) float32 centroids
    """
    rng = np.random.default_rng(42)
    unique = np.unique(pixels, axis=0)

    if len(unique) <= k:
        return unique.astype(np.float32)

    centroids = kmeans_pp_init(unique, k, rng)

    for _ in range(max_iter):
        # assign
        diffs = pixels[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # update
        new_centroids = np.copy(centroids)
        converged = True
        for c in range(k):
            mask = labels == c
            if mask.any():
                mean = pixels[mask].mean(axis=0)
                moved = np.sum((new_centroids[c] - mean) ** 2)
                if moved > 1.0:
                    converged = False
                new_centroids[c] = mean

        centroids = new_centroids
        if converged:
            break

    return np.clip(centroids, 0, 255).astype(np.float32)

PREVIEW_COLOR_COUNT = 20  # number of colors for preview quantization


def convert_to_color_pixel_art(img: Image.Image, grid_size: int, color_count: int = PREVIEW_COLOR_COUNT):
    """
    Convert an image to color-quantized pixel art.
    Returns:
        grid_w, grid_h: grid dimensions
        color_ids: list[int] — 0 = transparent, 1..N = palette colors
        palette: dict[int, str] — colorId → hex color (e.g. "#FF6B35")
        preview_image: PIL Image (RGBA) of the rendered pixel art
    """
    w, h = img.size
    aspect = h / w
    grid_w = grid_size
    grid_h = max(1, int(grid_size * aspect))

    # Downscale to grid dimensions (nearest neighbor for pixel look)
    scaled = img.resize((grid_w, grid_h), Image.Resampling.NEAREST)

    # Detect transparency
    has_alpha = scaled.mode in ("RGBA", "LA", "PA")
    alpha_mask = None

    if has_alpha:
        # Extract alpha channel before converting to RGB
        rgba = scaled.convert("RGBA")
        alpha_data = np.array(rgba)[:, :, 3].flatten()
        alpha_mask = alpha_data < 128  # True = transparent

        # Composite onto white for RGB extraction (opaque pixels only)
        background = Image.new("RGB", scaled.size, (255, 255, 255))
        background.paste(scaled, mask=scaled.split()[-1])
        scaled_rgb = background
    else:
        scaled_rgb = scaled.convert("RGB")

    # Extract pixel data
    all_pixels = np.array(scaled_rgb, dtype=np.float32).reshape(-1, 3)

    # Separate opaque pixels for K-Means
    if alpha_mask is not None:
        opaque_pixels = all_pixels[~alpha_mask]
    else:
        opaque_pixels = all_pixels

    if len(opaque_pixels) == 0:
        # Entirely transparent image
        color_ids = [0] * (grid_w * grid_h)
        palette = {}
        preview = Image.new("RGBA", (grid_w * PIXEL_SCALE, grid_h * PIXEL_SCALE), (0, 0, 0, 0))
        return grid_w, grid_h, color_ids, palette, preview

    # K-Means with color_count colors (on opaque pixels only)
    centroids = kmeans_quantize(opaque_pixels, k=min(color_count, len(np.unique(opaque_pixels, axis=0))))

    # Assign each pixel to nearest centroid
    diffs = all_pixels[:, None, :] - centroids[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    labels = np.argmin(dists, axis=1)

    # Build palette: colorId (1-based) → hex color
    palette = {}
    for idx, c in enumerate(centroids):
        r, g, b = int(np.clip(c[0], 0, 255)), int(np.clip(c[1], 0, 255)), int(np.clip(c[2], 0, 255))
        palette[idx + 1] = f"#{r:02X}{g:02X}{b:02X}"

    # Build color_ids: transparent → 0, opaque → 1-based label
    color_ids = []
    for i in range(len(all_pixels)):
        if alpha_mask is not None and alpha_mask[i]:
            color_ids.append(0)  # transparent
        else:
            color_ids.append(int(labels[i]) + 1)  # 1-based

    # Render pixel art preview image in "unfilled" grayscale style
    # Uses the same formula as ColoringScreen.kt for consistency:
    #   lum = 0.299*R + 0.587*G + 0.114*B
    #   gray = 180 + (lum * 60 / 255)  → range [180, 240]
    preview_w = grid_w * PIXEL_SCALE
    preview_h = grid_h * PIXEL_SCALE
    preview = Image.new("RGBA", (preview_w, preview_h), (0, 0, 0, 0))
    preview_pixels = preview.load()

    for row in range(grid_h):
        for col in range(grid_w):
            idx = row * grid_w + col
            cid = color_ids[idx]
            if cid == 0:
                color = (0, 0, 0, 0)  # transparent
            else:
                c = centroids[cid - 1]
                r, g, b = float(c[0]), float(c[1]), float(c[2])
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                gray = int(180 + (lum * 60 / 255))
                gray = max(180, min(240, gray))
                color = (gray, gray, gray, 255)
            # Fill the PIXEL_SCALE x PIXEL_SCALE block
            for dy in range(PIXEL_SCALE):
                for dx in range(PIXEL_SCALE):
                    preview_pixels[col * PIXEL_SCALE + dx, row * PIXEL_SCALE + dy] = color

    return grid_w, grid_h, color_ids, palette, preview


def process_image(image_path: Path, category: str) -> dict | None:
    """Process a single image and generate preview + data."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"  [FAIL] Cannot open {image_path.name}: {e}")
        return None

    stem = image_path.stem  # e.g. "anime_01"
    parent = image_path.parent

    print(f"  -> Processing {image_path.name}...", end=" ")

    grid_w, grid_h, color_ids, palette, preview_img = convert_to_color_pixel_art(img, GRID_SIZE)

    # Save preview PNG
    preview_path = parent / f"{stem}_preview.png"
    preview_img.save(preview_path, "PNG", optimize=True)

    # Save pixel array data JSON (includes palette for color rendering)
    data_path = parent / f"{stem}_data.json"
    data_json = {
        "id": stem,
        "width": grid_w,
        "height": grid_h,
        "colorIds": color_ids,
        "palette": palette,
    }
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, separators=(",", ":"))

    num_colors = len(palette)
    print(f"OK ({grid_w}x{grid_h}, {num_colors} colors, preview: {preview_path.name})")

    # Determine color count suggestion based on image complexity
    img_rgb = img.convert("RGBA") if img.mode in ("RGBA", "LA", "PA") else img.convert("RGB")
    img_small = img_rgb.resize((64, 64), Image.Resampling.LANCZOS).convert("RGB")
    unique_colors = len(set(img_small.getdata()))
    suggested_colors = min(max(8, unique_colors // 20), 20)

    return {
        "id": stem,
        "category": category,
        "imageUrl": f"image/{category}/{image_path.name}",
        "previewUrl": f"image/{category}/{stem}_preview.png",
        "dataUrl": f"image/{category}/{stem}_data.json",
        "gridSize": GRID_SIZE,
        "colorCount": suggested_colors,
        "width": grid_w,
        "height": grid_h,
        "isPremium": False,
        "isMystics": False,
        "isJigsaw": False,
        "jigsawGrid": None,
        "jigsawAdSlots": None,
    }


def cleanup_orphans(cat_dir: Path, source_stems: set[str]):
    """
    Remove _preview.png and _data.json files whose source image no longer exists.
    source_stems: set of stems of current source images (e.g. {"abstract_01", "abstract_02"})
    """
    removed = 0
    for f in cat_dir.iterdir():
        # Match generated files: *_preview.png and *_data.json
        if f.name.endswith("_preview.png"):
            base_stem = f.stem.removesuffix("_preview")
            if base_stem not in source_stems:
                f.unlink()
                print(f"  [CLEAN] Removed orphan: {f.name}")
                removed += 1
        elif f.name.endswith("_data.json"):
            base_stem = f.stem.removesuffix("_data")
            if base_stem not in source_stems:
                f.unlink()
                print(f"  [CLEAN] Removed orphan: {f.name}")
                removed += 1
    return removed


def main():
    print("=" * 60)
    print("Pixel Art Preview Generator")
    print("=" * 60)

    catalog = {"categories": []}
    total_cleaned = 0

    for cat_name in CATEGORIES:
        cat_dir = IMAGE_DIR / cat_name
        if not cat_dir.is_dir():
            print(f"\n[WARN] Category directory not found: {cat_dir}")
            continue

        print(f"\n[DIR] Category: {cat_name}")

        # Collect original images (exclude _preview.png, _data.json, and existing _hash.png)
        images = sorted([
            f for f in cat_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
            and "_preview" not in f.stem
            and not (f.suffix.lower() == ".png" and "_" in f.stem and len(f.stem.split("_")[-1]) == 8)
        ])

        # Cleanup orphaned preview/data files from deleted images
        source_stems = {img.stem for img in images}
        total_cleaned += cleanup_orphans(cat_dir, source_stems)

        if not images:
            print("  (no images found)")
            continue

        puzzles = []
        for img_path in images:
            result = process_image(img_path, cat_name)
            if result:
                puzzles.append(result)

        catalog["categories"].append({
            "id": cat_name,
            "name": cat_name.capitalize(),
            "puzzles": puzzles,
        })

    # Write master catalog
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    total_puzzles = sum(len(c["puzzles"]) for c in catalog["categories"])
    print(f"\n{'=' * 60}")
    print(f"[DONE] Generated {total_puzzles} previews across {len(catalog['categories'])} categories.")
    if total_cleaned > 0:
        print(f"[CLEAN] Removed {total_cleaned} orphaned files.")
    print(f"[FILE] Catalog written to: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
