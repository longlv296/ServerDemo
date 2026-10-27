#!/usr/bin/env python3
"""Create two isolated server-data variants for Android device comparison.

Every closed paper path keeps its own region id and takes only its color number from the source
artwork. This prevents a tap on one large region from painting disconnected small paths. Variant A
keeps the published line art. Variant B removes generated color-split borders while retaining ink
that is supported by the original colored artwork.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "hd_anime_midnight_magic_train"
SOURCE = ROOT / "artworks/anime" / SOURCE_ID
VARIANTS = (
    ("test_whitefix_current_lines", "TEST A - Paths fixed, current lines", False),
    ("test_whitefix_clean_lines", "TEST B - Paths fixed + clean lines", True),
)
CONTENT_VERSION = 2
INK_THRESHOLD = 8.0
INK_BLUR_SIGMA = 2.2
INK_SUPPORT_RADIUS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape(np.shape(rgb))


def preserve_paper_paths(data: dict, map_path: Path, image_path: Path) -> tuple[dict, np.ndarray, dict]:
    """Turn each id-0 component into its own region, colored from its source pixels.

    The old repair assigned an entire component to an existing region of a similar color. That
    removed white paper, but also made one tap paint unrelated closed paths. Keeping a unique id is
    the important part here; color similarity decides only the number printed inside that path.
    """
    labels = region_ids(map_path)
    paper, count = ndimage.label(labels == 0)
    objects = ndimage.find_objects(paper)
    with Image.open(image_path) as opened:
        picture = np.asarray(opened.convert("RGB").resize(labels.shape[::-1], Image.Resampling.LANCZOS))
    picture_lab = rgb_to_lab(picture)
    palette_numbers = np.array([entry["number"] for entry in data["palette"]], dtype=np.int32)
    palette_lab = rgb_to_lab(np.array([entry["rgb"] for entry in data["palette"]], dtype=np.uint8))

    next_id = max(int(labels.max()), max(region["id"] for region in data["regions"])) + 1
    added = []
    for component, box in enumerate(objects, start=1):
        if box is None:
            continue
        mask = paper[box] == component
        distance = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        peak = float(distance.max())
        # The centre is less affected by antialiased ink; very thin paths fall back to all pixels.
        core = mask & (distance >= max(1.5, peak * 0.45))
        samples = picture_lab[box][core if core.any() else mask]
        source_color = np.median(samples, axis=0)
        color_number = int(palette_numbers[np.argmin(np.linalg.norm(palette_lab - source_color, axis=1))])
        local_y, local_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        top, left = box[0].start, box[1].start
        bottom, right = box[0].stop, box[1].stop
        labels[box][mask] = next_id
        added.append(
            {
                "id": next_id,
                "colorNumber": color_number,
                "area": int(mask.sum()),
                "label": {"x": int(left + local_x), "y": int(top + local_y)},
                "labelRadius": round(peak, 1),
                "bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
            }
        )
        next_id += 1

    result = copy.deepcopy(data)
    result["regions"].extend(added)
    result["contentVersion"] = CONTENT_VERSION
    stats = {
        "strategy": "separate-source-colored-paths",
        "sourcePaperComponents": count,
        "newRegions": len(added),
        "paperPixelsAfter": int((labels == 0).sum()),
    }
    result.setdefault("generation", {})["paperPaths"] = stats
    return result, labels, stats


def write_region_map(labels: np.ndarray, target: Path) -> None:
    rgb = np.dstack((labels & 255, (labels >> 8) & 255, (labels >> 16) & 255)).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(target, optimize=True)


def write_thumbnail(line_art: Path, target: Path) -> None:
    with Image.open(line_art) as opened:
        rgba = opened.convert("RGBA")
    page = Image.new("RGBA", rgba.size, "white")
    page.alpha_composite(rgba)
    height = max(1, round(page.height * 512 / page.width))
    page.convert("L").resize((512, height), Image.Resampling.LANCZOS).save(
        target, "WEBP", quality=80, method=6
    )


def clean_generated_splits(line_art: Path, original: Path, target: Path) -> dict[str, float]:
    with Image.open(line_art) as opened:
        rgba = np.asarray(opened.convert("RGBA")).copy()
    with Image.open(original) as opened:
        rgb = np.asarray(opened.convert("RGB").resize((rgba.shape[1], rgba.shape[0])))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_light = cv2.GaussianBlur(gray, (0, 0), INK_BLUR_SIGMA)
    drawn_ink = (local_light - gray >= INK_THRESHOLD).astype(np.uint8)
    size = INK_SUPPORT_RADIUS * 2 + 1
    support = cv2.dilate(
        drawn_ink,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    support = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 0.7)

    before = rgba[..., 3].astype(np.float32)
    rgba[..., 3] = np.round(before * np.clip(support, 0.0, 1.0)).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(target, optimize=True)
    return {
        "alphaBefore": round(float(before.sum()), 1),
        "alphaAfter": round(float(rgba[..., 3].sum()), 1),
        "removedShare": round(1.0 - float(rgba[..., 3].sum()) / max(float(before.sum()), 1.0), 4),
    }


def prepare_variant(
    artwork: dict,
    path_data: dict,
    path_labels: np.ndarray,
    path_stats: dict,
    artwork_id: str,
    title: str,
    clean_lines: bool,
) -> dict:
    target = ROOT / "artworks/anime" / artwork_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ("image.jpg", "original.jpg", "thumbnail.jpg"):
        shutil.copy2(SOURCE / name, target / name)
    write_region_map(path_labels, target / "region_map.png")

    data = copy.deepcopy(path_data)
    experiment = {
        "sourceArtworkId": SOURCE_ID,
        "whiteRegions": path_stats["strategy"],
        "lines": "original-ink-only" if clean_lines else "current",
    }
    if clean_lines:
        experiment["lineCleanup"] = clean_generated_splits(
            SOURCE / "line_art.png", SOURCE / "original.jpg", target / "line_art.png"
        )
    else:
        shutil.copy2(SOURCE / "line_art.png", target / "line_art.png")
        shutil.copy2(SOURCE / "line_paths_v3.json", target / "line_paths_v3.json")
    write_thumbnail(target / "line_art.png", target / "line_art_thumb.webp")
    data.setdefault("generation", {})["comparisonExperiment"] = experiment
    (target / "paint_data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    result = copy.deepcopy(artwork)
    result.update(
        {
            "id": artwork_id,
            "title": title,
            "contentVersion": CONTENT_VERSION,
            "regionCount": len(data["regions"]),
            "isNew": True,
            "addedAt": "2026-10-27",
            "thumbnailUrl": f"artworks/anime/{artwork_id}/thumbnail.jpg",
            "imageUrl": f"artworks/anime/{artwork_id}/image.jpg",
            "originalImageUrl": f"artworks/anime/{artwork_id}/original.jpg",
            "lineArtUrl": f"artworks/anime/{artwork_id}/line_art.png",
            "lineArtThumbnailUrl": f"artworks/anime/{artwork_id}/line_art_thumb.webp",
            "linePathsUrl": f"artworks/anime/{artwork_id}/line_paths_v3.json",
            "regionMapUrl": f"artworks/anime/{artwork_id}/region_map.png",
            "paintDataUrl": f"artworks/anime/{artwork_id}/paint_data.json",
            "regionMapSha256": sha256(target / "region_map.png"),
            "imageSha256": sha256(target / "image.jpg"),
        }
    )
    return result


def main() -> None:
    catalog_path = ROOT / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    source = next(artwork for artwork in catalog["artworks"] if artwork["id"] == SOURCE_ID)
    source_data = json.loads((SOURCE / "paint_data.json").read_text(encoding="utf-8"))
    path_data, path_labels, path_stats = preserve_paper_paths(
        source_data, SOURCE / "region_map.png", SOURCE / "image.jpg"
    )
    test_ids = {artwork_id for artwork_id, _, _ in VARIANTS}
    catalog["artworks"] = [
        *(prepare_variant(source, path_data, path_labels, path_stats, *variant) for variant in VARIANTS),
        *(artwork for artwork in catalog["artworks"] if artwork["id"] not in test_ids),
    ]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
