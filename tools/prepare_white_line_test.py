#!/usr/bin/env python3
"""Create two isolated server-data variants for Android device comparison.

Both variants use the repaired region map from the white-speck experiment. Variant A keeps the
published line art. Variant B removes line pixels that are not supported by a dark stroke in the
original colored artwork, which drops generated color-split borders while retaining drawn ink.
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


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "hd_anime_midnight_magic_train"
SOURCE = ROOT / "artworks/anime" / SOURCE_ID
WHITE_DATA = SOURCE / "paint_data_v2.json"
WHITE_MAP = SOURCE / "region_map_v2.png"
VARIANTS = (
    ("test_whitefix_current_lines", "TEST A - White fix, current lines", False),
    ("test_whitefix_clean_lines", "TEST B - White fix + clean lines", True),
)
INK_THRESHOLD = 8.0
INK_BLUR_SIGMA = 2.2
INK_SUPPORT_RADIUS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def prepare_variant(artwork: dict, artwork_id: str, title: str, clean_lines: bool) -> dict:
    target = ROOT / "artworks/anime" / artwork_id
    target.mkdir(parents=True, exist_ok=True)
    for name in ("image.jpg", "original.jpg", "thumbnail.jpg"):
        shutil.copy2(SOURCE / name, target / name)
    shutil.copy2(WHITE_MAP, target / "region_map.png")

    data = json.loads(WHITE_DATA.read_text(encoding="utf-8"))
    data["contentVersion"] = 1
    experiment = {
        "sourceArtworkId": SOURCE_ID,
        "whiteRegions": "joined-to-color-matched-neighbours",
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
            "contentVersion": 1,
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
    test_ids = {artwork_id for artwork_id, _, _ in VARIANTS}
    catalog["artworks"] = [
        *(prepare_variant(source, *variant) for variant in VARIANTS),
        *(artwork for artwork in catalog["artworks"] if artwork["id"] not in test_ids),
    ]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
