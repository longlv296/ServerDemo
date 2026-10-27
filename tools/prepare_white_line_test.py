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
CONTENT_VERSION = 4
INK_THRESHOLD = 8.0
INK_BLUR_SIGMA = 2.2
INK_SUPPORT_RADIUS = 2
LINE_SCALE = 4
LINE_RELATIVE_THRESHOLD = 0.5
LINE_MIN_ALPHA = 36
LINE_MAX_ALPHA = 120
LINE_SMOOTH_SIGMA = 0.8
LINE_CLOSE_RADIUS = 1
LINE_MIN_SPECK_AREA = 4.0
LINE_MIN_HOLE_AREA = 6.0
MERGE_MIN_OPEN_PIXELS = 4
MERGE_MIN_OPEN_SHARE = 0.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape(np.shape(rgb))


def fill_small_line_holes(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    if count <= 1:
        return mask
    fill = stats[:, cv2.CC_STAT_AREA] < LINE_MIN_HOLE_AREA * LINE_SCALE * LINE_SCALE
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill[border] = False
    fill[0] = False
    result = mask.copy()
    result[fill[labels]] = 1
    return result


def source_line_pixels(line_art_path: Path) -> np.ndarray:
    """The data pixels covered by the original vectorized lines.

    This mirrors the max-stroke-width=0 mask used by the artwork generator. The mask is needed
    even for the clean-line variant: hidden borders still need their two color regions to meet at
    the former line centre instead of leaving a strip owned entirely by one side.
    """
    alpha = np.asarray(Image.open(line_art_path).convert("RGBA"), dtype=np.uint8)[..., 3]
    height, width = alpha.shape
    upscaled = cv2.resize(
        alpha.astype(np.float32),
        (width * LINE_SCALE, height * LINE_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    upscaled = cv2.GaussianBlur(upscaled, (0, 0), LINE_SMOOTH_SIGMA * LINE_SCALE)
    nearby = cv2.dilate(upscaled, np.ones((2 * LINE_SCALE + 1, 2 * LINE_SCALE + 1), np.uint8))
    threshold = np.clip(nearby * LINE_RELATIVE_THRESHOLD, LINE_MIN_ALPHA, LINE_MAX_ALPHA)
    mask = (upscaled >= threshold).astype(np.uint8)
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * LINE_CLOSE_RADIUS + 1, 2 * LINE_CLOSE_RADIUS + 1),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk)
    _, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= LINE_MIN_SPECK_AREA * LINE_SCALE * LINE_SCALE
    keep[0] = False
    mask = fill_small_line_holes(keep[components].astype(np.uint8))
    coverage = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return coverage > 0


def refresh_region_geometry(data: dict, labels: np.ndarray) -> None:
    areas = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    spans = ndimage.find_objects(labels)
    for region in data["regions"]:
        region_id = region["id"]
        span = spans[region_id - 1]
        region["area"] = int(areas[region_id])
        region["bounds"] = {
            "left": span[1].start,
            "top": span[0].start,
            "right": span[1].stop,
            "bottom": span[0].stop,
        }


def merge_same_color_without_line(data: dict, labels: np.ndarray, line_art_path: Path) -> dict:
    """Merge touching same-color ids only where the final page leaves an open boundary.

    A few generated split lines disappear in the clean-line variant. Keeping the two ids after
    hiding their separator makes one visible shape require two taps. A short antialiasing pinhole
    must not merge otherwise separate paths, so an opening must cover both a minimum pixel length
    and a meaningful share of the common boundary.
    """
    line = source_line_pixels(line_art_path)
    color_of = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for region in data["regions"]:
        color_of[region["id"]] = region["colorNumber"]

    boundary: dict[tuple[int, int], list[int]] = {}
    comparisons = (
        (labels[:, :-1], labels[:, 1:], line[:, :-1], line[:, 1:]),
        (labels[:-1], labels[1:], line[:-1], line[1:]),
    )
    for first, second, first_line, second_line in comparisons:
        same_color_edge = (
            (first != second)
            & (first > 0)
            & (second > 0)
            & (color_of[first] == color_of[second])
        )
        left = first[same_color_edge]
        right = second[same_color_edge]
        open_edge = ~(first_line[same_color_edge] | second_line[same_color_edge])
        for first_id, second_id, is_open in zip(left.tolist(), right.tolist(), open_edge.tolist()):
            pair = (first_id, second_id) if first_id < second_id else (second_id, first_id)
            counts = boundary.setdefault(pair, [0, 0])
            counts[0] += 1
            counts[1] += int(is_open)

    parent = np.arange(len(color_of), dtype=np.int32)

    def root(region_id: int) -> int:
        while parent[region_id] != region_id:
            parent[region_id] = parent[parent[region_id]]
            region_id = int(parent[region_id])
        return region_id

    accepted = []
    for pair, (total, opened) in boundary.items():
        if opened < MERGE_MIN_OPEN_PIXELS or opened / total < MERGE_MIN_OPEN_SHARE:
            continue
        first_root, second_root = root(pair[0]), root(pair[1])
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)
        accepted.append({"ids": list(pair), "boundary": total, "open": opened})

    lookup = np.array([root(region_id) for region_id in range(len(parent))], dtype=np.int32)
    merged = lookup[labels]
    removed = {region["id"] for region in data["regions"] if lookup[region["id"]] != region["id"]}
    data["regions"] = [region for region in data["regions"] if region["id"] not in removed]
    if "prefilledRegions" in data:
        data["prefilledRegions"] = sorted({int(lookup[region_id]) for region_id in data["prefilledRegions"]})
    labels[:] = merged
    refresh_region_geometry(data, labels)
    return {
        "strategy": "touching-same-color-with-open-final-line",
        "minimumOpenPixels": MERGE_MIN_OPEN_PIXELS,
        "minimumOpenShare": MERGE_MIN_OPEN_SHARE,
        "mergedPairs": len(accepted),
        "removedRegionIds": len(removed),
        "pairs": accepted,
    }


def preserve_paper_paths(
    data: dict,
    map_path: Path,
    image_path: Path,
    line_art_path: Path,
) -> tuple[dict, np.ndarray, dict]:
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
    first_added_id = next_id
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

    # The old map assigned every pixel under a surrounding line to a large paintable region,
    # because the small paths were still id 0 at that point. Re-run that ownership decision with
    # the new paths as anchors, but change only pixels whose nearest owner is a new path. Existing
    # region-to-region boundaries therefore remain byte-for-byte untouched.
    under_lines = source_line_pixels(line_art_path)
    new_path_pixels = labels >= first_added_id
    anchors = ((labels > 0) & ~under_lines) | new_path_pixels
    _, nearest = cv2.distanceTransformWithLabels(
        (~anchors).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    rows, columns = np.nonzero(anchors)
    owner = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    owner[nearest[rows, columns]] = labels[rows, columns]
    nearest_owner = owner[nearest]
    claimed_boundary = under_lines & (nearest_owner >= first_added_id) & ~new_path_pixels
    labels[claimed_boundary] = nearest_owner[claimed_boundary]

    result = copy.deepcopy(data)
    result["regions"].extend(added)
    refresh_region_geometry(result, labels)
    result["contentVersion"] = CONTENT_VERSION
    stats = {
        "strategy": "separate-source-colored-paths-with-centered-boundaries",
        "sourcePaperComponents": count,
        "newRegions": len(added),
        "boundaryPixelsReassigned": int(claimed_boundary.sum()),
        "pathsExpandedUnderLines": int(np.unique(labels[claimed_boundary]).size),
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

    data = copy.deepcopy(path_data)
    labels = path_labels.copy()
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
    merge_stats = merge_same_color_without_line(data, labels, target / "line_art.png")
    data.setdefault("generation", {})["sameColorPathMerges"] = merge_stats
    experiment["sameColorPathMerges"] = merge_stats
    write_region_map(labels, target / "region_map.png")
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
        source_data,
        SOURCE / "region_map.png",
        SOURCE / "image.jpg",
        SOURCE / "line_art.png",
    )
    test_ids = {artwork_id for artwork_id, _, _ in VARIANTS}
    catalog["artworks"] = [
        *(prepare_variant(source, path_data, path_labels, path_stats, *variant) for variant in VARIANTS),
        *(artwork for artwork in catalog["artworks"] if artwork["id"] not in test_ids),
    ]
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
