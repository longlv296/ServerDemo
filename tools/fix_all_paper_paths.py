#!/usr/bin/env python3
"""Repair non-paintable paper paths for every artwork in the server catalog.

Method A keeps the published line art unchanged. Every real id-0 paper component receives its own
region and source-derived color, line pixels are shared back to the nearest path, and raster-sized
antialias fragments are absorbed into the dominant touching region.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
MICRO_PATH_MAX_AREA = 50
MICRO_PATH_MAX_RADIUS = 1.0
LINE_RELATIVE_THRESHOLD = 0.5
LINE_MIN_ALPHA = 36
LINE_MAX_ALPHA = 120
LINE_SMOOTH_SIGMA = 0.8
LINE_CLOSE_RADIUS = 1
LINE_MIN_SPECK_AREA = 4.0
LINE_MIN_HOLE_AREA = 6.0
CATEGORY_PRIORITY = ("christmas", "new_year")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    shaped = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape(np.shape(rgb))


def fill_small_line_holes(mask: np.ndarray, scale: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    if count <= 1:
        return mask
    fill = stats[:, cv2.CC_STAT_AREA] < LINE_MIN_HOLE_AREA * scale * scale
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill[border] = False
    fill[0] = False
    result = mask.copy()
    result[fill[labels]] = 1
    return result


def source_line_pixels(line_art_path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(line_art_path) as opened:
        alpha_image = opened.convert("RGBA").getchannel("A")
        if alpha_image.size != (width, height):
            alpha_image = alpha_image.resize((width, height), Image.Resampling.LANCZOS)
        alpha = np.asarray(alpha_image, dtype=np.uint8)

    # Scale 4 matches the approved 1536px repair. Larger maps use scale 2 to keep peak memory
    # bounded while retaining subpixel coverage at their already higher source resolution.
    scale = 4 if max(width, height) <= 1600 else 2
    upscaled = cv2.resize(
        alpha.astype(np.float32),
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    upscaled = cv2.GaussianBlur(upscaled, (0, 0), LINE_SMOOTH_SIGMA * scale)
    nearby = cv2.dilate(upscaled, np.ones((2 * scale + 1, 2 * scale + 1), np.uint8))
    threshold = np.clip(nearby * LINE_RELATIVE_THRESHOLD, LINE_MIN_ALPHA, LINE_MAX_ALPHA)
    mask = (upscaled >= threshold).astype(np.uint8)
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * LINE_CLOSE_RADIUS + 1, 2 * LINE_CLOSE_RADIUS + 1),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk)
    _, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= LINE_MIN_SPECK_AREA * scale * scale
    keep[0] = False
    mask = fill_small_line_holes(keep[components].astype(np.uint8), scale)
    coverage = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return coverage > 0


def refresh_region_geometry(data: dict, labels: np.ndarray) -> None:
    areas = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    spans = ndimage.find_objects(labels)
    for region in data["regions"]:
        region_id = region["id"]
        span = spans[region_id - 1]
        if span is None:
            raise ValueError(f"Region {region_id} is declared but absent from the map")
        region["area"] = int(areas[region_id])
        region["bounds"] = {
            "left": span[1].start,
            "top": span[0].start,
            "right": span[1].stop,
            "bottom": span[0].stop,
        }


def collapse_microscopic_paper_paths(
    data: dict,
    labels: np.ndarray,
    added_region_ids: set[int],
) -> dict:
    microscopic = {
        region["id"]
        for region in data["regions"]
        if region["id"] in added_region_ids
        and region["area"] <= MICRO_PATH_MAX_AREA
        and region.get("labelRadius", 0.0) <= MICRO_PATH_MAX_RADIUS
    }
    base_stats = {
        "strategy": "absorb-raster-sized-paper-fragments",
        "maximumArea": MICRO_PATH_MAX_AREA,
        "maximumLabelRadius": MICRO_PATH_MAX_RADIUS,
    }
    if not microscopic:
        return {**base_stats, "components": 0, "regionsRemoved": 0, "pixelsReassigned": 0}

    graph = {region_id: set() for region_id in microscopic}
    outside_edges = {region_id: {} for region_id in microscopic}
    microscopic_lookup = np.zeros(int(labels.max()) + 1, dtype=bool)
    microscopic_lookup[list(microscopic)] = True
    comparisons = ((labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:]))
    for first, second in comparisons:
        boundary = (first != second) & (microscopic_lookup[first] | microscopic_lookup[second])
        for first_id, second_id in zip(first[boundary].tolist(), second[boundary].tolist()):
            if first_id in microscopic:
                if second_id in microscopic:
                    graph[first_id].add(second_id)
                elif second_id > 0:
                    outside_edges[first_id][second_id] = outside_edges[first_id].get(second_id, 0) + 1
            if second_id in microscopic:
                if first_id in microscopic:
                    graph[second_id].add(first_id)
                elif first_id > 0:
                    outside_edges[second_id][first_id] = outside_edges[second_id].get(first_id, 0) + 1

    areas = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
    assignments: list[dict] = []
    visited: set[int] = set()
    for start in sorted(microscopic):
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        candidates: dict[int, int] = {}
        while stack:
            region_id = stack.pop()
            if region_id in visited:
                continue
            visited.add(region_id)
            component.add(region_id)
            stack.extend(graph[region_id] - visited)
            for neighbor_id, edge_count in outside_edges[region_id].items():
                candidates[neighbor_id] = candidates.get(neighbor_id, 0) + edge_count
        if not candidates:
            continue
        target_id = max(
            candidates,
            key=lambda region_id: (candidates[region_id], int(areas[region_id]), -region_id),
        )
        for region_id in component:
            lookup[region_id] = target_id
        assignments.append(
            {
                "ids": sorted(component),
                "targetId": target_id,
                "sharedBoundary": candidates[target_id],
                "pixels": int(areas[list(component)].sum()),
            }
        )

    labels[:] = lookup[labels]
    removed = {region_id for assignment in assignments for region_id in assignment["ids"]}
    data["regions"] = [region for region in data["regions"] if region["id"] not in removed]
    if "prefilledRegions" in data:
        data["prefilledRegions"] = [region_id for region_id in data["prefilledRegions"] if region_id not in removed]
    refresh_region_geometry(data, labels)
    largest = sorted(assignments, key=lambda assignment: assignment["pixels"], reverse=True)[:10]
    return {
        **base_stats,
        "components": len(assignments),
        "regionsRemoved": len(removed),
        "pixelsReassigned": sum(assignment["pixels"] for assignment in assignments),
        "largestAssignments": largest,
    }


def repair_artwork(artwork: dict) -> dict | None:
    map_path = ROOT / artwork["regionMapUrl"]
    data_path = ROOT / artwork["paintDataUrl"]
    image_path = ROOT / artwork["imageUrl"]
    line_art_path = ROOT / artwork["lineArtUrl"]
    labels = region_ids(map_path)
    paper_mask = labels == 0
    paper_pixels = int(paper_mask.sum())
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if paper_pixels == 0:
        catalog_version = int(artwork.get("contentVersion", 0))
        data_version = int(data.get("contentVersion", 0))
        if catalog_version != data_version:
            synchronized_version = max(catalog_version, data_version)
            artwork["contentVersion"] = synchronized_version
            data["contentVersion"] = synchronized_version
            temp_data = data_path.with_name(f"{data_path.stem}.tmp.json")
            temp_data.write_text(
                json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temp_data.replace(data_path)
            return {
                "artworkId": artwork["id"],
                "metadataOnly": True,
                "contentVersion": synchronized_version,
            }
        return None

    paper, component_count = ndimage.label(paper_mask)
    objects = ndimage.find_objects(paper)
    height, width = labels.shape
    with Image.open(image_path) as opened:
        picture = np.asarray(opened.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
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

    under_lines = source_line_pixels(line_art_path, width, height)
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
    micro_stats = collapse_microscopic_paper_paths(
        result,
        labels,
        {region["id"] for region in added},
    )
    next_version = max(int(artwork.get("contentVersion", 0)), int(data.get("contentVersion", 0))) + 1
    result["contentVersion"] = next_version
    stats = {
        "strategy": "method-a-preserve-lines-and-paper-paths",
        "sourcePaperComponents": component_count,
        "newRegions": len(added),
        "sourcePaperPixels": paper_pixels,
        "boundaryPixelsReassigned": int(claimed_boundary.sum()),
        "paperPixelsAfter": int((labels == 0).sum()),
        "microscopicPaths": micro_stats,
    }
    result.setdefault("generation", {})["paperPathRepair"] = stats

    rgb = np.dstack((labels & 255, (labels >> 8) & 255, (labels >> 16) & 255)).astype(np.uint8)
    temp_map = map_path.with_name(f"{map_path.stem}.tmp.png")
    Image.fromarray(rgb, "RGB").save(temp_map, compress_level=6)
    temp_map.replace(map_path)
    temp_data = data_path.with_name(f"{data_path.stem}.tmp.json")
    temp_data.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp_data.replace(data_path)

    artwork["contentVersion"] = next_version
    artwork["regionCount"] = len(result["regions"])
    artwork["regionMapSha256"] = sha256(map_path)
    return {
        "artworkId": artwork["id"],
        "contentVersion": next_version,
        "regionCount": len(result["regions"]),
        **stats,
    }


def main() -> None:
    catalog_path = ROOT / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    changed = []
    started = time.monotonic()
    total = len(catalog["artworks"])
    for index, artwork in enumerate(catalog["artworks"], start=1):
        item_started = time.monotonic()
        result = repair_artwork(artwork)
        if result is not None:
            changed.append(result)
            if result.get("metadataOnly"):
                print(
                    f"[{index}/{total}] {artwork['id']}: "
                    f"synchronized contentVersion={result['contentVersion']}",
                    flush=True,
                )
                continue
            micro = result["microscopicPaths"]
            print(
                f"[{index}/{total}] {artwork['id']}: "
                f"paper={result['sourcePaperPixels']} components={result['sourcePaperComponents']} "
                f"micro={micro['regionsRemoved']} regions={result['regionCount']} "
                f"time={time.monotonic() - item_started:.1f}s",
                flush=True,
            )
    category_order = {category_id: index for index, category_id in enumerate(CATEGORY_PRIORITY)}
    catalog["categories"].sort(
        key=lambda category: category_order.get(category["id"], len(category_order))
    )
    temp_catalog = catalog_path.with_name("catalog.tmp.json")
    temp_catalog.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp_catalog.replace(catalog_path)
    print(
        f"Updated {len(changed)}/{total} artworks in {time.monotonic() - started:.1f}s; "
        f"remaining id-0 pixels: "
        f"{sum(item.get('paperPixelsAfter', 0) for item in changed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
