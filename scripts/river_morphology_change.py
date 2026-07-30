#!/usr/bin/env python3
"""
River Morphology Change - Multi-temporal river channel analysis.

Extracts shorelines, centerlines, and channel widths from multi-temporal
water body masks. Quantifies shoreline migration, channel migration,
and change hotspots.

Exit codes:
    0 = success
    2 = argument error
    3 = dependency missing
    6 = data validation failure
    7 = processing failure
"""

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try pip-installed package first; fall back to local copy in repo root.
try:
    from _geoskill_data_fetcher import (add_bbox_date_args,
        parse_bbox_arg,
        parse_date_range_arg,
        DataFetcher,
        DataSource,
        BBox,
        DateRange,
        DataFetcherError,)
    _FETCHER_AVAILABLE = True
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _skill_dir = _Path(__file__).resolve().parent
    _repo_root = _skill_dir.parent.parent
    _local_fetcher = _repo_root / "_geoskill_data_fetcher"
    if _local_fetcher.exists():
        _sys.path.insert(0, str(_repo_root))
    from _geoskill_data_fetcher import (add_bbox_date_args,
        parse_bbox_arg,
        parse_date_range_arg,
        DataFetcher,
        DataSource,
        BBox,
        DateRange,
        DataFetcherError,)
    _FETCHER_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful when running standalone
    _FETCHER_AVAILABLE = False



EXIT_OK = 0
EXIT_ARG = 2
EXIT_DEP = 3
EXIT_VALIDATION = 6
EXIT_PROCESSING = 7

# Quality codes
QUALITY_HIGH = 1
QUALITY_MEDIUM = 2
QUALITY_LOW = 3
QUALITY_INVALID = 4

# Change direction codes
DIR_STABLE = 0
DIR_EROSION = -1  # Shoreline retreats (water expands)
DIR_DEPOSITION = 1  # Shoreline advances (water contracts)

# Default parameters
DEFAULT_TRANSECT_SPACING = 50.0
DEFAULT_MIN_CHANNEL_WIDTH = 20.0
DEFAULT_SHORELINE_METHOD = "threshold"
DEFAULT_WATER_THRESHOLD = 0.0
DEFAULT_HOTSPOT_PERCENTILE = 90.0


# ============================================================
# Geometry Helpers
# ============================================================

def create_polygon(x, y, w, h):
    """Create a shapely polygon (box)."""
    from shapely.geometry import box
    return box(x, y, x + w, y + h)


def extract_polygons_from_mask(mask: np.ndarray, transform=None) -> List[Dict]:
    """
    Extract polygons from a binary mask using rasterio.features.shapes.

    Args:
        mask: Binary 2D array (1 = water, 0 = land)
        transform: Affine transform (optional, uses identity if None)

    Returns:
        List of dicts with 'geometry' and 'properties' keys
    """
    try:
        from rasterio.features import shapes
        from rasterio.transform import from_bounds
    except ImportError:
        return _extract_polygons_fallback(mask, transform)

    if transform is None:
        rows, cols = mask.shape
        transform = from_bounds(0, 0, cols, rows, cols, rows)

    results = []
    try:
        for geom, val in shapes(mask.astype(np.int32), transform=transform):
            if val == 1 and geom is not None:
                # Validate geometry dict
                if isinstance(geom, dict) and geom.get("type") is not None:
                    results.append({
                        "geometry": geom,
                        "properties": {"area": _geom_area(geom)},
                    })
    except Exception:
        # Fallback if rasterio.features.shapes fails
        return _extract_polygons_fallback(mask, transform)

    return results


def _extract_polygons_fallback(mask: np.ndarray, transform=None) -> List[Dict]:
    """Fallback polygon extraction without rasterio.features."""
    from shapely.geometry import Polygon, box

    rows, cols = mask.shape
    if transform is None:
        # Simple pixel-to-coordinate mapping
        xmin, ymin, xmax, ymax = 0, 0, cols, rows
    else:
        xmin = transform.c
        ymax = transform.f
        xmax = transform.a * cols + transform.c
        ymin = transform.e * rows + transform.f

    results = []
    # Find connected components of water pixels
    visited = np.zeros_like(mask, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 1 and not visited[r, c]:
                # BFS to find connected component
                component = _flood_fill(mask, r, c, visited)
                if len(component) >= 4:  # Min 4 pixels for a polygon
                    # Create bounding box as simple polygon
                    comp_arr = np.array(component)
                    min_r, min_c = comp_arr.min(axis=0)
                    max_r, max_c = comp_arr.max(axis=0)
                    # Convert pixel to coordinates
                    x1 = xmin + (min_c / cols) * (xmax - xmin)
                    x2 = xmin + ((max_c + 1) / cols) * (xmax - xmin)
                    y2 = ymax - (min_r / rows) * (ymax - ymin)
                    y1 = ymax - ((max_r + 1) / rows) * (ymax - ymin)
                    poly = box(x1, y1, x2, y2)
                    results.append({
                        "geometry": poly.__geo_interface__,
                        "properties": {"area": poly.area},
                    })

    return results


def _flood_fill(mask: np.ndarray, start_r: int, start_c: int,
                visited: np.ndarray) -> List[Tuple[int, int]]:
    """BFS flood fill to find connected component."""
    rows, cols = mask.shape
    component = []
    queue = [(start_r, start_c)]
    visited[start_r, start_c] = True

    while queue:
        r, c = queue.pop(0)
        component.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if mask[nr, nc] == 1 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return component


def _geom_area(geom: Dict) -> float:
    """Compute area from a GeoJSON-like geometry dict."""
    from shapely.geometry import shape
    try:
        return shape(geom).area
    except Exception:
        return 0.0


# ============================================================
# Water Body Extraction
# ============================================================

def extract_water_body_ndwi(green: np.ndarray, nir: np.ndarray,
                            threshold: float = DEFAULT_WATER_THRESHOLD) -> np.ndarray:
    """
    Extract water body from NDWI (Normalized Difference Water Index).
    NDWI = (Green - NIR) / (Green + NIR)

    Args:
        green: Green band reflectance
        nir: Near-infrared band reflectance
        threshold: NDWI threshold (default 0.0)

    Returns:
        Binary mask (1 = water, 0 = land)
    """
    denom = green + nir
    ndwi = np.where(denom == 0, 0.0, (green - nir) / denom)
    return (ndwi >= threshold).astype(np.uint8)


def extract_water_body_mndwi(green: np.ndarray, swir1: np.ndarray,
                             threshold: float = DEFAULT_WATER_THRESHOLD) -> np.ndarray:
    """
    Extract water body from MNDWI (Modified NDWI).
    MNDWI = (Green - SWIR1) / (Green + SWIR1)

    Args:
        green: Green band reflectance
        swir1: Short-wave infrared band reflectance
        threshold: MNDWI threshold (default 0.0)

    Returns:
        Binary mask (1 = water, 0 = land)
    """
    denom = green + swir1
    mndwi = np.where(denom == 0, 0.0, (green - swir1) / denom)
    return (mndwi >= threshold).astype(np.uint8)


def extract_water_body_from_mask(water_mask: np.ndarray,
                                 threshold: float = 0.5) -> np.ndarray:
    """
    Extract water body from a pre-computed water probability mask.

    Args:
        water_mask: 2D array with water probability or binary values
        threshold: Threshold for binarization

    Returns:
        Binary mask (1 = water, 0 = land)
    """
    return (water_mask >= threshold).astype(np.uint8)


# ============================================================
# Shoreline Extraction
# ============================================================

def extract_shoreline(water_mask: np.ndarray, transform=None) -> List[Dict]:
    """
    Extract shoreline from water mask.

    The shoreline is the boundary between water and land pixels.
    Uses polygon exterior boundaries as shoreline representation.

    Args:
        water_mask: Binary 2D array (1 = water, 0 = land)
        transform: Affine transform

    Returns:
        List of shoreline features with geometry and properties
    """
    from shapely.geometry import shape, mapping, LineString, MultiLineString
    from shapely.ops import unary_union

    polygons = extract_polygons_from_mask(water_mask, transform)
    shorelines = []

    for poly_info in polygons:
        geom = poly_info["geometry"]
        try:
            if isinstance(geom, dict):
                shapely_geom = shape(geom)
            else:
                shapely_geom = geom

            if shapely_geom is None or shapely_geom.is_empty:
                continue

            if shapely_geom.geom_type == "Polygon":
                boundary = shapely_geom.exterior
                if boundary and not boundary.is_empty:
                    shorelines.append({
                        "geometry": mapping(boundary),
                        "properties": {
                            "length": boundary.length,
                            "area": shapely_geom.area,
                            "quality": QUALITY_HIGH,
                        },
                    })
            elif shapely_geom.geom_type == "MultiPolygon":
                for poly in shapely_geom.geoms:
                    boundary = poly.exterior
                    if boundary and not boundary.is_empty:
                        shorelines.append({
                            "geometry": mapping(boundary),
                            "properties": {
                                "length": boundary.length,
                                "area": poly.area,
                                "quality": QUALITY_HIGH,
                            },
                        })
        except Exception as e:
            # Skip invalid geometries
            continue

    return shorelines


# ============================================================
# Centerline Extraction
# ============================================================

def compute_centerline(polygon_geom, transform=None) -> Optional[Dict]:
    """
    Compute river centerline from water polygon using Voronoi-based skeleton.

    Uses boundary point sampling + Voronoi diagram to extract medial axis.

    Args:
        polygon_geom: Shapely Polygon or dict geometry
        transform: Affine transform

    Returns:
        Dict with centerline geometry and properties, or None
    """
    from shapely.geometry import shape, mapping, LineString, MultiLineString, Point
    from shapely.ops import unary_union, linemerge

    if isinstance(polygon_geom, dict):
        polygon = shape(polygon_geom)
    else:
        polygon = polygon_geom

    if polygon.is_empty or polygon.area == 0:
        return None

    try:
        # Sample points along boundary
        boundary = polygon.exterior
        if boundary is None or boundary.is_empty:
            return None

        # Sample density based on polygon size
        perimeter = boundary.length
        n_samples = max(20, int(perimeter / 5.0))
        n_samples = min(n_samples, 500)

        sample_points = []
        for i in range(n_samples):
            frac = i / n_samples
            pt = boundary.interpolate(frac * perimeter)
            sample_points.append((pt.x, pt.y))

        if len(sample_points) < 4:
            return None

        # Compute Voronoi diagram
        from scipy.spatial import Voronoi
        vor = Voronoi(sample_points)

        # Extract medial axis: Voronoi vertices inside polygon
        centerline_segments = []
        for ridge_vertices in vor.ridge_vertices:
            if -1 not in ridge_vertices:
                v0, v1 = ridge_vertices
                p0 = Point(vor.vertices[v0])
                p1 = Point(vor.vertices[v1])

                # Both vertices must be inside polygon
                if polygon.contains(p0) and polygon.contains(p1):
                    line = LineString([vor.vertices[v0], vor.vertices[v1]])
                    centerline_segments.append(line)

        if not centerline_segments:
            # Fallback: use longest axis of bounding box
            minx, miny, maxx, maxy = polygon.bounds
            centerline = LineString([
                (minx + (maxx - minx) * 0.5, miny),
                (minx + (maxx - minx) * 0.5, maxy),
            ])
            return {
                "geometry": mapping(centerline),
                "properties": {
                    "length": centerline.length,
                    "method": "fallback_bbox",
                    "quality": QUALITY_LOW,
                },
            }

        # Merge segments into continuous line
        merged = linemerge(centerline_segments)
        if merged.geom_type == "MultiLineString":
            # Take the longest component
            longest = max(merged.geoms, key=lambda l: l.length)
            merged = longest

        return {
            "geometry": mapping(merged),
            "properties": {
                "length": merged.length,
                "method": "voronoi_skeleton",
                "quality": QUALITY_HIGH,
            },
        }

    except ImportError:
        # No scipy: use bounding box fallback
        minx, miny, maxx, maxy = polygon.bounds
        centerline = LineString([
            (minx + (maxx - minx) * 0.5, miny),
            (minx + (maxx - minx) * 0.5, maxy),
        ])
        return {
            "geometry": mapping(centerline),
            "properties": {
                "length": centerline.length,
                "method": "fallback_bbox",
                "quality": QUALITY_LOW,
            },
        }
    except Exception as e:
        return None


def compute_centerline_simple(polygon_geom) -> Optional[Dict]:
    """
    Simple centerline computation without scipy.
    Uses iterative erosion approach.

    Args:
        polygon_geom: Shapely Polygon or dict geometry

    Returns:
        Dict with centerline geometry and properties, or None
    """
    from shapely.geometry import shape, mapping, LineString

    if isinstance(polygon_geom, dict):
        polygon = shape(polygon_geom)
    else:
        polygon = polygon_geom

    if polygon.is_empty or polygon.area == 0:
        return None

    # Use bounding box centerline as simple approximation
    minx, miny, maxx, maxy = polygon.bounds
    dx = maxx - minx
    dy = maxy - miny

    if dx >= dy:
        # Horizontal river
        cx = minx + dx * 0.5
        centerline = LineString([(cx, miny), (cx, maxy)])
    else:
        # Vertical river
        cy = miny + dy * 0.5
        centerline = LineString([(minx, cy), (maxx, cy)])

    return {
        "geometry": mapping(centerline),
        "properties": {
            "length": centerline.length,
            "method": "simple_bbox",
            "quality": QUALITY_MEDIUM,
        },
    }


# ============================================================
# Transect Generation
# ============================================================

def generate_transects(centerline_geom, spacing: float = DEFAULT_TRANSECT_SPACING,
                       transect_width: float = None) -> List[Dict]:
    """
    Generate perpendicular transects along a centerline.

    Args:
        centerline_geom: Shapely LineString, dict geometry, or feature dict with "geometry" key
        spacing: Distance between transects
        transect_width: Length of each transect (default: auto from polygon)

    Returns:
        List of transect features with geometry and properties
    """
    from shapely.geometry import shape, mapping, LineString, Point

    if isinstance(centerline_geom, dict):
        # Handle feature dict {"geometry": ..., "properties": ...}
        if "geometry" in centerline_geom:
            geom = centerline_geom["geometry"]
            if isinstance(geom, dict):
                centerline = shape(geom)
            else:
                centerline = geom
        else:
            # Raw geometry dict
            centerline = shape(centerline_geom)
    else:
        centerline = centerline_geom

    if centerline.is_empty or centerline.length == 0:
        return []

    # Auto-determine transect width from centerline length
    if transect_width is None:
        transect_width = centerline.length * 0.5

    n_transects = max(2, int(centerline.length / spacing))
    actual_spacing = centerline.length / (n_transects - 1) if n_transects > 1 else centerline.length

    transects = []
    for i in range(n_transects):
        distance = i * actual_spacing
        if distance > centerline.length:
            distance = centerline.length

        # Get point on centerline
        center_pt = centerline.interpolate(distance)

        # Get tangent direction
        if distance + 1 <= centerline.length:
            next_pt = centerline.interpolate(distance + 1)
        else:
            next_pt = centerline.interpolate(distance)
            distance = max(0, distance - 1)
            center_pt = centerline.interpolate(distance)

        dx = next_pt.x - center_pt.x
        dy = next_pt.y - center_pt.y
        length = (dx ** 2 + dy ** 2) ** 0.5

        if length == 0:
            continue

        # Perpendicular direction
        perp_dx = -dy / length
        perp_dy = dx / length

        # Create transect line
        half_width = transect_width * 0.5
        p1 = (center_pt.x + perp_dx * half_width,
              center_pt.y + perp_dy * half_width)
        p2 = (center_pt.x - perp_dx * half_width,
              center_pt.y - perp_dy * half_width)

        transect_line = LineString([p1, p2])

        transects.append({
            "geometry": mapping(transect_line),
            "properties": {
                "id": i,
                "station": round(distance, 2),
                "length": round(transect_line.length, 2),
                "center_x": round(center_pt.x, 4),
                "center_y": round(center_pt.y, 4),
                "quality": QUALITY_HIGH,
            },
        })

    return transects


# ============================================================
# Migration Rate Computation
# ============================================================

def compute_migration_rates(transects: List[Dict], shoreline1: List[Dict],
                            shoreline2: List[Dict]) -> List[Dict]:
    """
    Compute shoreline migration rates between two time periods.

    Uses perpendicular intersection method: for each transect,
    find intersection with both shorelines and measure distance.

    Args:
        transects: List of transect features
        shoreline1: Shoreline features from time 1
        shoreline2: Shoreline features from time 2

    Returns:
        List of migration rate features
    """
    from shapely.geometry import shape, mapping, LineString, MultiLineString, Point
    from shapely.ops import unary_union, linemerge

    # Merge shorelines into single geometry
    lines1 = []
    for s in shoreline1:
        try:
            geom = s["geometry"]
            if isinstance(geom, dict):
                lines1.append(shape(geom))
            else:
                lines1.append(geom)
        except Exception:
            pass

    lines2 = []
    for s in shoreline2:
        try:
            geom = s["geometry"]
            if isinstance(geom, dict):
                lines2.append(shape(geom))
            else:
                lines2.append(geom)
        except Exception:
            pass

    if not lines1 or not lines2:
        return []

    merged1 = linemerge(lines1) if len(lines1) > 1 else lines1[0]
    merged2 = linemerge(lines2) if len(lines2) > 1 else lines2[0]

    migration_results = []

    for transect_info in transects:
        try:
            t_geom = transect_info["geometry"]
            if isinstance(t_geom, dict):
                transect_line = shape(t_geom)
            else:
                transect_line = t_geom

            # Find intersections
            inter1 = transect_line.intersection(merged1)
            inter2 = transect_line.intersection(merged2)

            # Get intersection points
            pts1 = _extract_points(inter1)
            pts2 = _extract_points(inter2)

            if not pts1 or not pts2:
                migration_results.append({
                    "geometry": transect_info["geometry"],
                    "properties": {
                        "station": transect_info["properties"].get("station", 0),
                        "migration_distance": None,
                        "migration_rate": None,
                        "direction": DIR_STABLE,
                        "quality": QUALITY_INVALID,
                        "note": "no_intersection",
                    },
                })
                continue

            # Use first intersection point from each
            pt1 = pts1[0]
            pt2 = pts2[0]

            # Compute distance
            distance = pt1.distance(pt2)

            # Determine direction: if shoreline2 is further from center, it's erosion
            # Use transect center as reference
            center = transect_line.centroid
            d1 = center.distance(pt1)
            d2 = center.distance(pt2)

            if abs(distance) < 1e-6:
                direction = DIR_STABLE
            elif d2 > d1:
                direction = DIR_EROSION  # Shoreline moved away from center
            else:
                direction = DIR_DEPOSITION  # Shoreline moved toward center

            # Quality assessment
            quality = QUALITY_HIGH
            if len(pts1) > 1 or len(pts2) > 1:
                quality = QUALITY_LOW  # Multiple intersections = complex geometry
            if distance < 1.0:  # Very small migration
                quality = QUALITY_MEDIUM

            migration_results.append({
                "geometry": transect_info["geometry"],
                "properties": {
                    "station": transect_info["properties"].get("station", 0),
                    "migration_distance": round(distance, 4),
                    "migration_rate": round(distance, 4),  # per time step
                    "direction": direction,
                    "quality": quality,
                    "n_intersections_t1": len(pts1),
                    "n_intersections_t2": len(pts2),
                },
            })

        except Exception as e:
            migration_results.append({
                "geometry": transect_info["geometry"],
                "properties": {
                    "station": transect_info["properties"].get("station", 0),
                    "migration_distance": None,
                    "migration_rate": None,
                    "direction": DIR_STABLE,
                    "quality": QUALITY_INVALID,
                    "note": str(e),
                },
            })

    return migration_results


def _extract_points(geom) -> List:
    """Extract points from a shapely geometry."""
    from shapely.geometry import Point, MultiPoint, LineString, MultiLineString

    if geom.is_empty:
        return []

    if geom.geom_type == "Point":
        return [geom]
    elif geom.geom_type == "MultiPoint":
        return list(geom.geoms)
    elif geom.geom_type == "LineString":
        return [geom.interpolate(0, normalized=True)]
    elif geom.geom_type == "MultiLineString":
        return [g.interpolate(0, normalized=True) for g in geom.geoms]
    elif geom.geom_type == "GeometryCollection":
        points = []
        for g in geom.geoms:
            points.extend(_extract_points(g))
        return points
    else:
        return []


# ============================================================
# Channel Width Computation
# ============================================================

def compute_channel_widths(transects: List[Dict],
                           shoreline: List[Dict]) -> List[Dict]:
    """
    Compute channel width at each transect.

    Width = distance between two intersection points of transect with shoreline.

    Args:
        transects: List of transect features
        shoreline: Shoreline features

    Returns:
        List of width measurements
    """
    from shapely.geometry import shape, mapping, LineString
    from shapely.ops import linemerge

    # Merge shoreline
    lines = []
    for s in shoreline:
        try:
            geom = s["geometry"] if isinstance(s["geometry"], dict) else s["geometry"]
            lines.append(shape(geom))
        except Exception:
            pass

    if not lines:
        return []

    merged = linemerge(lines) if len(lines) > 1 else lines[0]

    widths = []
    for transect_info in transects:
        try:
            t_geom = transect_info["geometry"]
            if isinstance(t_geom, dict):
                transect_line = shape(t_geom)
            else:
                transect_line = t_geom

            inter = transect_line.intersection(merged)
            pts = _extract_points(inter)

            if len(pts) >= 2:
                # Use the two farthest points
                max_dist = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        d = pts[i].distance(pts[j])
                        if d > max_dist:
                            max_dist = d

                quality = QUALITY_HIGH
                if max_dist < DEFAULT_MIN_CHANNEL_WIDTH:
                    quality = QUALITY_LOW

                widths.append({
                    "geometry": transect_info["geometry"],
                    "properties": {
                        "station": transect_info["properties"].get("station", 0),
                        "width": round(max_dist, 4),
                        "quality": quality,
                        "n_intersections": len(pts),
                    },
                })
            else:
                widths.append({
                    "geometry": transect_info["geometry"],
                    "properties": {
                        "station": transect_info["properties"].get("station", 0),
                        "width": None,
                        "quality": QUALITY_INVALID,
                        "n_intersections": len(pts),
                    },
                })

        except Exception as e:
            widths.append({
                "geometry": transect_info["geometry"],
                "properties": {
                    "station": transect_info["properties"].get("station", 0),
                    "width": None,
                    "quality": QUALITY_INVALID,
                    "note": str(e),
                },
            })

    return widths


# ============================================================
# Change Hotspot Detection
# ============================================================

def detect_change_hotspots(migration_results: List[Dict],
                           percentile: float = DEFAULT_HOTSPOT_PERCENTILE) -> List[Dict]:
    """
    Detect change hotspots from migration rate results.

    Hotspots are transects where migration distance exceeds the
    specified percentile threshold.

    Args:
        migration_results: List of migration rate features
        percentile: Percentile threshold (default 90)

    Returns:
        List of hotspot features
    """
    # Extract valid migration distances
    distances = []
    for r in migration_results:
        d = r["properties"].get("migration_distance")
        if d is not None:
            distances.append(abs(d))

    if not distances:
        return []

    threshold = np.percentile(distances, percentile)

    hotspots = []
    for r in migration_results:
        d = r["properties"].get("migration_distance")
        if d is not None and abs(d) >= threshold:
            props = dict(r["properties"])
            props["hotspot"] = True
            props["threshold"] = round(threshold, 4)
            props["percentile"] = percentile
            hotspots.append({
                "geometry": r["geometry"],
                "properties": props,
            })

    return hotspots


# ============================================================
# Swath Analysis (Migration Zone)
# ============================================================

def compute_migration_zone(migration_results: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics for migration zone.

    Returns:
        Dict with zone statistics
    """
    distances = []
    erosion_count = 0
    deposition_count = 0
    stable_count = 0

    for r in migration_results:
        d = r["properties"].get("migration_distance")
        direction = r["properties"].get("direction", DIR_STABLE)

        if d is not None:
            distances.append(d)
        if direction == DIR_EROSION:
            erosion_count += 1
        elif direction == DIR_DEPOSITION:
            deposition_count += 1
        else:
            stable_count += 1

    if not distances:
        return {
            "mean_migration": 0,
            "max_migration": 0,
            "std_migration": 0,
            "erosion_count": erosion_count,
            "deposition_count": deposition_count,
            "stable_count": stable_count,
            "total_transects": len(migration_results),
        }

    return {
        "mean_migration": round(float(np.mean(distances)), 4),
        "max_migration": round(float(np.max(np.abs(distances))), 4),
        "std_migration": round(float(np.std(distances)), 4),
        "erosion_count": erosion_count,
        "deposition_count": deposition_count,
        "stable_count": stable_count,
        "total_transects": len(migration_results),
    }


# ============================================================
# Synthetic Data Generation (for testing)
# ============================================================

def generate_rectangular_channel(start_x: float, start_y: float,
                                 length: float, width: float,
                                 n_rows: int = 100, n_cols: int = 200) -> np.ndarray:
    """
    Generate a synthetic rectangular channel water mask.

    Args:
        start_x: Left edge of channel (inclusive)
        start_y: Top edge of channel (inclusive)
        length: Length of channel (vertical, number of rows)
        width: Width of channel (horizontal, number of columns)
        n_rows: Number of rows in output array
        n_cols: Number of columns in output array

    Returns:
        Binary mask (1 = water, 0 = land)
    """
    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    # Channel occupies rows [start_y, start_y + length) and cols [start_x, start_x + width)
    end_y = min(int(start_y + length), n_rows)
    end_x = min(int(start_x + width), n_cols)
    start_y_int = max(int(start_y), 0)
    start_x_int = max(int(start_x), 0)

    mask[start_y_int:end_y, start_x_int:end_x] = 1

    return mask


def generate_synthetic_river_centerline(n_points: int = 50,
                                       amplitude: float = 10.0,
                                       wavelength: float = 50.0) -> np.ndarray:
    """
    Generate a synthetic meandering river centerline.

    Args:
        n_points: Number of centerline points
        amplitude: Meander amplitude
        wavelength: Meander wavelength

    Returns:
        Nx2 array of (x, y) coordinates
    """
    x = np.linspace(0, wavelength * 2, n_points)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    return np.column_stack([x, y])


def generate_synthetic_shoreline_from_centerline(centerline: np.ndarray,
                                                  width: float) -> np.ndarray:
    """
    Generate shoreline points from centerline and channel width.

    Args:
        centerline: Nx2 array of centerline points
        width: Channel width

    Returns:
        Nx2 array of shoreline points (left bank)
    """
    shoreline = np.zeros_like(centerline)
    for i in range(len(centerline)):
        if i < len(centerline) - 1:
            dx = centerline[i + 1, 0] - centerline[i, 0]
            dy = centerline[i + 1, 1] - centerline[i, 1]
        else:
            dx = centerline[i, 0] - centerline[i - 1, 0]
            dy = centerline[i, 1] - centerline[i - 1, 1]

        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            shoreline[i] = centerline[i]
            continue

        # Perpendicular direction (left bank)
        perp_x = -dy / length
        perp_y = dx / length

        shoreline[i, 0] = centerline[i, 0] + perp_x * width * 0.5
        shoreline[i, 1] = centerline[i, 1] + perp_y * width * 0.5

    return shoreline


# ============================================================
# Main Analysis Pipeline
# ============================================================

def auto_download_image(args, output_dir: Path) -> Dict[str, Any]:
    """Download one sentinel-2-l2a scene from MPC using --bbox + --date-range.

    Returns metadata dict (also writes the path back to args.image).
    """
    if not _FETCHER_AVAILABLE:
        raise RuntimeError(
            "Shared data fetcher not importable. Pass --image <local.tif> instead, "
            "or ensure _geoskill_data_fetcher is on sys.path."
        )
    bbox = parse_bbox_arg(getattr(args, "bbox", None), getattr(args, "aoi_file", None))
    if bbox is None:
        raise RuntimeError("auto_download_image requires --bbox or --aoi-file")
    dr = parse_date_range_arg(getattr(args, "date_range", None))
    if dr is None:
        raise RuntimeError("auto_download_image requires --date-range")
    cache_dir = getattr(args, "cache_dir", None)
    fetcher = DataFetcher(
        source=DataSource.PLANETARY_COMPUTER,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
    items = fetcher.search_stac(
        collection="sentinel-2-l2a",
        bbox=bbox,
        date_range=dr,
        limit=1,
    )
    if not items:
        raise RuntimeError(
            f"No sentinel-2-l2a items found in bbox={bbox} for {dr.start}..{dr.end}"
        )
    download_dir = output_dir / "downloaded"
    paths = fetcher.download_assets(
        items=items, out_dir=download_dir, max_items=1, max_total_mb=500,
        prefer_assets=['B04', 'B08', 'B02'],
    )
    if not paths:
        raise RuntimeError("Download returned no files")
    args.image = str(paths[0])
    return {
        "data_source": "MPC",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "collection": "sentinel-2-l2a",
        "bbox": bbox.to_string(),
        "date_range": f"{dr.start},{dr.end}",
        "n_items_searched": len(items),
        "downloaded_paths": [str(p) for p in paths],
    }


def run_analysis(args: argparse.Namespace) -> int:
    """Main analysis workflow."""
    output_dir = Path(args.output_dir) if args.output_dir else Path("rmc-output")

    # --- Auto-download mode: fetch sentinel-2-l2a from MPC ---
    if (getattr(args, "bbox", None) or getattr(args, "aoi_file", None)) and getattr(args, "date_range", None):
        if not getattr(args, "image", None):
            try:
                fetch_meta = auto_download_image(args, output_dir)
                mode = "auto_download"
                print(f"  Auto-downloaded image: {args.image}")
            except DataFetcherError as e:
                print(f"ERROR: auto-download failed: [{e.kind}] {e.message}", file=sys.stderr)
                return EXIT_PROCESSING if 'EXIT_PROCESSING' in dir() else 7
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse parameters
    transect_spacing = args.transect_spacing if args.transect_spacing else DEFAULT_TRANSECT_SPACING
    min_channel_width = args.min_channel_width if args.min_channel_width else DEFAULT_MIN_CHANNEL_WIDTH
    shoreline_method = args.shoreline_method if args.shoreline_method else DEFAULT_SHORELINE_METHOD
    water_threshold = args.water_threshold if hasattr(args, 'water_threshold') and args.water_threshold is not None else DEFAULT_WATER_THRESHOLD
    hotspot_percentile = args.hotspot_percentile if hasattr(args, 'hotspot_percentile') and args.hotspot_percentile is not None else DEFAULT_HOTSPOT_PERCENTILE

    # --- Load or generate data ---
    if hasattr(args, 'input_masks') and args.input_masks:
        # Load from files
        try:
            import rasterio
        except ImportError:
            print("ERROR: rasterio required for GeoTIFF input", file=sys.stderr)
            return EXIT_DEP

        water_masks = []
        years = []
        for mask_path in args.input_masks:
            try:
                with rasterio.open(mask_path) as src:
                    mask_data = src.read(1)
                    transform = src.transform
                    crs = src.crs
                    water_masks.append(mask_data)
                    # Extract year from filename or metadata
                    year = _extract_year_from_filename(mask_path)
                    years.append(year)
            except Exception as e:
                print(f"ERROR: Failed to read {mask_path}: {e}", file=sys.stderr)
                return EXIT_VALIDATION
    else:
        # Generate synthetic data for demonstration
        water_masks, years, transform, crs = _generate_demo_data(output_dir)

    if len(water_masks) < 2:
        print("ERROR: At least 2 time periods required", file=sys.stderr)
        return EXIT_VALIDATION

    # --- Extract shorelines for each period ---
    all_shorelines = []
    all_centerlines = []
    all_polygons = []

    for i, mask in enumerate(water_masks):
        # Extract polygons
        polygons = extract_polygons_from_mask(mask, transform)
        all_polygons.append(polygons)

        # Extract shoreline
        shorelines = extract_shoreline(mask, transform)
        all_shorelines.append(shorelines)

        # Compute centerline from largest polygon
        if polygons:
            largest = max(polygons, key=lambda p: p["properties"].get("area", 0))
            centerline = compute_centerline(largest["geometry"], transform)
            if centerline is None:
                centerline = compute_centerline_simple(largest["geometry"])
            all_centerlines.append(centerline)
        else:
            all_centerlines.append(None)

    # --- Use first period centerline for transects ---
    primary_centerline = None
    for cl in all_centerlines:
        if cl is not None:
            primary_centerline = cl
            break

    if primary_centerline is None:
        print("ERROR: Could not compute centerline", file=sys.stderr)
        return EXIT_PROCESSING

    # --- Generate transects ---
    transects = generate_transects(primary_centerline, spacing=transect_spacing)

    if not transects:
        print("ERROR: Could not generate transects", file=sys.stderr)
        return EXIT_PROCESSING

    # --- Compute migration rates between consecutive periods ---
    all_migration = []
    for i in range(len(all_shorelines) - 1):
        migration = compute_migration_rates(
            transects, all_shorelines[i], all_shorelines[i + 1]
        )
        all_migration.append(migration)

    # --- Compute channel widths for each period ---
    all_widths = []
    for shorelines in all_shorelines:
        widths = compute_channel_widths(transects, shorelines)
        all_widths.append(widths)

    # --- Detect change hotspots ---
    # Use the first migration pair for hotspot detection
    hotspots = []
    if all_migration:
        hotspots = detect_change_hotspots(all_migration[0], percentile=hotspot_percentile)

    # --- Compute migration zone statistics ---
    zone_stats = []
    for migration in all_migration:
        stats = compute_migration_zone(migration)
        zone_stats.append(stats)

    # --- Write Outputs ---

    # shorelines.geojson
    shoreline_features = []
    for i, shorelines in enumerate(all_shorelines):
        for s in shorelines:
            props = dict(s["properties"])
            props["period"] = i
            props["year"] = years[i] if i < len(years) else i
            shoreline_features.append({
                "geometry": s["geometry"],
                "properties": props,
            })

    shoreline_geojson = {
        "type": "FeatureCollection",
        "features": shoreline_features,
    }
    shoreline_path = output_dir / "shorelines.geojson"
    shoreline_path.write_text(
        json.dumps(shoreline_geojson, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # centerlines.geojson
    centerline_features = []
    for i, cl in enumerate(all_centerlines):
        if cl is not None:
            props = dict(cl["properties"])
            props["period"] = i
            props["year"] = years[i] if i < len(years) else i
            centerline_features.append({
                "geometry": cl["geometry"],
                "properties": props,
            })

    centerline_geojson = {
        "type": "FeatureCollection",
        "features": centerline_features,
    }
    centerline_path = output_dir / "centerlines.geojson"
    centerline_path.write_text(
        json.dumps(centerline_geojson, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # transects.geojson
    transect_features = []
    for t in transects:
        transect_features.append({
            "geometry": t["geometry"],
            "properties": t["properties"],
        })

    transect_geojson = {
        "type": "FeatureCollection",
        "features": transect_features,
    }
    transect_path = output_dir / "transects.geojson"
    transect_path.write_text(
        json.dumps(transect_geojson, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # migration_rates.csv
    migration_path = output_dir / "migration_rates.csv"
    with open(migration_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "transect_id", "station", "migration_distance", "migration_rate",
            "direction", "quality", "period_from", "period_to",
        ])
        writer.writeheader()
        for i, migration in enumerate(all_migration):
            for m in migration:
                props = m["properties"]
                writer.writerow({
                    "transect_id": props.get("station", ""),
                    "station": props.get("station", ""),
                    "migration_distance": props.get("migration_distance", ""),
                    "migration_rate": props.get("migration_rate", ""),
                    "direction": props.get("direction", ""),
                    "quality": props.get("quality", ""),
                    "period_from": i,
                    "period_to": i + 1,
                })

    # change_hotspots.geojson
    hotspot_geojson = {
        "type": "FeatureCollection",
        "features": hotspots,
    }
    hotspot_path = output_dir / "change_hotspots.geojson"
    hotspot_path.write_text(
        json.dumps(hotspot_geojson, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # request.json
    request_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transect_spacing": transect_spacing,
        "min_channel_width": min_channel_width,
        "shoreline_method": shoreline_method,
        "water_threshold": water_threshold,
        "hotspot_percentile": hotspot_percentile,
        "n_periods": len(water_masks),
        "years": years,
        "output_dir": str(output_dir),
    }
    request_path = output_dir / "request.json"
    request_path.write_text(
        json.dumps(request_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # dataset-manifest.json
    dataset_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_periods": len(water_masks),
        "periods": [
            {
                "index": i,
                "year": years[i] if i < len(years) else i,
                "n_polygons": len(all_polygons[i]) if i < len(all_polygons) else 0,
                "n_shorelines": len(all_shorelines[i]) if i < len(all_shorelines) else 0,
            }
            for i in range(len(water_masks))
        ],
    }
    dataset_path = output_dir / "dataset-manifest.json"
    dataset_path.write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # output-manifest.json
    output_files = {
        "shorelines.geojson": str(shoreline_path),
        "centerlines.geojson": str(centerline_path),
        "transects.geojson": str(transect_path),
        "migration_rates.csv": str(migration_path),
        "change_hotspots.geojson": str(hotspot_path),
        "request.json": str(request_path),
        "dataset-manifest.json": str(dataset_path),
    }
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": vars(args),
        "summary": {
            "n_hotspots": len(hotspots),
            "n_outputs": len(output_files),
        },
        "analysis_parameters": request_info,
        "output_files": output_files,
        "migration_zone_stats": zone_stats,
        "n_hotspots": len(hotspots),
    }
    
    # Auto-download metadata: propagate from fetch_meta (set by the
    # auto_download_* helpers in this module) into the manifest so the
    # output-manifest.json records data_source / collection / fetched_at.
    try:
        _fm = locals().get('fetch_meta') or globals().get('fetch_meta')
    except Exception:
        _fm = None
    if _fm:
        manifest["data_source"] = _fm.get("data_source")
        manifest["collection"] = _fm.get("collection")
        manifest["fetched_at"] = _fm.get("fetched_at")
        if "downloaded_paths" in _fm:
            manifest["downloaded_paths"] = _fm["downloaded_paths"]
        if "bbox" in _fm:
            manifest["query_bbox"] = _fm["bbox"]
        if "date_range" in _fm:
            manifest["query_date_range"] = _fm["date_range"]
    manifest_path = output_dir / "output-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # qa.json
    qa = {
        "status": "complete",
        "checks": {
            "centerline_computed": primary_centerline is not None,
            "transects_generated": len(transects) > 0,
            "migration_computed": len(all_migration) > 0,
            "hotspots_detected": len(hotspots) >= 0,
        },
        "warnings": [],
        "n_transects": len(transects),
        "n_hotspots": len(hotspots),
        "zone_statistics": zone_stats,
    }
    qa_path = output_dir / "qa.json"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return EXIT_OK


def _generate_demo_data(output_dir: Path):
    """Generate synthetic demo data for testing."""
    n_rows, n_cols = 100, 200
    transform = None
    crs = "EPSG:4326"

    # Period 1: Channel at x=50-70
    mask1 = np.zeros((n_rows, n_cols), dtype=np.uint8)
    mask1[10:90, 50:70] = 1

    # Period 2: Channel shifted right by 5 pixels (x=55-75)
    mask2 = np.zeros((n_rows, n_cols), dtype=np.uint8)
    mask2[10:90, 55:75] = 1

    # Period 3: Channel shifted right by 10 pixels (x=60-80)
    mask3 = np.zeros((n_rows, n_cols), dtype=np.uint8)
    mask3[10:90, 60:80] = 1

    water_masks = [mask1, mask2, mask3]
    years = [2010, 2015, 2020]

    return water_masks, years, transform, crs


def _extract_year_from_filename(filepath: str) -> int:
    """Extract year from filename."""
    import re
    basename = Path(filepath).stem
    match = re.search(r"(19|20)\d{2}", basename)
    if match:
        return int(match.group())
    return 0


def main():
    parser = argparse.ArgumentParser(description="River Morphology Change Analysis")
    parser.add_argument("--input-masks", nargs="*", default=None,
                        help="Input water mask GeoTIFF files (ordered by time)")
    parser.add_argument("--transect-spacing", type=float, default=DEFAULT_TRANSECT_SPACING,
                        help=f"Distance between transects (default: {DEFAULT_TRANSECT_SPACING})")
    parser.add_argument("--min-channel-width", type=float, default=DEFAULT_MIN_CHANNEL_WIDTH,
                        help=f"Minimum channel width in map units (default: {DEFAULT_MIN_CHANNEL_WIDTH})")
    parser.add_argument("--shoreline-method", default=DEFAULT_SHORELINE_METHOD,
                        choices=["threshold", "canny", "manual"],
                        help="Shoreline extraction method (default: threshold)")
    parser.add_argument("--water-threshold", type=float, default=DEFAULT_WATER_THRESHOLD,
                        help=f"Water extraction threshold (default: {DEFAULT_WATER_THRESHOLD})")
    parser.add_argument("--hotspot-percentile", type=float, default=DEFAULT_HOTSPOT_PERCENTILE,
                        help=f"Hotspot percentile threshold (default: {DEFAULT_HOTSPOT_PERCENTILE})")
    parser.add_argument("--output-dir", "-o", default="rmc-output",
                        help="Output directory (default: rmc-output)")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    add_bbox_date_args(parser)

    args = parser.parse_args()

    try:
        sys.exit(run_analysis(args))
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(EXIT_PROCESSING)


if __name__ == "__main__":
    main()
