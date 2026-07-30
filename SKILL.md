---
name: river-morphology-change
description: >
  Extract shorelines, centerlines and channel widths from multi-temporal
  water body masks. Quantify shoreline migration, channel migration,
  change hotspots and migration zones.
  Use when analyzing river channel changes, identifying erosion/deposition
  areas, or generating river morphology reports.
---

# River Morphology Change

Extracts shorelines, centerlines, and channel widths from multi-temporal
water body masks. Quantifies shoreline migration, channel migration,
and change hotspots.

## Trigger

Use when the user wants to:
- Analyze river channel migration from multi-temporal imagery
- Identify erosion and deposition areas along river banks
- Compute channel width changes over time
- Detect change hotspots where migration is most severe
- Generate river morphology change reports
- Compare shorelines between different time periods

## CLI Usage

```bash
# Basic analysis with default parameters
python scripts/river_morphology_change.py --output-dir ./rmc-output

# Custom transect spacing and hotspot percentile
python scripts/river_morphology_change.py \
  --transect-spacing 30.0 \
  --hotspot-percentile 85.0 \
  --output-dir ./rmc-output

# With custom water threshold
python scripts/river_morphology_change.py \
  --water-threshold 0.1 \
  --min-channel-width 15.0 \
  --output-dir ./rmc-output
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--input-masks` | None | Input water mask GeoTIFF files (ordered by time) |
| `--transect-spacing` | 50.0 | Distance between transects in map units |
| `--min-channel-width` | 20.0 | Minimum channel width for quality flagging |
| `--shoreline-method` | threshold | Shoreline extraction method: threshold, canny, manual |
| `--water-threshold` | 0.0 | Water extraction threshold (NDWI/MNDWI) |
| `--hotspot-percentile` | 90.0 | Percentile threshold for hotspot detection |
| `--output-dir` | ./rmc-output | Output directory |

## Output

| File | Description |
|---|---|
| `shorelines.geojson` | Shoreline features for each time period |
| `centerlines.geojson` | Centerline features for each time period |
| `transects.geojson` | Perpendicular transect lines along centerline |
| `migration_rates.csv` | Migration distances and rates per transect |
| `change_hotspots.geojson` | Hotspot features where migration exceeds threshold |
| `request.json` | Analysis request metadata |
| `dataset-manifest.json` | Dataset inventory and period information |
| `output-manifest.json` | Output file inventory and statistics |
| `qa.json` | Quality assurance checks |

## Quality Codes

| Code | Name | Description |
|---|---|---|
| 1 | high | Clean intersection, single shoreline crossing |
| 2 | medium | Small migration or minor complexity |
| 3 | low | Multiple intersections, complex geometry |
| 4 | invalid | No intersection or processing error |

## Direction Codes

| Code | Name | Description |
|---|---|---|
| -1 | erosion | Shoreline retreats (water expands) |
| 0 | stable | No significant change |
| 1 | deposition | Shoreline advances (water contracts) |

## Key Algorithms

### Water Body Extraction
Uses NDWI (Normalized Difference Water Index) or MNDWI (Modified NDWI)
to extract water bodies from multispectral imagery:
- NDWI = (Green - NIR) / (Green + NIR)
- MNDWI = (Green - SWIR1) / (Green + SWIR1)

### Centerline Extraction
Uses Voronoi-based medial axis extraction:
1. Sample points along water polygon boundary
2. Compute Voronoi diagram of boundary points
3. Extract Voronoi vertices inside polygon as centerline
4. Fallback to bounding box centerline if scipy unavailable

### Transect Generation
Generates perpendicular transects along the centerline at regular intervals.
Each transect is perpendicular to the local tangent direction.

### Migration Rate Computation
Uses perpendicular intersection method:
1. For each transect, find intersection with both shorelines
2. Measure distance between intersection points
3. Determine direction (erosion/deposition) based on distance from center

### Change Hotspot Detection
Identifies transects where migration distance exceeds the specified
percentile threshold (default: 90th percentile).

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Argument error |
| 3 | Dependency missing |
| 6 | Data validation failure |
| 7 | Processing failure |

## Limitations

- Water level changes may be misinterpreted as morphological change
- Narrow channels near pixel resolution limit have reduced accuracy
- Centerline topology handling is complex for braided/anastomosing channels
- Results are auxiliary analysis only; engineering decisions require manual review
- Same-season imagery recommended to minimize water level effects

## References

- Pavelsky & Smith 2008, IEEE GRSL (RivWidth algorithm)
- Isikdogan et al. 2017, IGARSS (RivMap automatic river width extraction)
- Fisher et al. 2013, Nature Geoscience (global river delta assessment)


## 数据下载

本 skill 可自动从 Microsoft Planetary Computer 下载数据 (无需 API key):

```bash
python river_morphology_change.py --bbox 116,39,117,40 --date-range 2024-06-01,2024-06-30 --output-dir <tmp>
```

- `--bbox W,S,E,N`: WGS-84 边界框 (西, 南, 东, 北)
- `--date-range START,END`: 日期范围 (YYYY-MM-DD,YYYY-MM-DD)
- `--aoi-file <path.geojson>`: 替代 --bbox 的 GeoJSON 多边形
- `--cache-dir <path>`: 缓存目录 (默认 ~/.geoskill_cache)

当用户只给 `--bbox + --date-range` (没有 `--image`) 时，skill 自动下载数据。
当用户给 `--image` 时，走原文件路径 (向后兼容)。
