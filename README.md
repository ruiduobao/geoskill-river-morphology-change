# geoskill-river-morphology-change

> Extract shorelines, centerlines and channel widths from multi-temporal water body masks. Quantify shoreline migration, channel migration, change hotspots and migration zones. Use when analyzing river channel changes, identifying erosion/deposition areas, or generating river morphology reports.

See [SKILL.md](SKILL.md) for full usage, inputs and outputs.

## Quick start

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

## License

MIT
