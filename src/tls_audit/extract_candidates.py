#!/usr/bin/env python3
"""
Engineering reconstruction: Extract HookNet-TLS XML polygons into YOLO patches.
Updated for robust coordinate validation, multi-context extraction, and detailed metadata.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
import hashlib
from pathlib import Path

def get_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def parse_asap_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = root.find('Annotations')
    if annotations is None:
        return []

    polygons = []
    for ann in annotations.findall('Annotation'):
        name = ann.get('Name', '')
        group = ann.get('PartOfGroup', '')

        coords = []
        for coord in ann.find('Coordinates').findall('Coordinate'):
            x = float(coord.get('X'))
            y = float(coord.get('Y'))
            coords.append((x, y))

        if len(coords) > 0:
            min_x = min([c[0] for c in coords])
            max_x = max([c[0] for c in coords])
            min_y = min([c[1] for c in coords])
            max_y = max([c[1] for c in coords])

            polygons.append({
                'name': name,
                'group': group,
                'coords': coords,
                'bbox': (min_x, min_y, max_x, max_y)
            })

    return polygons

def extract_metadata(slide):
    props = slide.properties
    meta = {
        "dimensions": slide.dimensions,
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": slide.level_downsamples,
        "mpp_x": props.get('openslide.mpp-x'),
        "mpp_y": props.get('openslide.mpp-y'),
        "objective_power": props.get('openslide.objective-power'),
        "vendor": props.get('openslide.vendor')
    }
    return meta

def extract_patches(wsi_path, xml_path, out_dir, paddings):
    try:
        import openslide
        from PIL import Image, ImageDraw
    except ImportError:
        print("Error: openslide-python and Pillow required")
        sys.exit(1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(wsi_path)
    wsi_meta = extract_metadata(slide)

    with open(out_dir / "wsi_metadata.json", "w") as f:
        json.dump(wsi_meta, f, indent=2)

    polygons = parse_asap_xml(xml_path)

    # Validation: Do polygons exceed slide dims?
    slide_w, slide_h = slide.dimensions
    valid_polys = []
    max_error = 0.0
    for poly in polygons:
        if poly['group'].lower() != 'tls':
            continue
        min_x, min_y, max_x, max_y = poly['bbox']

        # XML coordinates are presumed to be at level-0 of the provided WSI
        if min_x < -1 or min_y < -1 or max_x > slide_w + 1 or max_y > slide_h + 1:
            print(f"Error: Polygon {poly['name']} bbox {poly['bbox']} exceeds slide dimensions {slide.dimensions}.")
            print("Cannot prove XML and WSI are in the same coordinate space. Aborting.")
            sys.exit(1)

        valid_polys.append(poly)

    print(f"Found {len(valid_polys)} TLS polygons.")

    if len(valid_polys) == 0:
        print("No TLS objects found.")
        for padding in paddings:
            (out_dir / f"extracted_patches_{int(padding*100)}").mkdir(exist_ok=True)
        with open(out_dir / "crop_metadata.json", "w") as f:
            json.dump([], f, indent=2)
        slide.close()
        return

    try:
        thumb = slide.get_thumbnail((2048, 2048))
        thumb_w, thumb_h = thumb.size
        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h
        draw = ImageDraw.Draw(thumb)
    except Exception as e:
        print(f"Warning: Could not create thumbnail: {e}")
        thumb = None

    metadata = []

    for padding in paddings:
        pad_dir = out_dir / f"extracted_patches_{int(padding*100)}"
        pad_dir.mkdir(exist_ok=True)

        for i, poly in enumerate(valid_polys):
            min_x, min_y, max_x, max_y = poly['bbox']
            w = max_x - min_x
            h = max_y - min_y

            pad_x = w * padding
            pad_y = h * padding

            crop_x = max(0, int(min_x - pad_x))
            crop_y = max(0, int(min_y - pad_y))
            crop_w = int(w + 2*pad_x)
            crop_h = int(h + 2*pad_y)

            # Make square
            if crop_w > crop_h:
                diff = crop_w - crop_h
                crop_y = max(0, int(crop_y - diff/2))
                crop_h = crop_w
            elif crop_h > crop_w:
                diff = crop_h - crop_w
                crop_x = max(0, int(crop_x - diff/2))
                crop_w = crop_h

            # Clamp to slide boundaries
            if crop_x + crop_w > slide_w:
                crop_w = slide_w - crop_x
            if crop_y + crop_h > slide_h:
                crop_h = slide_h - crop_y

            crop_img = slide.read_region((crop_x, crop_y), 0, (crop_w, crop_h)).convert('RGB')
            out_name = f"patch_{i}.jpeg"
            out_path = pad_dir / out_name
            crop_img.save(out_path, format='JPEG', quality=95)

            mpp_x = float(wsi_meta['mpp_x']) if wsi_meta.get('mpp_x') else 0.5
            mpp_y = float(wsi_meta['mpp_y']) if wsi_meta.get('mpp_y') else 0.5
            meta = {
                'patch_id': i,
                'patch_file': out_name,
                'padding_ratio': padding,
                'polygon_coordinates': poly['coords'],
                'level0_bbox': [min_x, min_y, max_x, max_y],
                'physical_bbox_um': [min_x*mpp_x, min_y*mpp_y, max_x*mpp_x, max_y*mpp_y],
                'crop_origin': (crop_x, crop_y),
                'crop_size': (crop_w, crop_h),
                'clamped_to_boundary': (crop_x + crop_w == slide_w) or (crop_y + crop_h == slide_h),
                'mpp_x': mpp_x,
                'mpp_y': mpp_y,
                'sha256': get_hash(out_path)
            }
            metadata.append(meta)

            if thumb and padding == paddings[0]: # Draw only once
                draw.rectangle([
                    crop_x * scale_x, crop_y * scale_y,
                    (crop_x + crop_w) * scale_x, (crop_y + crop_h) * scale_y
                ], outline="red", width=2)

    if thumb:
        thumb.save(out_dir / "overlay_bboxes.png")

    with open(out_dir / "crop_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    slide.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paddings", nargs="+", type=float, default=[0.1, 0.25])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if out_dir.exists() and list(out_dir.glob("*.json")) and not args.overwrite:
        print("Error: Output dir already exists. Use --overwrite.")
        sys.exit(1)

    extract_patches(args.wsi, args.xml, out_dir, args.paddings)
    print("Extraction successful.")

if __name__ == "__main__":
    main()
