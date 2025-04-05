import os
import math
import gzip
import io
import sys

import mapbox_vector_tile
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
import matplotlib.pyplot as plt
from PIL import Image

MAX_DATA_ZOOM = 14  # We only have real tiles up to zoom=14

# ---------------------------------------------------------------------
#  Slippy-map tile <-> lat/lon helpers
# ---------------------------------------------------------------------

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile x/y indices (slippy map)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile

def num2deg(x, y, zoom):
    """Return (lat_deg, lon_deg) of the *upper-left* corner of tile (x,y) at zoom."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2*y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

# Returns bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon
def get_tile_bbox_latlon(x, y, zoom):
    lat1, lon1 = num2deg(x,   y,   zoom)   # top-left
    lat2, lon2 = num2deg(x+1, y+1, zoom)   # bottom-right

    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    return (min_lon, min_lat, max_lon, max_lat)

# ---------------------------------------------------------------------
#  Reading a tile (assuming .pbf.gz or .png compressed as gz)
# ---------------------------------------------------------------------

def get_tile_path(base_dir, zoom, x, y):
    """Construct file path. Adjust extension as needed (e.g. .pbf.gz, .png, .mvt, etc.)."""
    filename = f"{y}.png"  # Example. If you truly have vector tiles as .pbf.gz, change here.
    path = os.path.join(base_dir, str(zoom), str(x), filename)
    print(f"[DEBUG] get_tile_path -> z={zoom}, x={x}, y={y}")
    print(f"        => {path}")
    return path

def decompress_tile(tile_path):
    """Open a gz file and return raw bytes."""
    print(f"[DEBUG] decompress_tile -> Opening gz file: {tile_path}")
    with gzip.open(tile_path, 'rb') as f_in:
        data = f_in.read()
    print(f"[DEBUG] decompress_tile -> Read {len(data)} bytes.")
    return data

def decode_tile(tile_data):
    """Decode the PBF vector tile data into a Python dict of layers."""
    print(f"[DEBUG] decode_tile -> Attempting decode...")
    decoded = mapbox_vector_tile.decode(tile_data)
    layer_names = list(decoded.keys())
    print(f"[DEBUG] decode_tile -> Layers found: {layer_names}")
    return decoded

# ---------------------------------------------------------------------
#  Overzoom logic (clamp to z=14, clip to sub-tile if z>14)
# ---------------------------------------------------------------------

def load_overzoomed_tile(base_dir, requested_x, requested_y, requested_z, extent=4096):
    """
    Return (gdf, (min_lon, min_lat, max_lon, max_lat)) for the tile
    (requested_x, requested_y, requested_z). If requested_z>MAX_DATA_ZOOM,
    we overzoom from the parent tile at z=14 by clipping.

    gdf: GeoDataFrame in EPSG:4326 lat/lon
    (min_lon, min_lat, max_lon, max_lat): bounding box that exactly covers
    the geometry we end up with.
    """
    print(f"[DEBUG] load_overzoomed_tile -> req_x={requested_x}, req_y={requested_y}, req_z={requested_z}")
    parent_z = min(requested_z, MAX_DATA_ZOOM)
    print(f"[DEBUG] load_overzoomed_tile -> parent_z={parent_z}")

    if requested_z <= MAX_DATA_ZOOM:
        # Simple case: no overzoom, just load tile (requested_x, requested_y) at requested_z
        tile_path = get_tile_path(base_dir, parent_z, requested_x, requested_y)
        if not os.path.exists(tile_path):
            raise FileNotFoundError(f"No tile at z={parent_z}, x={requested_x}, y={requested_y} -> {tile_path}")
        data = decompress_tile(tile_path)
        decoded = decode_tile(data)
        gdf = decode_tile_to_latlon(decoded, requested_x, requested_y, requested_z, extent)
        print(f"[DEBUG] load_overzoomed_tile -> direct decode -> {len(gdf)} features.")
        # The bounding box is the full tile at (requested_x, requested_y, requested_z)
        min_lon, min_lat, max_lon, max_lat = get_tile_bbox_latlon(requested_x, requested_y, requested_z)
        return gdf, (min_lon, min_lat, max_lon, max_lat)

    # Overzoom path (requested_z > 14)
    factor = 2 ** (requested_z - parent_z)
    parent_x = requested_x // factor
    parent_y = requested_y // factor
    sub_x = requested_x % factor
    sub_y = requested_y % factor

    print(f"[DEBUG] load_overzoomed_tile -> Overzoom needed: factor={factor}, parent_x={parent_x}, parent_y={parent_y}, sub_x={sub_x}, sub_y={sub_y}")

    parent_path = get_tile_path(base_dir, parent_z, parent_x, parent_y)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"Parent tile missing at z={parent_z}, x={parent_x}, y={parent_y} -> {parent_path}")
    data = decompress_tile(parent_path)
    decoded_parent = decode_tile(data)

    # Convert everything to lat/lon
    parent_gdf = decode_tile_to_latlon(decoded_parent, parent_x, parent_y, parent_z, extent)
    print(f"[DEBUG] load_overzoomed_tile -> parent decode -> {len(parent_gdf)} features total.")

    if parent_gdf.empty:
        # Nothing to clip
        return parent_gdf, (0,0,0,0)

    # BBox for the entire parent tile in lat/lon
    p_min_lon, p_min_lat, p_max_lon, p_max_lat = get_tile_bbox_latlon(parent_x, parent_y, parent_z)

    # Subdivide that bounding box for the sub-tile
    lon_width = (p_max_lon - p_min_lon) / factor
    lat_height = (p_max_lat - p_min_lat) / factor

    sub_min_lon = p_min_lon + sub_x * lon_width
    sub_max_lon = p_min_lon + (sub_x+1) * lon_width
    sub_min_lat = p_min_lat + sub_y * lat_height
    sub_max_lat = p_min_lat + (sub_y+1) * lat_height

    print("[DEBUG] Overzoom sub-bbox in lat/lon:")
    print(f"         sub_min_lon={sub_min_lon}, sub_min_lat={sub_min_lat}")
    print(f"         sub_max_lon={sub_max_lon}, sub_max_lat={sub_max_lat}")

    clip_box_geom = box(sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat)

    clipped_rows = []
    for i, row in parent_gdf.iterrows():
        geom = row["geometry"]
        inter = geom.intersection(clip_box_geom)
        if not inter.is_empty:
            new_row = row.to_dict()
            new_row["geometry"] = inter
            clipped_rows.append(new_row)

    clipped_gdf = gpd.GeoDataFrame(clipped_rows, crs="EPSG:4326")
    print(f"[DEBUG] load_overzoomed_tile -> clipped -> {len(clipped_gdf)} features remain.")

    # Our bounding box is the sub-box we actually used
    return clipped_gdf, (sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat)

# ---------------------------------------------------------------------
#  Decoding from tile coords -> lat/lon
# ---------------------------------------------------------------------

def tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent=4096):
    """
    Convert [0..extent]^2 tile coords -> lat/lon by flipping y so y_local=0=bottom.
    """
    top_lat, left_lon  = num2deg(tile_x,   tile_y,   tile_z)
    bot_lat, right_lon = num2deg(tile_x+1, tile_y+1, tile_z)
    lat = bot_lat + (top_lat - bot_lat) * (y_local / extent)
    lon = left_lon + (right_lon - left_lon) * (x_local / extent)
    return lat, lon

def decode_tile_to_latlon(decoded_tile, tile_x, tile_y, tile_z, extent=4096):
    """
    Convert all geometry in 'decoded_tile' from [0..extent] tile coords to lat/lon.
    Return GeoDataFrame in EPSG:4326.
    """
    feats_list = []
    for layer_name, layer_dict in decoded_tile.items():
        fcount = len(layer_dict.get("features", []))
        print(f"[DEBUG] decode_tile_to_latlon -> layer '{layer_name}' has {fcount} features")
        for feat in layer_dict.get("features", []):
            geom_json = feat.get("geometry")
            if not geom_json:
                continue
            try:
                latlon_geom = transform_tilegeom_to_latlon(geom_json, tile_x, tile_y, tile_z, extent)
                if latlon_geom:
                    shapely_geom = shape(latlon_geom)
                    row = {"geometry": shapely_geom, "layer": layer_name}
                    row.update(feat.get("properties", {}))
                    feats_list.append(row)
            except Exception as ex:
                print(f"[DEBUG] transform error in layer {layer_name}: {ex}")
    print(f"[DEBUG] decode_tile_to_latlon -> total {len(feats_list)} features after transform.")
    return gpd.GeoDataFrame(feats_list, crs="EPSG:4326")

def transform_tilegeom_to_latlon(geom_json, tile_x, tile_y, tile_z, extent=4096):
    """Recursively transform a GeoJSON-like geometry from tile coords -> lat/lon."""
    g_type = geom_json["type"]
    coords = geom_json["coordinates"]
    if g_type == "Point":
        x_local, y_local = coords
        lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
        return {"type": "Point", "coordinates": (lon, lat)}

    elif g_type == "MultiPoint":
        new_pts = []
        for (x_local, y_local) in coords:
            lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
            new_pts.append((lon, lat))
        return {"type": "MultiPoint", "coordinates": new_pts}

    elif g_type == "LineString":
        line = []
        for (x_local, y_local) in coords:
            lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
            line.append((lon, lat))
        return {"type": "LineString", "coordinates": line}

    elif g_type == "MultiLineString":
        all_lines = []
        for line_coords in coords:
            line = []
            for (x_local, y_local) in line_coords:
                lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
                line.append((lon, lat))
            all_lines.append(line)
        return {"type": "MultiLineString", "coordinates": all_lines}

    elif g_type == "Polygon":
        new_rings = []
        for ring_coords in coords:
            ring = []
            for (x_local, y_local) in ring_coords:
                lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
                ring.append((lon, lat))
            new_rings.append(ring)
        return {"type": "Polygon", "coordinates": new_rings}

    elif g_type == "MultiPolygon":
        new_polygons = []
        for poly in coords:
            poly_rings = []
            for ring_coords in poly:
                ring = []
                for (x_local, y_local) in ring_coords:
                    lat, lon = tile_extent_to_latlon(tile_x, tile_y, tile_z, x_local, y_local, extent)
                    ring.append((lon, lat))
                poly_rings.append(ring)
            new_polygons.append(poly_rings)
        return {"type": "MultiPolygon", "coordinates": new_polygons}
    
    return None  # skip unrecognized geometry types

# ---------------------------------------------------------------------
#  Final: render_overzoomed_tile(...) and example usage
# ---------------------------------------------------------------------

def render_overzoomed_tile(base_dir, lat_center, lon_center, zoom,
                           fig_size=(10,10), dpi=200):
    """
    1) Convert lat/lon -> tile_x/tile_y at 'zoom' (which may be >14).
    2) load_overzoomed_tile(...) returns (gdf, bbox).
    3) Plot that geometry with bounding box from the actual clipped region.
    4) Return a PIL Image.
    """
    print(f"[DEBUG] render_overzoomed_tile -> center=({lat_center}, {lon_center}), zoom={zoom}")
    tile_x, tile_y = deg2num(lat_center, lon_center, zoom)
    print(f"[DEBUG] render_overzoomed_tile -> tile=({tile_x},{tile_y})")

    try:
        gdf, (box_min_lon, box_min_lat, box_max_lon, box_max_lat) = load_overzoomed_tile(
            base_dir, tile_x, tile_y, zoom, extent=4096
        )
    except FileNotFoundError as e:
        print(f"[DEBUG] render_overzoomed_tile -> FileNotFoundError: {e}")
        return None

    if gdf.empty:
        print("[DEBUG] render_overzoomed_tile -> GDF empty, no geometry.")
        return None

    print("[DEBUG] final bounding box for the clipped/overzoomed data:")
    print(f"         min_lon={box_min_lon}, min_lat={box_min_lat}")
    print(f"         max_lon={box_max_lon}, max_lat={box_max_lat}")

    # Optional layer exclusions / styling
    excluded_layers = {"water_name", "transportation_name", "poi", "place", "landuse"}
    layer_styles = {
        "water":           {"color": "#a6cee3", "edgecolor": "none", "alpha": 0.7},
        "waterway":        {"color": "#1f78b4", "linewidth": 0.8},
        "landcover":       {"color": "#b2df8a", "edgecolor": "none", "alpha": 0.7},
        "boundary":        {"color": "black",   "linestyle": "--", "linewidth": 0.8},
        "transportation":  {"color": "#fb9a99", "linewidth": 1},
        "building":        {"color": "#fdbf6f", "edgecolor": "black", "linewidth": 0.5, "alpha": 0.8},
    }

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_aspect("equal", "box")

    # Group by layer, skip excluded, apply style
    for layer_name, subdf in gdf.groupby("layer"):
        if layer_name in excluded_layers:
            print(f"[DEBUG]   skipping excluded layer '{layer_name}'")
            continue
        style = layer_styles.get(layer_name, {})
        print(f"[DEBUG]   plotting layer '{layer_name}' -> {len(subdf)} features, style={style}")
        subdf.plot(ax=ax, **style)

    # Now we set the final axis to the sub-tile bounding box
    ax.set_xlim(box_min_lon, box_max_lon)
    ax.set_ylim(box_min_lat, box_max_lat)

    ax.set_title(f"Overzoom: tile=({tile_x},{tile_y}) at z={zoom}, parent_z={MAX_DATA_ZOOM}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Convert figure to a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ------------------------------------------
# Example usage
# ------------------------------------------
if __name__ == "__main__":
    BASE_DIR = "/Users/jacobvaught/Downloads/map_folder"

    # We'll pretend to "zoom" to z=16 (overzoom).
    center_lat = 33.988880
    center_lon = -81.028963
    requested_zoom = 16

    print("[DEBUG] main -> Overzoom example, requesting z=16.")
    out_img = render_overzoomed_tile(
        base_dir=BASE_DIR,
        lat_center=center_lat,
        lon_center=center_lon,
        zoom=requested_zoom,
        fig_size=(12, 12),  # bigger figure => more resolution
        dpi=300
    )
    if out_img:
        print("[DEBUG] main -> Overzoomed tile rendered. Showing image...")
        out_img.show()
        # out_img.save("overzoom_z16_fixed.png")
    else:
        print("[DEBUG] main -> No image produced.")
