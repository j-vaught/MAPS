#generate_map.py
import os, math, gzip, json
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor, as_completed
import mapbox_vector_tile

# ------------------ Coordinate Transform ------------------ #
def deg2num(lat, lon, z):
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    return int((lon + 180) / 360 * n), int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)

def num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def get_tile_boundaries(x, y, z):
    top, left = num2deg(x, y, z)
    bot, right = num2deg(x + 1, y + 1, z)
    return top, bot, left, right

# ------------------ Tile and Geometry Utils ------------------ #
def tile_to_path(base, z, x, y):
    return os.path.join(base, str(z), str(x), f"{y}.png")

def decompress_tile(path):
    with gzip.open(path, "rb") as f:
        return f.read()

def decode_tile(data):
    return mapbox_vector_tile.decode(data)

def tile_coord_to_latlon(x, y, extent, bounds):
    top, bot, left, right = bounds
    return bot + (top - bot) * (y / extent), left + (right - left) * (x / extent)

def transform_geometry(tile_x, tile_y, zoom, geom, extent=4096):
    bounds = get_tile_boundaries(tile_x, tile_y, zoom)
    conv = lambda x, y: tile_coord_to_latlon(x, y, extent, bounds)[::-1]

    if geom["type"] == "Point":
        return {"type": "Point", "coordinates": conv(*geom["coordinates"])}
    if geom["type"] == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": [conv(*pt) for pt in geom["coordinates"]]}
    if geom["type"] == "LineString":
        return {"type": "LineString", "coordinates": [conv(*pt) for pt in geom["coordinates"]]}
    if geom["type"] == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [[conv(*pt) for pt in line] for line in geom["coordinates"]]}
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [[conv(*pt) for pt in ring] for ring in geom["coordinates"]]}
    if geom["type"] == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[[conv(*pt) for pt in ring] for ring in poly] for poly in geom["coordinates"]]}
    return None

# ------------------ Tile Processing ------------------ #
excluded_layers = {"poi", "water_name", "transportation_name", "place", "housenumber", "mountain_peak", "landuse"}

layer_styles = {
    "water":          {"color": "#a6cee3", "edgecolor": "none", "alpha": 0.7},
    "waterway":       {"color": "#1f78b4", "linewidth": 0.8},
    "landcover":      {"color": "#b2df8a", "edgecolor": "none", "alpha": 0.7},
    "boundary":       {"color": "black",   "linestyle": "--", "linewidth": 0.8},
    "transportation": {"color": "#fb9a99", "linewidth": 1},
    "building":       {"color": "#fdbf6f", "edgecolor": "black", "linewidth": 0.5, "alpha": 0.8},
}

def process_tile(base, z, x, y, extent):
    path = tile_to_path(base, z, x, y)
    if not os.path.exists(path): return []
    try:
        decoded = decode_tile(decompress_tile(path))
        features = []
        for layer, data in decoded.items():
            if layer in excluded_layers: continue
            for feat in data.get("features", []):
                geom = transform_geometry(x, y, z, feat.get("geometry"), extent)
                if geom:
                    features.append({"geometry": shape(geom), "layer": layer, **feat.get("properties", {})})
        return features
    except Exception as e:
        print(f"[ERROR] {e}")
        return []

# ------------------ Main Map Generator ------------------ #
def generate_map_image(base_dir, lat, lon, zoom, inches, dpi, extent=4096, output_dir="."):
    cx, cy = deg2num(lat, lon, zoom)
    tx_range = range(cx - 1, cx + 2)
    ty_range = range(cy - 1, cy + 2)

    bounds_top, _, bounds_left, _ = get_tile_boundaries(cx - 1, cy - 1, zoom)
    _, bounds_bot, _, bounds_right = get_tile_boundaries(cx + 1, cy + 1, zoom)
    filename = f"{bounds_bot:.8f}_{bounds_left:.8f}_{bounds_top:.8f}_{bounds_right:.8f}.png"
    output_path = os.path.join(output_dir, filename)

    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = [executor.submit(process_tile, base_dir, zoom, x, y, extent) for x in tx_range for y in ty_range]
        features = [f for fut in as_completed(futures) for f in fut.result()]

    if not features:
        print("[ERROR] No features extracted.")
        return

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.set_aspect("equal")
    plt.axis("off")

    for layer in ["landcover", "water", "waterway", "transportation", "boundary", "building"]:
        if layer in gdf["layer"].unique():
            gdf[gdf["layer"] == layer].plot(ax=ax, **layer_styles.get(layer, {}))

    ax.set_xlim(bounds_left, bounds_right)
    ax.set_ylim(bounds_bot, bounds_top)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[INFO] Saved to {output_path}")
    return output_path

# ------------------ Example Execution ------------------ #
if __name__ == "__main__":
    # generate_map_image(Directory, Latitude, Longitude, Zoom Level, Inches, DPI)
    generate_map_image(
        base_dir="/Users/jacobvaught/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/GPS_Project/MAPS/map_folder",
        lat=34.0007,
        lon=-81.0348,
        zoom=14,
        inches=10,
        dpi=300
    )