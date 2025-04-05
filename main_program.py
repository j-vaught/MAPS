import os
import math
import gzip
import io
import sys
import mapbox_vector_tile
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
#       Helper Functions
# -----------------------------

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
    """Convert tile x/y indices to lat/lon of the upper-left corner."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_path(base_dir, zoom, x, y):
    """
    Return the path to a .pbf.gz tile, .mvt, or .png, etc.
    Adapt the file extension for your data format.
    """
    filename = f"{y}.png"  # or .pbf, .pbf.gz, etc.
    path = os.path.join(base_dir, str(zoom), str(x), filename)
    
    print(f"[DEBUG] get_tile_path => zoom={zoom}, x={x}, y={y}")
    print(f"        Constructed path: {path}")
    print(f"        File exists? {os.path.exists(path)}\n")
    return path

def decompress_tile(tile_path):
    """Load gzipped tile data from disk."""
    with gzip.open(tile_path, 'rb') as f_in:
        data = f_in.read()
    return data

def decode_tile(tile_data):
    """Decode Mapbox vector tile data into a Python dict."""
    return mapbox_vector_tile.decode(tile_data)

def tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=4096):
    """
    Convert a point (x_local,y_local) in [0..extent]×[0..extent] tile coordinates
    to actual lat/lon, based on tile X/Y at given Z.

    By default, slippy map tile_y increases southward (so bigger tile_y => smaller lat).
    If you want the row of smaller tile_y to appear visually 'above' in the final image,
    we flip the way we map y_local -> lat. That is, we interpret y_local=0 as the BOTTOM
    and y_local=extent as the TOP, effectively reversing them.
    """
    top_lat, left_lon = num2deg(tile_x,   tile_y,   zoom)
    bot_lat, right_lon = num2deg(tile_x+1, tile_y+1, zoom)

    # Flip vertical so y_local=0 => lat=bot_lat, y_local=extent => lat=top_lat
    lat = bot_lat + (top_lat - bot_lat) * (y_local / extent)
    lon = left_lon + (right_lon - left_lon) * (x_local / extent)
    return lat, lon

def transform_geometry_to_latlon(tile_x, tile_y, zoom, geom_json, extent=4096):
    """
    Recursively walk the geometry (in tile-coords [0..extent]) and convert
    each point to lat/lon. Returns a new GeoJSON-like dict.
    """
    g_type = geom_json["type"]

    if g_type == "Point":
        x_local, y_local = geom_json["coordinates"]
        lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
        return {"type": "Point", "coordinates": (lon, lat)}

    elif g_type == "MultiPoint":
        new_coords = []
        for (x_local, y_local) in geom_json["coordinates"]:
            lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
            new_coords.append((lon, lat))
        return {"type": "MultiPoint", "coordinates": new_coords}

    elif g_type == "LineString":
        new_coords = []
        for (x_local, y_local) in geom_json["coordinates"]:
            lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
            new_coords.append((lon, lat))
        return {"type": "LineString", "coordinates": new_coords}

    elif g_type == "MultiLineString":
        new_lines = []
        for line in geom_json["coordinates"]:
            new_line = []
            for (x_local, y_local) in line:
                lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
                new_line.append((lon, lat))
            new_lines.append(new_line)
        return {"type": "MultiLineString", "coordinates": new_lines}

    elif g_type == "Polygon":
        new_rings = []
        for ring in geom_json["coordinates"]:
            new_ring = []
            for (x_local, y_local) in ring:
                lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
                new_ring.append((lon, lat))
            new_rings.append(new_ring)
        return {"type": "Polygon", "coordinates": new_rings}

    elif g_type == "MultiPolygon":
        new_polygons = []
        for poly in geom_json["coordinates"]:
            new_poly = []
            for ring in poly:
                new_ring = []
                for (x_local, y_local) in ring:
                    lat, lon = tile_extent_to_latlon(tile_x, tile_y, zoom, x_local, y_local, extent=extent)
                    new_ring.append((lon, lat))
                new_poly.append(new_ring)
            new_polygons.append(new_poly)
        return {"type": "MultiPolygon", "coordinates": new_polygons}

    else:
        # Possibly "GeometryCollection" or something else not handled
        return None

# -----------------------------
#    Styling / Exclusions
# -----------------------------
excluded_layers = {
    "poi", "water_name", "transportation_name", "place",
    "housenumber", "mountain_peak", "landuse"
}

layer_styles = {
    "water":           {"color": "#a6cee3", "edgecolor": "none", "alpha": 0.7},
    "waterway":        {"color": "#1f78b4", "linewidth": 0.8},
    "landcover":       {"color": "#b2df8a", "edgecolor": "none", "alpha": 0.7},
    "boundary":        {"color": "black",   "linestyle": "--", "linewidth": 0.8},
    "transportation":  {"color": "#fb9a99", "linewidth": 1},
    "building":        {"color": "#fdbf6f", "edgecolor": "black", "linewidth": 0.5, "alpha": 0.8},
}

# -----------------------------
#   Stitching 3x3 Tiles
# -----------------------------
def stitch_3x3_vector_tiles(base_dir, center_lat, center_lon, zoom,
                            extent=4096, fig_size=(12, 12), dpi=600):
    """
    Decode the 3x3 set of tiles around (center_lat,center_lon) at 'zoom'.
    Convert each tile's geometry from [0..extent] coords to lat/lon,
    then plot them all in a single figure. The PNG resolution is controlled
    by fig_size (in inches) and dpi (dots per inch).

    For a very high-resolution image:
        fig_size=(20,20), dpi=300
    which yields 6000 × 6000 pixels.
    """
    # 1) Find which tile covers our center lat/lon
    center_tile_x, center_tile_y = deg2num(center_lat, center_lon, zoom)
    print(f"[DEBUG] Center lat={center_lat}, lon={center_lon} => tile_x={center_tile_x}, tile_y={center_tile_y}")

    all_layers = []

    # Collect the features from the 3x3 set of neighboring tiles
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tile_x = center_tile_x + dx
            tile_y = center_tile_y + dy
            tile_path = get_tile_path(base_dir, zoom, tile_x, tile_y)
            
            if not os.path.exists(tile_path):
                print(f"Warning: Missing tile {tile_x}/{tile_y} at zoom {zoom} (Path: {tile_path})")
                continue
            
            print(f"[DEBUG] Loading tile {tile_x}/{tile_y} from: {tile_path}")
            try:
                data = decompress_tile(tile_path)
                decoded = decode_tile(data)
            except Exception as e:
                print(f"Error reading tile {tile_x}/{tile_y}: {e}")
                continue
            
            # For each layer in the tile, transform its features
            for layer_name, layer_data in decoded.items():
                if layer_name in excluded_layers:
                    continue
                features = layer_data.get("features", [])
                for feat in features:
                    geom_json = feat.get("geometry")
                    props = feat.get("properties", {})
                    
                    if geom_json is None:
                        continue
                    try:
                        # Transform geometry to lat/lon
                        geom_latlon = transform_geometry_to_latlon(
                            tile_x, tile_y, zoom, geom_json, extent=extent
                        )
                        if not geom_latlon:
                            continue
                        
                        shapely_geom = shape(geom_latlon)
                        all_layers.append({
                            "geometry": shapely_geom,
                            "layer": layer_name,
                            **props  # store feature properties if you want
                        })
                    except Exception as ex:
                        print(f"Geometry transform error in layer '{layer_name}': {ex}")

    if not all_layers:
        print("No features found in any of the 3x3 tiles!")
        return None

    # Create GeoDataFrame in lat/lon
    gdf = gpd.GeoDataFrame(all_layers, crs="EPSG:4326")

    # 2) Plot everything in one figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_aspect("equal", 'box')

    # Group by layer and apply styles
    for layer_name, sub_gdf in gdf.groupby("layer"):
        style = layer_styles.get(layer_name, {})
        sub_gdf.plot(ax=ax, **style)

    ax.set_title(f"3×3 Vector Tiles around ({center_lat:.4f}, {center_lon:.4f}) @ zoom {zoom}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Convert Matplotlib figure to a PIL.Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# -----------------------------
#   Example usage
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = "/Users/jacobvaught/Downloads/MAPS/map_folder"
    center_lat = 34.0007
    center_lon = -81.0348
    zoom = 3

    # Increase 'fig_size' and 'dpi' for higher-resolution
    # Example: fig_size=(20, 20), dpi=300 -> 6000×6000 px
    stitched_map = stitch_3x3_vector_tiles(
        base_dir=BASE_DIR,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom=zoom,
        extent=4096,
        fig_size=(20, 20),
        dpi=300
    )
    
    if stitched_map is not None:
        stitched_map.show()
        # Or stitched_map.save("stitched_flip.png")
    else:
        print("No image generated.")