#main.py

import os
import math
from PIL import Image, ImageDraw

# Import your new backend function
# For example, if your backend is in 'generate_map.py':
from generate_map import generate_map_image

def crop_map_fixed_size(
    base_dir,
    center_lat,
    center_lon,
    zoom=14,
    inches=10,
    dpi=300,
    extent=4096,
    output_dir=".",
    out_width_px=600,    # <--- Desired final width in pixels
    out_height_px=600,   # <--- Desired final height in pixels
    marker_radius=20
):

    # 1) Generate the stitched map with your backend
    map_path = generate_map_image(
        base_dir=base_dir,
        lat=center_lat,
        lon=center_lon,
        zoom=zoom,
        inches=inches,
        dpi=dpi,
        extent=extent,
        output_dir=output_dir
    )
    if not map_path or not os.path.exists(map_path):
        print("[ERROR] Could not generate/find the stitched map.")
        return

    # 2) Parse bounding corners from filename: e.g. "bot_left_top_right.png"
    filename = os.path.basename(map_path)
    corners = filename.replace(".png", "").split("_")
    if len(corners) != 4:
        print("[ERROR] Could not parse <bot>_<left>_<top>_<right> from:", filename)
        return
    bot, left, top, right = map(float, corners)

    # 3) Open in Pillow and define lat/lon => pixel transforms
    img = Image.open(map_path)
    width_px, height_px = img.size

    # The image covers longitude in [left, right], latitude in [bot, top].
    # lat decreases as y increases, so top => row 0, bot => row height_px.
    def lon_to_x(lon):
        return (lon - left) / (right - left) * width_px

    def lat_to_y(lat):
        return (top - lat) / (top - bot) * height_px

    # Convert the desired center lat/lon to pixel coords
    cx = lon_to_x(center_lon)
    cy = lat_to_y(center_lat)

    # 4) Crop: we want out_width_px x out_height_px around (cx, cy)
    # The box in Pillow is (left_px, upper_px, right_px, lower_px)
    left_px  = int(cx - out_width_px / 2)
    upper_px = int(cy - out_height_px / 2)
    right_px = left_px + out_width_px
    lower_px = upper_px + out_height_px

    # Clamp to image boundaries
    left_px  = max(0, left_px)
    upper_px = max(0, upper_px)
    right_px = min(width_px, right_px)
    lower_px = min(height_px, lower_px)

    # If the requested crop is partially out of bounds, we'll get a smaller region
    cropped = img.crop((left_px, upper_px, right_px, lower_px))

    # 5) Draw the marker at the new local coords
    # The center inside 'cropped' is (cx - left_px, cy - upper_px)
    draw = ImageDraw.Draw(cropped)

    local_cx = cx - left_px
    local_cy = cy - upper_px

    # Outer black ring
    # We'll define ring_radius from the function param
    left_ring = local_cx - marker_radius
    right_ring = local_cx + marker_radius
    top_ring = local_cy - marker_radius
    bottom_ring = local_cy + marker_radius
    # Draw black ring (outline):
    draw.ellipse([left_ring, top_ring, right_ring, bottom_ring],
                 outline="black", width=int(marker_radius/8), fill=None)

    # White ring on top, slightly smaller, with a blue fill
    ring_radius2 = marker_radius - 2  # or whatever offset you like
    left_ring2   = local_cx - ring_radius2
    right_ring2  = local_cx + ring_radius2
    top_ring2    = local_cy - ring_radius2
    bottom_ring2 = local_cy + ring_radius2
    draw.ellipse([left_ring2, top_ring2, right_ring2, bottom_ring2],
                 outline="white", width = int(marker_radius / 3), fill="#4285F4")

    # 6) Save the final
    final_name = "final_cropped_map.png"
    final_path = os.path.join(output_dir, final_name)
    cropped.save(final_path)
    print(f"[INFO] Final cropped image saved at: {final_path}")

if __name__ == "__main__":
    BASE_DIR = "/Users/jacobvaught/Library/CloudStorage/OneDrive-UniversityofSouthCarolina/GPS_Project/MAPS/map_folder"
    center_lat = 34.0007
    center_lon = -81.0348
    zoom = 14

    # Example: produce a 600×400 cropped image around the center
    crop_map_fixed_size(
        base_dir=BASE_DIR,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom=zoom,
        inches=48, #adjust this to get the 'zoom' you want.
        dpi=176, #keep this at 176 for easier math later lol [specific to our screen]. 
        #this means 1 square inch on the image will fit our screen perfectly 
        # (so if inches is set to 20, then the resulting image will be 1/20 of the 3 tiles)
        extent=4096,
        output_dir=".",
        out_width_px=176,#specific to our screen
        out_height_px=176,#specific to our screen
        marker_radius=16 #<--- Adjust this to change the size of the marker
    )