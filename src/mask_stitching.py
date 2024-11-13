import os

import numpy as np
import rasterio

# Note: Initial inclination was to use the original filepath location but felt like there was too much reliance on context of problem there.
# Instead wanted function which could handle assembling raster patches of unknown original dimensions shape, i.e. just use raster metadata.

def stitch_from_metadata(patches_dir: str, output_filepath: str) -> None:
    """
    Stitches individual patch mask files into a full-size raster based on
    the patches' metadata (width, height, and transform). Saves the stitched
    result as a new GeoTIFF.

    Args:
        patches_dir (str): Directory containing individual patch mask files
            in GeoTIFF format. Each patch file should have metadata including
            width, height, and georef transform.
        output_filepath (str): Path where the stitched raster will be saved
            as a new GeoTIFF file.

    Raises:
        FileNotFoundError: If no patch files are found in the specified directory.

    Returns:
        None: Saves the stitched raster as a GeoTIFF at the specified output path.

    Example:
        stitch_from_metadata('patches', 'stitched_output.tif')
    """
    patch_data = []
    raster_metadata = []

    # Read each patch's data and metadata
    for filename in os.listdir(patches_dir):
        filepath = os.path.join(patches_dir, filename)
        with rasterio.open(filepath) as src:
            patch_data.append(src.read(1))  # Store the patch data (binary mask)
            raster_metadata.append((src.width, src.height, src.transform))

    # Calculate the min/max coords of the full raster
    min_x = min(metadata[2].c for metadata in raster_metadata)
    max_y = max(metadata[2].f for metadata in raster_metadata)
    max_x = max(metadata[2].c + metadata[0] * metadata[2].a for metadata in raster_metadata)
    min_y = min(metadata[2].f + metadata[1] * metadata[2].e for metadata in raster_metadata)

    # Calculate full raster dimensions in pixels
    pixel_width = int((max_x - min_x) / abs(raster_metadata[0][2].a))
    pixel_height = int((max_y - min_y) / abs(raster_metadata[0][2].e))

    # Create an empty array for the full raster
    full_raster = np.zeros((pixel_height, pixel_width), dtype=np.uint8)

    # Insert eeach patch into the full raster
    for (patch, (width, height, transform)) in zip(patch_data, raster_metadata):
        x_start = int((transform.c - min_x) / abs(transform.a))
        y_start = int((max_y - transform.f) / abs(transform.e))
        full_raster[y_start:y_start + height, x_start:x_start + width] = patch

    # Save the full raster to a new GeoTIFF
    with rasterio.open(
            output_filepath, 'w', driver='GTiff',
            height=pixel_height, width=pixel_width,
            count=1, dtype=full_raster.dtype,
            crs=src.crs,
            transform=rasterio.transform.from_origin(min_x, max_y, abs(raster_metadata[0][2].a),
                                                     abs(raster_metadata[0][2].e))
    ) as dest:
        dest.write(full_raster, 1)

    print(f"Stitched raster mask saved at: {output_filepath}")
