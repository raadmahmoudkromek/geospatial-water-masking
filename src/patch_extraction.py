import os

import rasterio
from rasterio.windows import Window


def extract_patches(input_filepath: str, output_dir: str, patch_size: int = 512) -> None:
    """
    Extracts square patches from a GeoTIFF and saves them as individual files.

    Parameters:
    - input_filepath: Path to the input GeoTIFF file.
    - output_dir: Directory where the patches will be saved.
    - patch_size: Size of each square patch in pixels (e.g., 128x128).
    """

    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_filepath) as dataset:

        # Get the number of columns and rows in the image
        img_width, img_height = dataset.width, dataset.height

        # Read through the raster in patch-sized chunks
        patch_id = 0
        for row_start in range(0, img_height, patch_size):
            for col_start in range(0, img_width, patch_size):
                # Define a window to read just this patch
                window = Window(col_start, row_start, patch_size, patch_size)

                # Store the transform to inject into the saved raster patch later
                transform = dataset.window_transform(window)

                # Read the image data within the window
                patch = dataset.read([1, 2, 3], window=window)  # For RGB data

                # Generate a unique filename for each patch
                patch_filename = os.path.join(output_dir, f"patch_{patch_id}.tif")

                # Save the patch as a new GeoTIFF file
                with rasterio.open(
                        patch_filename, 'w',
                        driver='GTiff',  # Save as a GeoTIFF file
                        height=patch.shape[1],
                        width=patch.shape[2],
                        count=3,  # No. of bands
                        dtype=patch.dtype,
                        crs=dataset.crs,
                        transform=transform
                ) as patch_file:
                    patch_file.write(patch)

                patch_id += 1
                print(f"Saved patch {patch_id} at {patch_filename}")
