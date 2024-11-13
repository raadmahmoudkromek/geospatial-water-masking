import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.mask_stitching import stitch_from_metadata


class TestStitchMaskRaster(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patches_dir = os.path.join(self.temp_dir.name, 'patches')
        os.makedirs(self.patches_dir, exist_ok=True)

        self.original_filepath = os.path.join(self.temp_dir.name, 'original.tif')
        self.output_filepath = os.path.join(self.temp_dir.name, 'stitched_output.tif')
        self.patch_size = 128
        self.full_width, self.full_height = 256, 256  # 2x2 grid of patches for simplicity

        # Create a mock original file with the same properties
        transform = from_origin(0, 256, 1, 1)
        with rasterio.open(
                self.original_filepath, 'w', driver='GTiff',
                height=self.full_height, width=self.full_width,
                count=1, dtype=np.uint8, crs='EPSG:4326', transform=transform) as dst:
            dst.write(np.zeros((self.full_height, self.full_width), dtype=np.uint8), 1)

        # Generate test patches with unique values so they can be easily ID'd
        for patch_id in range(4):  # 2x2 patches
            # Calculate the correct transform for each patch
            row = patch_id // 2
            col = patch_id % 2

            # Calculate the top-left corner of each patch in the original image
            transform = from_origin(0 + col * self.patch_size, 256 - (row + 1) * self.patch_size, 1, 1)

            patch_data = np.full((self.patch_size, self.patch_size), patch_id, dtype=np.uint8)
            patch_filename = os.path.join(self.patches_dir, f"mask_patch_{patch_id}.tif")
            with rasterio.open(
                    patch_filename, 'w', driver='GTiff',
                    height=self.patch_size, width=self.patch_size,
                    count=1, dtype=np.uint8, crs='EPSG:4326',
                    transform=transform
            ) as patch_file:
                patch_file.write(patch_data, 1)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_raster_size(self):
        stitch_from_metadata(
            self.patches_dir,
            self.output_filepath,
        )

        # Check the size of the output raster
        with rasterio.open(self.output_filepath) as output_raster:
            self.assertEqual(output_raster.width, self.full_width)
            self.assertEqual(output_raster.height, self.full_height)

    def test_patch_placement(self):
        stitch_from_metadata(
            self.patches_dir,
            self.output_filepath,
        )

        # Check that patches were placed correctly in the output raster
        with rasterio.open(self.output_filepath) as output_raster:
            stitched_data = output_raster.read(1)

            self.assertTrue((stitched_data[0:128, 0:128] == 0).all())  # Patch 0
            self.assertTrue((stitched_data[0:128, 128:256] == 1).all())  # Patch 1
            self.assertTrue((stitched_data[128:256, 0:128] == 2).all())  # Patch 2
            self.assertTrue((stitched_data[128:256, 128:256] == 3).all())  # Patch 3


if __name__ == "__main__":
    unittest.main()
