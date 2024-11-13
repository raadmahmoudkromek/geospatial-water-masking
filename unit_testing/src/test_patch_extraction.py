import os
import shutil
import unittest
import numpy as np
import rasterio
from pathlib import Path
from src.patch_extraction import extract_patches

class TestPatchExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up a temporary directory and a sample GeoTIFF for testing
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        cls.input_filepath = cls.test_dir / "test_image.tif"
        cls.output_dir = cls.test_dir / "patches"
        cls.patch_size = 256

        # Create a mock GeoTIFF file with random data for testing
        with rasterio.open(
            cls.input_filepath, 'w',
            driver='GTiff',
            height=1024,
            width=1024,
            count=3,
            dtype='uint8',
            crs='+proj=latlong'
        ) as dataset:
            dataset.write(np.random.randint(0, 255, (3, 1024, 1024), dtype='uint8'))

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files and directories after tests
        shutil.rmtree(cls.test_dir)  # Removes the entire test_dir and its contents

    def test_patch_creation(self):
        # Run the patch extraction function
        extract_patches(input_filepath=self.input_filepath, output_dir=self.output_dir, patch_size=self.patch_size)

        # Check that patches are created in the output directory
        patches = list(self.output_dir.glob("patch_*.tif"))
        self.assertTrue(len(patches) > 0, "No patches created in output directory")

    def test_patch_dimensions(self):
        # Run the patch extraction function
        extract_patches(input_filepath=self.input_filepath, output_dir=self.output_dir, patch_size=self.patch_size)

        # Load the first patch and check its dimensions
        first_patch = next(self.output_dir.glob("patch_*.tif"))
        with rasterio.open(first_patch) as patch:
            self.assertEqual(patch.width, self.patch_size, f"Expected width {self.patch_size}, got {patch.width}")
            self.assertEqual(patch.height, self.patch_size, f"Expected height {self.patch_size}, got {patch.height}")

if __name__ == "__main__":
    unittest.main()
