import os

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import ndimage as ndimage
from torchvision import models, transforms


def simple_water_segmentation(patch: np.ndarray, min_area: int = 1000) -> np.ndarray:
    """
    Generate a binary mask with noise reduction for water in an RGB patch.
    In this simple algo, water is detected if the blue channel is the strongest channel, and noise is reduced
    by removing small isolated pixels and smoothing edges.

    Parameters:
    - patch: A 3D NumPy array of shape (3, height, width) representing the RGB image patch.
    - min_area: Minimum area (in pixels) for water regions to be kept, to remove small speckles.

    Returns:
    - water_mask: A 2D NumPy binary array of the same width and height as the input patch.
    """
    # TODO Consider edge detection first with e.g. Sobel filters.

    # Threshold values
    blue_strength_threshold = 0.15
    blue_dominance_factor = 1.1

    # Extract channels
    red, green, blue = patch[0], patch[1], patch[2]

    # Create a binary mask for water
    water_mask = (
            (blue > blue_strength_threshold) &
            (blue > blue_dominance_factor * red) &
            (blue > blue_dominance_factor * green)
    )
    water_mask = water_mask.astype(np.uint8)

    # Identify regions by analysing connected components.
    labeled_mask, num_features = ndimage.label(water_mask)
    region_sizes = ndimage.sum(water_mask, labeled_mask,
                               range(num_features + 1))  # 1D array describing sizes of each region

    # Keep only regions larger than `min_area`
    large_regions = region_sizes >= min_area  # Boolean array describing regions larger than threshold
    filtered_water_mask = large_regions[labeled_mask]

    # Apply morphological closing to try and fill small gaps
    filtered_water_mask = 255 * ndimage.binary_closing(filtered_water_mask, structure=np.ones((3, 3))).astype(np.uint8)

    return filtered_water_mask


# The below is mostly a dummy class which uses a pretrained DeepLabV3 instance to produce binary masks.
# Of course, this DeepLab model is not trained for geospatial water segmentation, and so its predictions would currently be rubbish.

class DeepLabInference:
    def __init__(self):
        # Load the DeepLabV3 model with pretrained weights
        self.model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()

        # Transform for input images, here, to match the means and standard deviations in the ImageNet set
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run_inference(self, patch: np.ndarray) -> np.ndarray:
        """
        Run inference using the DeepLabV3 model and generate a binary water mask.

        Parameters:
        - patch: A numpy array representing the image patch (RGB channels).

        Returns:
        - binary_mask: A binary mask indicating water areas.
        """
        # Convert the image patch to a PIL Image and preprocess
        input_image = Image.fromarray(np.transpose(patch, (1, 2, 0)))
        input_tensor = self.preprocess(input_image).unsqueeze(0)  # Create a batch of 1

        # Run the model in evaluation mode, ie. without gradient calculations
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]

        # Get segmentation result by taking the argmax of the output logits
        segmentation = torch.argmax(output, 0).numpy()

        # Create binary water mask; 19 is assumed as the water class label
        binary_mask = np.where(segmentation == 19, 255, 0).astype(np.uint8)
        return binary_mask


class WaterSegmentationInference:
    def __init__(self, method="simple"):
        """
        Initializes the WaterSegmentationInference with a specified method.

        Parameters:
        - method: A string, either "simple" for simple thresholding segmentation or "deep" for DeepLabV3 segmentation.
        """
        self.method = method

        if self.method == "deep":
            # Load DeepLabV3 model for deep segmentation
            self.deeplab_inference = DeepLabInference()
        elif self.method == "simple":
            # Use simple segmentation function
            self.segmentation_function = simple_water_segmentation
        else:
            raise ValueError("Invalid method. Choose 'simple' or 'deep'.")

    def segment_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Segments a single patch based on the chosen method.

        Parameters:
        - patch: A numpy array representing the image patch.

        Returns:
        - A binary mask numpy array.
        """
        if self.method == "simple":
            return self.segmentation_function(patch)
        elif self.method == "deep":
            return self.deeplab_inference.run_inference(patch)

    def run_inference_on_patches(self, input_dir: str, output_dir: str) -> None:
        """
        Runs water segmentation on each patch and saves the binary mask.

        Parameters:
        - input_dir: Directory containing the image patches.
        - output_dir: Directory where the binary masks will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        for patch_filename in os.listdir(input_dir):
            patch_path = os.path.join(input_dir, patch_filename)
            with rasterio.open(patch_path) as patch_dataset:
                patch = patch_dataset.read([1, 2, 3])  # Load RGB bands
                binary_mask = self.segment_patch(patch)

                # Save the binary mask
                mask_filename = os.path.join(output_dir, f"mask_{patch_filename}")
                with rasterio.open(
                        mask_filename,
                        'w',
                        driver='GTiff',
                        height=binary_mask.shape[0],
                        width=binary_mask.shape[1],
                        count=1,  # Binary mask has a single channel
                        dtype='uint8',
                        crs=patch_dataset.crs,
                        transform=patch_dataset.transform
                ) as mask_dataset:
                    mask_dataset.write(binary_mask, 1)

                print(f"Saved mask at {mask_filename}")
