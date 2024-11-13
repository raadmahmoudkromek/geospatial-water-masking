# Geospatial Water Segmentation with Deep Learning
## Overview
This project explores a framework for water segmentation on satellite imagery.

`main.py` is broken into calls to functions which:
1) Retrieve satellite data from Sentinel-2
2) Crop those images into patches (`patch_extraction.py`), 
3) Run either a simple image processing or deep learning algorithm for water segmentation (`segmentation.py`)
4) Stitch the resultant masks back together to the size of the original raster (`mask_stitching.py`)

To prepare the virtual environment, from the project directory run

`conda env create -f environment.yml`

## Patch Extraction
### Overview
The src.patch_extraction.extract_patches function is used to split large GeoTIFF images into smaller, manageable patches for processing. The patches are saved as individual GeoTIFF files, each representing a section of the original image.

### Considerations for Patch Extraction
The patch size is set to [512 x 512] pixels by default. The tile size should be adjusted based on memory constraints and the resolution of the original image. For higher resolution images, we would likely use a smaller tile size to limit memory usage. 
For low res rasters, we could likely get away with larger tiles.

If overlap between tiles is required (to avoid edge artifacts), a sliding window approach could be used with a specified overlap percentage (e.g., 10-20%), but we do not use this approach here.

In the event of NoData pixels, we might in future consider replacement of these dead pixels to e.g. the local median, or by setting them to zero.

The patches are stored as uint8 values, which are suitable for binary masks or RGB images. If we required higher per-pixel precision, we may consider a float or long datatype for storage.

For water segmentation, the strength of the blue band over the other two bands is used in our simple approach. But since we are dealing with satellite data, we may incorporate other bands such as the NIR, which has been shown to improve network-based segmentation performance (e.g.https://doi.org/10.1016/j.rse.2023.113452)

Before feeding to a segmentation model, we would ensure that images are properly preprocessed before being fed into the segmentation model. E.g. if using a model pre-trained on ImageNet data, we would normalise to the mean and std of that set, and we would resize them to ensuring consistency in the input dimensions.





## Water Segmentation
### Simple Water Segmentation
This method uses a simple thresholding approach to detect water bodies based on pixel intensity, specifically looking for areas with high blue channel values. It's an efficient but less accurate method, especially when water bodies are not clearly distinguished by color thresholds.
### Deep Learning-Based Water Segmentation
In theory, the deep learning model could be used for more sophisticated segmentation, which would leverage a pre-trained network such as DeepLabV3 for semantic segmentation. We include the framework for this here in segmentation.py, but DeepLabV3 is not currently trained on geospatial water bodies, so would not yield good results currently.

### Considerations for Training a Model
#### Training/fine-tuning in 2 Weeks
To train or fine-tune a model within 2 weeks, we would use a pre-trained model like DeepLabV3 (already imported in an untrained form in segmentation.py). It's already been trained on ImageNet, and fine-tuning this should not take long provided we have a representative dataset.

We would aim to fine tune it using a existing water-mask dataset, or we could try and generate our own by building upon the simple segmentation function already made here, to try and remove smaller patches and morphologically close obvious bodies of water. After vetting, this could provide us with a simple dataset to begin with. 
Given that small set, we could use augmentation techniques such as rotation, flipping, and scaling to increase the variability of the training data and improve generalisation.

Once that data is available, we would freeze the weights of early layers of the network, and only fine-tune the last few layers, such as the decoder part of DeepLabV3. The most general features are already learned so a full retrain is probably unnecessary.
A job like this could probably be done on a single consumer-grade GPU.

### Measuring Model Performance
For binary segmentation, the pixel accuracy (percentage of correctly predicted pixels) is a simple and useful metric. Given that water bodies will be much less frequent than land though, F1 score would probably be a better metric for this problem.

To better understand where inaccuracies are, we might also calculate the Intersection over Union (IoU) foe each mask, which measures the overlap between the predicted mask and the ground truth.

### If we had 2 months
With more time, we could enlist annotators to label more data in a greater variety of terrains, improving model generalisation. Platforms like V7 or SuperAnnotate could be used for this.

Rather than an off-the-shelf DeepLabV3, we could train and modify bespoke models from scratch, based on more sophisticated architectures like Mask RCNN.

Given two months and therefore the opportunity for many mini-trains, we could look at optimising hyperparameters using something like a Bayesian hyperband search (via e.g. Weights + Biases)

### Deploying to Production for ~100 images per day
Once the model is fine-tuned and cross-validated, we would save the trained weights and export the model in a format that can be used for inference (e.g., ONNX).

We could set up a containerised (e.g. Docker) API (e.g. FastAPI) via Sagemaker or EC2 which would require only coordinates and time ranges for the input raster, and which would return the assembled mask.

If usage were to scale up from 100 images per day, we could implement AWS load balancing to spin up or down instances holding model clones, based on client demand. These would require monitoring middleware to ensure the service was stable, which AWS can provide.

Finally, we could create a feedback loop where users can validate and annotate results, allowing the model to be retrained periodically with updated data.