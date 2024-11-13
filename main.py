import os

import numpy as np
import rasterio
import yaml
from matplotlib import pyplot as plt
from sentinelhub import DataCollection
from sentinelhub import SHConfig
from sentinelhub import SentinelHubRequest, MimeType, CRS, BBox

from src.mask_stitching import stitch_from_metadata
from src.patch_extraction import extract_patches
from src.segmentation import WaterSegmentationInference

sat_ims_path = os.path.join('outputs', 'satellite_images')

sentinelhub_config = SHConfig()
sentinelhub_config.instance_id = ''  # Can be found in sentinel hub dashboad -- may prefer to store as env variable and import with os.get_env() instead?
sentinelhub_config.sh_client_id = ''
sentinelhub_config.sh_client_secret = ''
sentinelhub_config.save()  # Save the configuration

# Load config YAML
with open(os.path.join("configs", "retrieval_and_extraction_cfg.yaml")) as file:
    orig_raster_config = yaml.safe_load(file)

# Define the area of interest and time interval from the config fle
bbox = BBox(bbox=(orig_raster_config['LONGITUDE_MIN'], orig_raster_config['LATITUDE_MIN'],
                  orig_raster_config['LONGITUDE_MAX'], orig_raster_config['LATITUDE_MAX']),
            crs=CRS.WGS84) #
time_interval = (orig_raster_config['DATE_MIN'], orig_raster_config['DATE_MAX'])  # Adjust dates as needed

########################################################################################################################
# Step 1: Request Sentinel-2 data, prioritising minimal cloud coverage (leastCC) #######################################
########################################################################################################################
request = SentinelHubRequest(
    data_folder=os.path.join(sat_ims_path),
    evalscript=orig_raster_config['SENT2_EVAL_SCRIPT'],
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            # Going for L1C as had trouble accessing Sentinel2-L2A data, but understand that has atmospheric corrections applied so would be preferable.
            time_interval=time_interval,
            mosaicking_order='leastCC'  # Allows us to prioritise the least cloudy image in the date interval
        )
    ],
    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
    bbox=bbox,
    size=(orig_raster_config['RASTER_Y_SIZE'], orig_raster_config['RASTER_X_SIZE']),
    config=sentinelhub_config
)

# Execute the request and save data
response = request.get_data(save_data=True)
saved_file = request.get_filename_list()[0]
response_hash = os.path.dirname(saved_file)
print(f"GeoTIFF saved at: {saved_file}")

# Load and display the saved GeoTIFF
with rasterio.open(os.path.join(sat_ims_path, saved_file)) as dataset:
    image = dataset.read([1, 2, 3])  # Read RGB bands
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.axis('off')
plt.show()

########################################################################################################################
# Step 2: Extract image patches ########################################################################################
########################################################################################################################
extract_patches(
    input_filepath=os.path.join(sat_ims_path, saved_file),
    output_dir=os.path.join(sat_ims_path, response_hash, 'patches'),
    patch_size=orig_raster_config['PATCH_SIZE'],
)

########################################################################################################################
# Step 3 Instantiate and run the water segmentation class, #############################################################
# specifiying whether we want to use the simple inference tool or the DeepLab tool (untrained). ########################
########################################################################################################################
water_segmenter = WaterSegmentationInference(method='simple')
water_segmenter.run_inference_on_patches(input_dir=os.path.join(sat_ims_path, response_hash, 'patches'),
                                         output_dir=os.path.join(sat_ims_path, response_hash, 'water_masks'))

########################################################################################################################
# Step 4: Assemble the mask patches into a raster of the original image size. ##########################################
########################################################################################################################
stitch_from_metadata(
    patches_dir=os.path.join(sat_ims_path, response_hash, 'water_masks'),
    output_filepath=os.path.join(sat_ims_path, response_hash, 'reconstructed_water_mask.tif'),
)
