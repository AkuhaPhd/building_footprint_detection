import glob
import cv2
import sys
import matplotlib.pyplot as plt

from preprocessing import preprocess_image
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from postprocessing import mask_to_polygon, draw_polygons_on_image, show_annotations, write_polygons_to_geojson

# Import satellite or aerial images from the designated data directory
aerial_images = glob.glob("./data/*")

# Load segment anything (SAM) model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Segment major features in aerial image
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

image = preprocess_image(aerial_images[0])
masks = mask_generator_.generate(image)

# Show SAM result
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_annotations(masks)
plt.axis('off')
plt.show()

# Get all segmentations and create a polygon list
segmentations = [m["segmentation"] for m in masks]
polygons = []

for seg in segmentations:
    # Convert the mask to a polygon
    polygon = mask_to_polygon(seg)
    if polygon:
        polygons.append(polygon)

# Draw the polygons on the image
result_image = draw_polygons_on_image(image, polygons)

# Display the original image and the image with the drawn polygons
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Image with Polygons')
plt.show()

# Specify the output GeoJSON file path
output_geojson_file = "file.geojson"

# Write the list of polygons to the GeoJSON file
write_polygons_to_geojson(polygons, output_geojson_file)
