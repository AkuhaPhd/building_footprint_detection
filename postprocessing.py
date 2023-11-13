import cv2
import geojson
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def mask_to_polygon(binary_mask):
    """
    Convert a binary mask to a Shapely Polygon.

    Parameters:
    - binary_mask: NumPy array representing a binary mask.

    Returns:
    - polygon: Shapely Polygon representing the contour of the binary mask.
              Returns None if the mask is empty.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the first contour (assuming there's only one object in the mask)
    if len(contours) > 0:
        contour = contours[0]

        # Create a polygon from the contour
        polygon = Polygon([tuple(point[0]) for point in contour])
        return polygon

    return None


def draw_polygons_on_image(image, polygons, color=(0, 255, 0), thickness=2):
    """
    Draw a list of polygons on the given image.

    Parameters:
    - image: The input image (NumPy array).
    - polygons: List of Shapely Polygons to be drawn.
    - color: Color of the polygon outlines (BGR format).
    - thickness: Thickness of the polygon outlines.

    Returns:
    - image_with_polygons: Image with the polygons drawn on it.
    """
    # Create a copy of the image to avoid modifying the original
    image_with_polygons = np.copy(image)

    # Draw each polygon on the image
    for polygon in polygons:
        # Extract the coordinates of the polygon's exterior ring
        polygon_coords = np.array(polygon.exterior.xy).T.astype(np.int32)

        # Draw the polygon on the image
        cv2.polylines(image_with_polygons, [polygon_coords], isClosed=True, color=color, thickness=thickness)

    return image_with_polygons


def show_annotations(anns):
    """
    Display image annotations represented by a list of dictionaries.

    Parameters:
    - anns: List of dictionaries representing image annotations.
           Each dictionary should have a 'segmentation' key containing a binary mask (NumPy array),
           and an 'area' key representing the area of the annotation.

    Returns:
    - None

    Note:
    - The function uses matplotlib to display the annotations.
    - The annotations are sorted based on area in descending order before display.
    - If the input list is empty, the function returns without displaying anything.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def write_polygons_to_geojson(polygons, output_file):
    """
    Write a list of Shapely Polygons to a GeoJSON file.

    Parameters:
    - polygons: List of Shapely Polygons.
    - output_file: Output GeoJSON file path.
    """
    # Convert each Shapely Polygon to a GeoJSON feature
    features = [geojson.Feature(geometry=geojson.Polygon([list(polygon.exterior.coords)])) for polygon in polygons]

    # Create a GeoJSON feature collection
    feature_collection = geojson.FeatureCollection(features)

    # Write GeoJSON to the specified file
    with open(output_file, 'w') as geojson_file:
        geojson.dump(feature_collection, geojson_file)
