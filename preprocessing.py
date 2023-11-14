import cv2
import numpy as np


def preprocess_image(image, normalize=False):
    """
    Resize an image while maintaining its original aspect ratio.

    Parameters:
    - image: The input image (NumPy array).
    - normalize: Whether to normalize pixel values to the range [0, 1].

    Returns:
    - resized_image: The resized image.
    """
    # Calculate aspect ratio
    image = cv2.imread(image)
    original_width, original_height = image.shape[:2]
    down_size = original_height//420
    target_size = (original_height//down_size, original_width//down_size)

    # Resize the image
    resized_img = cv2.resize(image, target_size)

    # Normalize pixel values to the range [0, 1] if specified
    if normalize:
        # Convert the image to float32
        resized_img = resized_img.astype(np.float32)
        resized_img /= 255.0

    return resized_img


