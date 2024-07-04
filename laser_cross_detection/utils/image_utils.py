import numpy as np
import cv2
import skimage
import scipy


def rotate_image(image, angle, order=3, impl="cv"):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    if impl in ["cv", "cv2", "opencv", "OpenCV"]:
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=order
        )
    elif impl in ["skimage", "scikit-image"]:
        result = skimage.transform.rotate(
            image,
            angle,
            resize=False,
            center=image_center,
            order=order,
        )
    elif impl in ["scipy", "ndimage"]:
        result = scipy.ndimage.rotate(image, angle, reshape=False, order=order)
    return result
