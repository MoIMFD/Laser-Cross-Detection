from collections import namedtuple

import cv2
import numpy as np
import numpy.typing as nptyping
import scipy
import skimage

ImageDimension = namedtuple("ImageDimension", "height width")


def rotate_image(
    image: nptyping.NDArray, angle: float, order: int = 3, impl: str = "cv"
) -> nptyping.NDArray:
    """Rotates an image by an angle.

    Args:
        image (nptyping.NDArray): image to rotate
        angle (float): angle to rotate the image in degree
        order (int, optional): order of the interpolation scheme to use.
            Defaults to 3.
        impl (str, optional): implementation to use.
            Defaults to "cv". Valid values are:
            openCV: cv, cv2, opencv, OpenCV
            scikit-image: skimage, scikit-image, ski
            scipy.ndimage: scipy, ndimage

    Returns:
        nptyping.NDArray: image rotate by the specified amount
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    if impl in ["cv", "cv2", "opencv", "OpenCV"]:
        orders = {
            0: cv2.INTER_NEAREST,
            1: cv2.INTER_LINEAR,
            3: cv2.INTER_CUBIC,
        }
        if order not in orders:
            raise NotImplementedError(f"order {order} not implemented in opencv")
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=orders[order])
    elif impl in ["skimage", "scikit-image", "ski"]:
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
