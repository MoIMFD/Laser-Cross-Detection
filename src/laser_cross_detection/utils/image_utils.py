from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import cv2
import scipy
import skimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RotateFunc(Protocol):
    """Protocol for image rotation functions."""

    def __call__(self, image: NDArray, angle: float, order: int) -> NDArray: ...


class ImageDimension(NamedTuple):
    height: int
    width: int


_CV_ORDER_MAP = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    3: cv2.INTER_CUBIC,
}


def _rotate_opencv(image: NDArray, angle: float, order: int) -> NDArray:
    """OpenCV implementation of image rotation."""
    if order not in _CV_ORDER_MAP:
        raise NotImplementedError(f"order {order} not implemented in opencv")
    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=_CV_ORDER_MAP[order])


def _rotate_skimage(image: NDArray, angle: float, order: int) -> NDArray:
    """scikit-image implementation of image rotation."""
    image_center = (image.shape[1] * 0.5, image.shape[0] * 0.5)
    return skimage.transform.rotate(
        image,
        angle,
        resize=False,
        center=image_center,
        order=order,
    )


def _rotate_scipy(image: NDArray, angle: float, order: int) -> NDArray:
    """scipy.ndimage implementation of image rotation."""
    return scipy.ndimage.rotate(image, angle, reshape=False, order=order)


@dataclass(frozen=True)
class RotateImpl:
    """Image rotation implementation.

    Attributes:
        name: Human-readable name of the implementation
        func: The rotation function to use
    """

    name: str
    func: RotateFunc

    def __call__(self, image: NDArray, angle: float, order: int) -> NDArray:
        """Execute the rotation function.

        Args:
            image: Image to rotate
            angle: Rotation angle in degrees
            order: Interpolation order

        Returns:
            Rotated image
        """
        return self.func(image, angle, order)


class Implementations:
    """Available rotation implementations.

    Attributes:
        OPENCV: OpenCV implementation (cv2.warpAffine) - fastest option
        SKIMAGE: scikit-image implementation
        SCIPY: scipy.ndimage implementation
    """

    OPENCV = RotateImpl("opencv", _rotate_opencv)
    SKIMAGE = RotateImpl("skimage", _rotate_skimage)
    SCIPY = RotateImpl("scipy", _rotate_scipy)


def rotate_image(
    image: NDArray,
    angle: float,
    order: int = 3,
    impl: RotateImpl | str = Implementations.OPENCV,
) -> NDArray:
    """Rotates an image by an angle.

    Args:
        image (NDArray): image to rotate
        angle (float): angle to rotate the image in degree
        order (int, optional): order of the interpolation scheme to use.
            Defaults to 3.
        impl (RotateImpl | str, optional): implementation to use.
            Defaults to Implementations.OPENCV.
            If str, valid values are:
            openCV: cv, cv2, opencv, OpenCV
            scikit-image: skimage, scikit-image, ski
            scipy.ndimage: scipy, ndimage

    Returns:
        NDArray: image rotate by the specified amount
    """
    if isinstance(impl, str):
        impl = _parse_impl_string(impl)

    return impl(image, angle, order)


def _parse_impl_string(impl: str) -> RotateImpl:
    """Parse legacy string implementation identifiers to RotateImpl."""
    if impl in ["cv", "cv2", "opencv", "OpenCV"]:
        return Implementations.OPENCV
    elif impl in ["skimage", "scikit-image", "ski"]:
        return Implementations.SKIMAGE
    elif impl in ["scipy", "ndimage"]:
        return Implementations.SCIPY
    else:
        raise ValueError(
            f"Unknown implementation: {impl}. Valid values: opencv, skimage, scipy"
        )
