import numpy.typing as nptyping
import skimage as ski

from abc import ABC, abstractmethod
from typing import Any


class DetectionMethodABC(ABC):
    """Abstract Basis Class for detection methods

    Args:
        ABC (ABC): Abstract Basis Class
    """

    @abstractmethod
    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        pass

    @staticmethod
    def binarize_image(arr: nptyping.NDArray) -> nptyping.NDArray:
        """Preprocess an image prior to probabilistic hough transform. Image is
        blurred using Gaussian blur and binarized by thresholding.

        Args:
            arr (nptyping.NDArray): image to preprocess

        Returns:
            nptyping.NDArray: preprocessed binary image
        """

        arr = ski.util.img_as_float(arr)
        arr = ski.filters.gaussian(arr, 3)
        return (arr > ski.filters.threshold_otsu(arr)).astype(bool)
