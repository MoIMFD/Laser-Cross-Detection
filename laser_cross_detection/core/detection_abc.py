from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import skimage as ski

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DetectionMethodABC(ABC):
    """Abstract Basis Class for detection methods

    Args:
        ABC (ABC): Abstract Basis Class
    """

    @abstractmethod
    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        pass

    @staticmethod
    def binarize_image(arr: NDArray) -> NDArray:
        """Preprocess an image prior to probabilistic hough transform. Image is
        blurred using Gaussian blur and binarized by thresholding.

        Args:
            arr (NDArray): image to preprocess

        Returns:
            NDArray: preprocessed binary image
        """

        arr = ski.util.img_as_float(arr)
        arr = ski.filters.gaussian(arr, 3)
        return (arr > ski.filters.threshold_otsu(arr)).astype(bool)
