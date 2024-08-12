import numpy as np
import numpy.typing as nptyping
import cv2

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
        arr = cv2.convertScaleAbs(arr)
        blur = cv2.GaussianBlur(arr, (5, 5), 0)
        _, arr = cv2.threshold(
            np.array(blur, dtype=np.uint16),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        arr = arr.astype(bool)
        return arr
