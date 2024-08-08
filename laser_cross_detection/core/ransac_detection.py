import numpy as np
import numpy.typing as nptyping

from scipy.sparse import csr_matrix
import sklearn as skl

from typing import List, Tuple

from .detection_abc import DetectionMethodABC
from . import HessNormalLine


class Ransac(DetectionMethodABC):
    """Laser Cross Detection Method based on Ransac Algorithm. Implementation
    by Robert Hardege. Details provide in
    https://doi.org/10.1007/s00348-023-03729-1

    Minor changes to fit in the new frame by Kluwe
    """

    def __call__(
        self, arr: nptyping.NDArray, *args, **kwargs
    ) -> nptyping.NDArray:
        """Calculates the point of intersection of two beams in an image
        containing both.

        Args:
            arr (nptyping.NDArray): image with beams

        Returns:
            nptyping.NDArray: point of intersection
        """
        # arr_copy = np.copy(arr)
        # gaussian filter & binarize image/array
        arr = Ransac.binarize_image(arr)

        # convert arr to x y list of white pixel coordinates
        indices = Ransac.__get_indices_sparse(arr)
        x, y = indices[1][1], indices[1][0]
        # detect first line
        coef_1, intrcpt_1, res_x, res_y = Ransac.__ransac(x, y)
        # detect second line
        coef_2, intrcpt_2, _, _ = Ransac.__ransac(res_x, res_y)

        # calc intersection point
        line1 = HessNormalLine.from_intercept_and_slope(intrcpt_1, coef_1)
        line2 = HessNormalLine.from_intercept_and_slope(intrcpt_2, coef_2)

        return line1.intersect_crossprod(line2)

    @classmethod
    def __ransac(
        self, x: nptyping.NDArray, y: nptyping.NDArray
    ) -> Tuple[float, float, List[float], List[float]]:
        """Performs ransac algorithm on a set of points and returns slope,
        intercept and points with highest residuals.

        Args:
            x (nptyping.NDArray): x coordinates
            y (nptyping.NDArray): y coordinates

        Returns:
            Tuple[float, float, List[float], List[float]]: slope, intercept,
                x coordinate for points with highest residuals,
                y coordinate for points with highest residuals
        """
        ransac = skl.linear_model.RANSACRegressor(
            stop_probability=0.9999,
            residual_threshold=10,
            max_trials=500,
            loss="absolute_error",
        )
        ransac.fit(x[:, np.newaxis], y[:, np.newaxis])
        inlier_mask = ransac.inlier_mask_.astype(bool)

        # get residual x, y (outliers)
        res_x = x[~inlier_mask]
        res_y = y[~inlier_mask]

        # get coefficient/y-interception
        coef = ransac.estimator_.coef_[0][0]
        intrcpt = ransac.estimator_.intercept_[0]

        return coef, intrcpt, np.array(res_x), np.array(res_y)

    # this is some fast magic to get the indices from a numpy array:
    # https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
    @classmethod
    def __compute_M(self, data: nptyping.NDArray) -> csr_matrix:
        cols = np.arange(data.size)
        return csr_matrix(
            (cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size)
        )

    @classmethod
    def __get_indices_sparse(self, data: nptyping.NDArray) -> List[Tuple[int]]:
        M = self.__compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]
