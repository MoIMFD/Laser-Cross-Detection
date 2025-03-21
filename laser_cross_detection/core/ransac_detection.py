import numpy as np
import numpy.typing as nptyping

from scipy.sparse import csr_matrix
import sklearn as sklearn


from typing import List, Tuple

from .detection_abc import DetectionMethodABC
from . import HessNormalLine, ComplexHessLine


class Ransac(DetectionMethodABC):
    """Laser Cross Detection Method based on Ransac Algorithm. Implementation
    by Robert Hardege. Details provide in
    https://doi.org/10.1007/s00348-023-03729-1

    Minor changes to fit in the new frame by Kluwe
    """

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(
        self,
        arr: nptyping.NDArray,
        return_lines: bool = False,
        slope_thresh=5,
        *args,
        **kwargs,
    ) -> nptyping.NDArray:
        """Calculates the point of intersection of two beams in an image
        containing both.

        Args:
            arr (nptyping.NDArray): image with beams

        Returns:
            nptyping.NDArray: point of intersection
        """
        arr = self.binarize_image(arr)
        image_center = np.array(arr.shape[::-1]) / 2
        # arr_copy = np.copy(arr)
        # gaussian filter & binarize image/array

        # convert arr to x y list of white pixel coordinates
        indices = self.__get_indices_sparse(arr)
        x, y = np.array(indices[1][1], dtype=float), np.array(
            indices[1][0], dtype=float
        )
        x = x.copy() - image_center[0]
        y = y.copy() - image_center[1]

        # detect first line
        coef_1, intercept_1, res_x, res_y = self.__ransac(x, y)
        # if the slope is too steep, flip x and y
        if abs(coef_1) > slope_thresh:
            coef_1, intercept_1, temp_x, temp_y = self.__ransac(-y, x)

            normal = np.array((-coef_1, 1))
            normal = normal / np.linalg.norm(normal)
            distance = np.dot([0, intercept_1], normal)
            angle = np.arctan2(normal[1], normal[0])

            line1 = ComplexHessLine.from_intercept_and_slope(
                intercept_1, coef_1, center=image_center
            )

            # image_center = np.flip(image_center)
            res_x, res_y = temp_y, -temp_x
            line1 = ComplexHessLine.from_intercept_and_slope(
                intercept_1, coef_1, center=image_center
            )
            # restore original orientation
            line1.rotate(-np.pi / 2)
        else:
            line1 = ComplexHessLine.from_intercept_and_slope(
                intercept_1, coef_1, center=image_center
            )

        # detect second line
        coef_2, intercept_2, _, _ = self.__ransac(res_x, res_y)
        # if the slope is too steep, flip x and y
        if abs(coef_2) > slope_thresh:
            coef_2, intercept_2, _, _ = self.__ransac(-res_y, res_x)
            line2 = ComplexHessLine.from_intercept_and_slope(
                intercept_2, coef_2, center=image_center
            )
            line2.rotate(-np.pi / 2)
        else:
            line2 = ComplexHessLine.from_intercept_and_slope(
                intercept_2, coef_2, center=image_center
            )

        if return_lines:
            return (
                line1.intersect(line2),
                line1,
                line2,
            )
        else:
            return line1.intersect(line2)

    def __ransac(
        self, x: nptyping.NDArray, y: nptyping.NDArray
    ) -> Tuple[float, Tuple[float, float], np.ndarray[float], np.ndarray[float]]:
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
        ransac = sklearn.linear_model.RANSACRegressor(
            stop_probability=0.9999,
            residual_threshold=10,
            max_trials=500,
            loss="absolute_error",
            random_state=self.seed,
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

    def __compute_M(self, data: nptyping.NDArray) -> csr_matrix:
        cols = np.arange(data.size)
        return csr_matrix(
            (cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size)
        )

    def __get_indices_sparse(self, data: nptyping.NDArray) -> List[Tuple[int, int]]:
        M = self.__compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]
