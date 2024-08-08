import cv2
import numpy as np
import numpy.typing as nptyping
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn import linear_model

from .detection_abc import DetectionMethodABC
from . import HessNormalLine


class Ransac(DetectionMethodABC):
    """Laser Cross Detection Method based on Ransac Algorithm. Implementation
    by Robert Hardege. Details provide in
    https://doi.org/10.1007/s00348-023-03729-1

    Minor changes to fit in the new frame by Kluwe
    """

    def __call__(self, arr: nptyping.NDArray, *args, **kwargs):
        # arr_copy = np.copy(arr)
        # gaussian filter & binarize image/array
        arr = Ransac.__preprocess(arr)

        # convert arr to x y list of white pixel coordinates
        indices = Ransac.__get_indices_sparse(arr)
        x, y = indices[1][1], indices[1][0]
        # detect first line
        coef_1, intrcpt_1, res_x, res_y = Ransac.__ransac(x, y)
        # detect second line
        coef_2, intrcpt_2, _, _ = Ransac.__ransac(res_x, res_y)

        def plot():
            plt.matshow(arr)
            plt.plot(
                [0, arr.shape[1]],
                [intrcpt_1, intrcpt_1 + arr.shape[1] * coef_1],
            )
            plt.xlim([0, arr.shape[1]])
            plt.ylim([arr.shape[0], 0])
            # plt.scatter(res_x, res_y)
            plt.show()

        # if there was an error with ransac:
        if None in [coef_1, intrcpt_1, coef_2, intrcpt_2]:
            plot()
            return None, None
        # calc intersection point
        line1 = HessNormalLine.from_intercept_and_slope(intrcpt_1, coef_1)
        line2 = HessNormalLine.from_intercept_and_slope(intrcpt_2, coef_2)

        return line1.intersect_crossprod(line2)

    @classmethod
    def __preprocess(self, arr: nptyping.NDArray):
        blur = cv2.GaussianBlur(arr, (5, 5), 0)
        _, arr = cv2.threshold(
            np.array(blur, dtype=np.uint16),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        arr = arr.astype(bool)
        return arr

    @classmethod
    def __ransac(self, x: nptyping.NDArray, y: nptyping.NDArray):
        # scikit learn ransac
        ransac = linear_model.RANSACRegressor(
            stop_probability=0.9999,
            residual_threshold=10,
            max_trials=500,
            loss="absolute_error",
        )
        try:
            ransac.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        except Exception as e:
            print(e)
            return None, None, [], []
        # print(f"solution found with {ransac.n_trials_} trials")
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # get coefficient/y-interception
        coef = ransac.estimator_.coef_[0][0]  # the indizes are needed bc they
        intrcpt = ransac.estimator_.intercept_[
            0
        ]  # are packed in useless arrays

        # get residual x, y
        n = np.count_nonzero(outlier_mask)
        res_x, res_y = [None for _ in range(n)], [None for _ in range(n)]
        i = 0
        for x_, y_, outlM in zip(x, y, outlier_mask):
            if outlM == 1:
                res_x[i] = x_
                res_y[i] = y_
                i += 1

        return coef, intrcpt, np.array(res_x), np.array(res_y)

    # this is some fast magic to get the indices from a numpy array:
    # https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where
    @classmethod
    def __compute_M(self, data: nptyping.NDArray):
        cols = np.arange(data.size)
        return csr_matrix(
            (cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size)
        )

    @classmethod
    def __get_indices_sparse(self, data: nptyping.NDArray):
        M = self.__compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]
