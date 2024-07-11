import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import (
    hough_line,
    probabilistic_hough_line,
    hough_line_peaks,
)


from .hess_normal_line import HessNormalLine
from .detection_abc import DetectionMethodABC
from ..utils import image_utils

PI = np.pi


class Hough(DetectionMethodABC):
    def __call__(self, arr, *args, **kwargs):
        arr_copy = np.copy(arr)
        arr = self.__preprocess(arr=arr)
        lines = probabilistic_hough_line(arr, threshold=100)
        lines_array = np.array(lines)

        angles = []
        for line in lines:
            p0, p1 = line
            d_x, d_y = p1[0] - p0[0], p1[1] - p0[1]
            angles.append(np.arctan2(d_y, d_x))
        angles = np.array(angles)
        # angles, close to pi, get pi subtracted to be close to 0
        # -> that is a problem when there are horizontal lines, as these can either be 0 or pi
        angles[
            np.logical_and(
                angles > PI - PI / 180 * 5, angles < PI + PI / 180 * 5
            )
        ] -= PI
        angles = np.abs(angles)
        div_value = (np.min(angles) + np.max(angles)) / 2

        lines_1 = lines_array[angles < div_value]
        lines_2 = lines_array[angles >= div_value]

        intrsctn_pnts = []
        for line_1 in lines_1:
            for line_2 in lines_2:
                x1, y1 = line_1[0]
                x2, y2 = line_1[1]
                x3, y3 = line_2[0]
                x4, y4 = line_2[1]
                line1 = HessNormalLine.from_two_points((x1, y1), (x2, y2))
                line2 = HessNormalLine.from_two_points((x3, y3), (x4, y4))
                intrsctn_pnts.append(line1.intersect_crossprod(line2))

        intrsctn_pnts = np.array(intrsctn_pnts, dtype=np.float64)

        # self.__plot(arr=arr, lines=lines, intersection_point=np.mean(intrsctn_pnts, axis=0))

        hor = np.mean(intrsctn_pnts, axis=0)[0]
        ver = np.mean(intrsctn_pnts, axis=0)[1]
        return np.array([hor, ver])

    @classmethod
    def __preprocess(self, arr):
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

    # @classmethod
    def __plot(
        self,
        arr,
        lines,
        intersection_point=None,
        figsize=(8, 5),
        colormap="viridis",
    ):
        fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

        ax.imshow(arr, cmap=colormap)
        for line in lines:
            p0, p1 = line
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
        if intersection_point is not None:
            ax.scatter(
                intersection_point[0],
                intersection_point[1],
                c="black",
                zorder=999,
            )
        ax.set_xlim((0, arr.shape[1]))
        ax.set_ylim((arr.shape[0], 0))
        plt.show()
