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


def average_angles(angles):
    z = np.exp(1j * np.array(angles))
    z_mean = np.mean(z)
    return np.angle(z_mean)


class Hough(DetectionMethodABC):
    def __call__(self, arr, *args, **kwargs):
        arr = self.__preprocess(arr=arr)
        lines = probabilistic_hough_line(
            arr, threshold=100, theta=np.linspace(0, np.pi, 180)
        )

        hess_lines = []
        for line in lines:
            p0, p1 = line
            hess_lines.append(HessNormalLine.from_two_points(p1, p0))

        angles = [line.angle for line in hess_lines]
        threshold = (max(angles) + min(angles)) / 2

        lines_1, lines_2 = [], []
        for line in hess_lines:
            if line.angle < threshold:
                lines_1.append(line)
            else:
                lines_2.append(line)

        rho1 = np.mean([line.distance for line in lines_1])
        theta1 = average_angles([[line.angle for line in lines_1]])

        rho2 = np.mean([line.distance for line in lines_2])
        theta2 = average_angles([[line.angle for line in lines_2]])

        line1 = HessNormalLine(rho1, theta1)
        line2 = HessNormalLine(rho2, theta2)

        return line1.intersect_crossprod(line2)

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
