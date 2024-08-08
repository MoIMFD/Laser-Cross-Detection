import numpy as np
import numpy.typing as nptyping
from skimage.transform import probabilistic_hough_line

from .hess_normal_line import HessNormalLine
from .detection_abc import DetectionMethodABC

PI = np.pi


def average_angles(angles: nptyping.NDArray) -> float:
    """Calculates the average angle from a list of angles. To handle the
    overflows, negative angles and other unwanted behavior, the angles
    are converted to complex numbers, averaged and transformed back.

    Args:
        angles (nptyping.NDArray): list of angles in randians

    Returns:
        float: average angle in radians
    """
    z = np.exp(1j * np.array(angles))
    z_mean = np.mean(z)
    return np.angle(z_mean)


class Hough(DetectionMethodABC):
    """Laser Cross Detection Method based on Probabilistic Hough Transform
    Algorithm. Implementation by Robert Hardege. Details provide in
    https://doi.org/10.1007/s00348-023-03729-1

    Minor changes to fit in the new frame by Kluwe

    Args:
        DetectionMethodABC (ABC): Hough
    """

    def __call__(
        self, arr: nptyping.NDArray, *args, **kwargs
    ) -> nptyping.NDArray:
        """Takes an image of two intersecting beams and returns the estimated
        point of intersection of the beams.

        Args:
            arr (nptyping.NDArray): image to process

        Returns:
            nptyping.NDArray: point of intersection (2d)
        """
        arr = Hough.binarize_image(arr=arr)
        lines = probabilistic_hough_line(
            arr, threshold=100, theta=np.linspace(0, PI, 180)
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
