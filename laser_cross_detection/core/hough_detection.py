import numpy as np
import numpy.typing as nptyping
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import KMeans

from .detection_abc import DetectionMethodABC
from .hess_normal_line import ComplexHessLine

PI = np.pi


def average_angles(angles: nptyping.NDArray) -> float:
    """Calculates the average angle from a list of angles. To handle the
    overflows, negative angles and other unwanted behavior, the angles
    are converted to complex numbers, averaged and transformed back.

    Args:
        angles (nptyping.NDArray): list of angles in radians

    Returns:
        float: average angle in radians
    """
    angles = np.array(angles) * 2
    z = np.exp(1j * angles)
    z_mean = np.mean(z)
    return np.angle(z_mean) / 2


class Hough(DetectionMethodABC):
    """Laser Cross Detection Method based on Probabilistic Hough Transform
    Algorithm. Implementation by Robert Hardege. Details provide in
    https://doi.org/10.1007/s00348-023-03729-1

    Minor changes to fit in the new frame by Kluwe

    Args:
        DetectionMethodABC (ABC): Hough
    """

    def __call__(
        self,
        arr: nptyping.NDArray,
        seed: int = 0,
        return_lines: bool = False,
        *args,
        **kwargs,
    ) -> nptyping.NDArray:
        """Takes an image of two intersecting beams and returns the estimated
        point of intersection of the beams.

        Args:
            arr (nptyping.NDArray): image to process

        Returns:
            nptyping.NDArray: point of intersection (2d)
        """
        arr = Hough.binarize_image(arr=arr)
        image_center = np.array(arr.shape[::-1]) / 2
        point_pairs = probabilistic_hough_line(
            arr,
            threshold=100,
            theta=np.linspace(0, PI, 360, endpoint=False),
            rng=seed,
        )

        hess_lines = []
        for line in point_pairs:
            p0, p1 = line
            hess_lines.append(
                ComplexHessLine.from_two_points(p1, p0, center=image_center)
            )

        angles = [line.angle for line in hess_lines]

        labels, _ = group_angles(angles)

        lines_1 = [line for line, label in zip(hess_lines, labels) if label == 0]
        lines_2 = [line for line, label in zip(hess_lines, labels) if label == 1]

        line1 = ComplexHessLine.from_averaged_lines(lines_1)
        line2 = ComplexHessLine.from_averaged_lines(lines_2)

        if return_lines:
            return (
                line1.intersect(line2),
                line1,
                line2,
            )
        else:
            return line1.intersect_crossprod(line2)


def group_angles(angles, n_groups=2):
    angles = 2 * np.array(angles)
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    kmeans = KMeans(n_clusters=n_groups, random_state=0)
    labels = kmeans.fit_predict(points)

    centers = kmeans.cluster_centers_
    center_angles = np.arctan2(centers[:, 1], centers[:, 0]) / 2

    return labels, center_angles
