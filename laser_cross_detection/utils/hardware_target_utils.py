import numpy as np
import pandas as pd
import lmfit
import skimage as ski
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar


from math import atan2, cos, sin, pi as PI
from functools import cached_property
from collections import deque
from itertools import combinations, product


def rotate_2d_vector(vector, angle):
    x, y = vector
    r = (x**2 + y**2) ** 0.5
    phi = np.atan2(y, x)
    phi += angle
    x = r * cos(phi)
    y = r * sin(phi)
    return np.array([x, y])


def fit_ellipse_direct(points):
    """
    Fit an ellipse to a set of points using the direct least squares method.
    Based on the paper: "Direct Least Squares Fitting of Ellipses" by Fitzgibbon et al.
    """
    # Extract coordinates
    x, y = points[:, 1], points[:, 0]

    # Build design matrix
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])

    # Build constraint matrix
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    # Solve eigenvalue problem
    from scipy import linalg

    try:
        eigenvalues, eigenvectors = linalg.eig(
            np.dot(np.linalg.inv(np.dot(D.T, D)), C)
        )
        # Find the only positive eigenvalue
        idx = np.where(eigenvalues > 0)[0]
        if len(idx) == 0:  # No positive eigenvalue found
            return None
        a = eigenvectors[:, idx[0]]
    except np.linalg.LinAlgError:
        # If we have a singular matrix, revert to basic centroid
        return np.mean(points, axis=0)

    # Extract ellipse parameters
    A, B, C, D, E, F = a

    # Calculate center of the ellipse
    x0 = (B * E - 2 * C * D) / (4 * A * C - B * B)
    y0 = (B * D - 2 * A * E) / (4 * A * C - B * B)

    return (y0, x0)  # Return in (y, x) format for images


def fit_2d_gaussian(point):
    model = lmfit.models.Gaussian2dModel()
    intensity = point.image_intensity
    y = np.arange(point.bbox[0], point.bbox[2])
    x = np.arange(point.bbox[1], point.bbox[3])
    xx, yy = np.meshgrid(x, y)
    params = model.guess(intensity.ravel(), xx.ravel(), yy.ravel())
    params["centerx"].value = point.centroid[1]
    params["centery"].value = point.centroid[0]
    result = model.fit(
        intensity.ravel(), x=xx.ravel(), y=yy.ravel(), params=params
    )
    return result.best_values["centery"], result.best_values["centerx"]


class DistanceOutlierDetector:
    def __init__(self, window_size=5, threshold_factor=2.5):
        """
        Initialize distance-based outlier detector.

        Parameters:
        - window_size: Number of recent distances to consider
        - contamination: Expected proportion of outliers
        - threshold_factor: Factor to multiply std deviation for simple detection
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.threshold_factor = threshold_factor

    def is_outlier(self, value):
        self.values.append(value)

        if len(self.values) < 3:
            return False

        mean_dist = np.mean(
            list(self.values)[:-1]
        )  # Mean of previous distances
        std_dist = np.std(
            list(self.values)[:-1]
        )  # Std dev of previous distances

        # If std is very small, use a minimum value to avoid division by zero issues
        std_dist = max(std_dist, 0.001 * mean_dist)

        # Calculate z-score
        z_score = abs(value - mean_dist) / std_dist

        # Return True if z-score exceeds threshold
        is_outlier = z_score > self.threshold_factor

        if is_outlier:
            self.values.pop()
        return is_outlier


class HardwareCalibrationTarget:
    def __init__(self, target_image):
        self.target_image = target_image
        self.target_regions = sorted(
            get_target_markers(self.target_image),
            key=lambda x: x.area_filled / x.area_bbox,
        )
        self.calibration_params = {
            "initial_x_axis": None,
            "initial_y_axis": None,
            "initial_basis": None,
            "x_axis": None,
            "y_axis": None,
            "basis": None,
            "grid1": None,
            "grid2": None,
        }

    @cached_property
    def marker_positions_gaussian(self):
        return np.array([fit_2d_gaussian(point) for point in self.points])

    @property
    def points(self):
        return self.target_regions[1:-1]

    @property
    def square(self):
        return self.target_regions[-1]

    @property
    def triangle(self):
        return self.target_regions[0]

    @property
    def aligned_points(self):
        assert (
            self.calibration_params["basis"] is not None
        ), "Perform axis alignement first"
        return np.dot(
            self.marker_positions_gaussian - self.triangle.centroid,
            self.calibration_params["basis"],
        )

    def _draw_region(self, regions, image):
        if not isinstance(regions, (list, tuple)):
            regions = list(regions)
        for region in regions:
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            image[rows, cols] = region.label
        return image

    @property
    def label_image(self):
        image = np.zeros_like(self.target_image)
        self._draw_region(self.points, image)
        return image

    @property
    def target_overlay(self):
        label_image = ski.color.label2rgb(
            self.label_image, image=self.target_image, bg_label=0
        )
        rows, cols = self.triangle.coords[:, 0], self.triangle.coords[:, 1]
        label_image[rows, cols, :] = (1, 0, 0)
        rows, cols = self.square.coords[:, 0], self.square.coords[:, 1]
        label_image[rows, cols, :] = (0, 0, 1)
        return label_image

    def initial_axis_alignment(self):
        axis = np.array(self.square.centroid) - self.triangle.centroid
        x_axis = rotate_2d_vector(axis, 0.3 * np.pi)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = rotate_2d_vector(axis, -0.2 * np.pi)
        y_axis = y_axis / np.linalg.norm(y_axis)

        self.calibration_params["initial_x_axis"] = x_axis
        self.calibration_params["initial_y_axis"] = y_axis
        self.calibration_params["initial_basis"] = np.vstack([x_axis, y_axis])

        return self

    def refine_axis_alignement(self):
        if not self.calibration_params["initial_basis"]:
            self.initial_axis_alignment()
        points = self.marker_positions_gaussian.copy()

        def cost_function(angle):
            new_x_axis = rotate_2d_vector(
                self.calibration_params["initial_x_axis"], angle
            )
            new_y_axis = rotate_2d_vector(
                self.calibration_params["initial_y_axis"], angle
            )

            x_proj = np.dot(points, new_x_axis)
            y_proj = np.dot(points, new_y_axis)

            return np.max(np.abs(x_proj)) + np.max(np.abs(y_proj))

        result = minimize_scalar(
            cost_function, 0.0, bounds=((-np.pi / 8, np.pi / 8))
        )

        assert result.success, f"optimization did not converged: {result!s}"
        self.calibration_params["x_axis"] = rotate_2d_vector(
            self.calibration_params["initial_x_axis"], result.x
        )
        self.calibration_params["y_axis"] = rotate_2d_vector(
            self.calibration_params["initial_y_axis"], result.x
        )
        self.calibration_params["basis"] = np.vstack(
            [
                self.calibration_params["x_axis"],
                self.calibration_params["y_axis"],
            ]
        )

        return self

    def assign_points_to_lines(self, weight_x=12.0, weight_y=0.1):
        detector = DistanceOutlierDetector(window_size=10, threshold_factor=7)
        used_index = []
        lines = []
        for _ in range(25):
            points = self.aligned_points.copy()
            points[used_index, :] = np.array((np.inf, np.inf))
            points_x = points[:, 0]
            points_y = points[:, 1]
            start_index, *_ = np.lexsort((points[:, 0], points[:, 1]))
            current_index = start_index
            line = [start_index]
            for _ in range(11):
                dists_x = points_x - points_x[current_index]
                dists_y = points_y - points_y[current_index]

                dists = np.sqrt(weight_x * dists_x**2 + weight_y * dists_y**2)
                dists[line] = np.inf

                current_index = np.argmin(dists)

                if detector.is_outlier(dists[current_index]):
                    break

                line.append(current_index)

            lines.append(line)
            used_index.extend(line)

            if len(used_index) >= len(self.aligned_points):
                break

        lines = sorted(lines, key=lambda x: np.mean(self.aligned_points[x, 0]))
        lines1 = lines[::2]
        lines2 = lines[1::2]
        return lines1, lines2

    def build_grids(self):
        idx = list(range(11))
        grid1 = {i: None for i in list(product(idx, idx))}
        idx = list(range(10))
        grid2 = {i: None for i in list(product(idx, idx))}

        lines1, lines2 = self.assign_points_to_lines()

        for i, line in enumerate(lines1):
            for j, point_idx in enumerate(
                sorted(line, key=lambda x: self.aligned_points[x, 1])
            ):
                if i == 1:
                    grid1[(i, j + 1)] = point_idx
                else:
                    grid1[(i, j)] = point_idx

        for i, line in enumerate(lines2):
            for j, point_idx in enumerate(
                sorted(line, key=lambda x: self.aligned_points[x, 1])
            ):
                grid2[(i, j)] = point_idx

        self.calibration_params["grid1"] = grid1
        self.calibration_params["grid2"] = grid2

        return self


def get_target_markers(image, num_largest_elements=300):
    # Initial thresholding and morphological operations
    smoothed = ski.filters.gaussian(image, sigma=1.0)
    threshold = ski.filters.threshold_otsu(smoothed)
    bw = smoothed > threshold
    cleared = ski.segmentation.clear_border(bw)

    # Label the image
    label_image = ski.measure.label(cleared)

    sorted_regions = sorted(
        ski.measure.regionprops(label_image, intensity_image=image),
        key=lambda x: x.area,
    )[::-1][:num_largest_elements]

    # Apply filtering criteria to the labels
    valid_regions = list(
        filter(
            lambda x: (x.eccentricity < 0.6) & (x.area > 1000),
            sorted_regions,
        )
    )

    return valid_regions


def process_square(square, pad_width=20):
    padded_image = np.pad(
        square.intensity_image, (pad_width, pad_width), constant_values=0
    )
    coords = ski.feature.corner_peaks(
        ski.feature.corner_harris(padded_image),
        min_distance=pad_width - 1,
        threshold_rel=0.22,
    )
    assert coords.shape[0] == 4, f"{coords.shape}"
    coords_subpix = ski.feature.corner_subpix(
        padded_image, coords, window_size=15
    )
    indices = np.lexsort((coords_subpix[:, 0], coords_subpix[:, 1]))
    return (
        coords_subpix[indices, 1] - pad_width,
        coords_subpix[indices, 0] - pad_width,
    )


def process_triangle(triangle, pad_width=15):
    padded_image = np.pad(
        triangle.intensity_image, (pad_width, pad_width), constant_values=0
    )
    coords = ski.feature.corner_peaks(
        ski.feature.corner_harris(padded_image),
        min_distance=pad_width,
        threshold_rel=0.22,
    )
    assert coords.shape[0] == 3, f"{coords.shape}"
    coords_subpix = ski.feature.corner_subpix(
        padded_image, coords, window_size=25
    )
    indices = np.lexsort((coords_subpix[:, 1], coords_subpix[:, 0]))
    return (
        coords_subpix[indices, 1] - pad_width,
        coords_subpix[indices, 0] - pad_width,
    )
