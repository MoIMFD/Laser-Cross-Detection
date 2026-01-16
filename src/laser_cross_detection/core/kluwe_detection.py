from __future__ import annotations

from math import ceil, log, sqrt
from typing import TYPE_CHECKING, NamedTuple

import lmfit
import numpy as np
import scipy

if TYPE_CHECKING:
    from numpy.typing import NDArray

from laser_cross_detection.core.detection_abc import DetectionMethodABC
from laser_cross_detection.core.hess_normal_line import ComplexHessLine
from laser_cross_detection.utils.image_utils import Implementations, rotate_image


class AngleSpaceDimension(NamedTuple):
    start: float
    range: float
    steps: int


def angle_diff(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles in degrees,
    accounting for the circular nature of angles.

    Args:
        angle1, angle2: Angles in degrees

    Returns:
        float: Smallest angular difference in degrees (0-90)
    """
    # Convert angles to radians for complex number representation
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)

    # Convert to complex unit vectors
    z1 = np.exp(1j * rad1)
    z2 = np.exp(1j * rad2)

    # Calculate the angular difference using the dot product
    # The dot product of two unit vectors is cos(theta)
    cos_diff = np.real(z1 * np.conj(z2))

    # Clamp to [-1, 1] to handle numerical errors
    cos_diff = np.clip(cos_diff, -1.0, 1.0)

    # Convert back to degrees
    return np.degrees(np.arccos(cos_diff))


class Kluwe(DetectionMethodABC):
    def __init__(
        self,
        start_angle: float = 0,
        angle_range: float = 180,
        angle_steps: int = 180,
        beam_model: lmfit.Model | None = None,
        optimization_method: str = "COBYLA",
        profile_fit_method: str = "leastsq",
    ) -> None:
        """Method to detect the intersection of two light beams in 2d images.

        Args:
            start_angle (float, optional): start of the interval considered
                for angles. Defaults to 0.
            angle_range (float, optional): range of the interval considered
                for angles. Defaults to 180.
            angle_steps (int, optional): number of steps between start_angle
                and start_angle + angle_range. Defaults to 180.
            beam_model (lmfit.Model, optional): lmfit model of the beam shape.
                Defaults to lmfit.models.GaussianModel().
            interpolation_order (int, optional): order of interpolation scheme
                used for rotating. Defaults to 3.
        """
        self.angle_space_dim = AngleSpaceDimension(
            start_angle, angle_range, angle_steps
        )
        self.angle_step_size = angle_range / angle_steps
        self.beam_model = beam_model or lmfit.models.GaussianModel()
        self.optimization_method = optimization_method
        self.profile_fit_method = profile_fit_method

    def calc_lines(self, image: NDArray) -> tuple[ComplexHessLine, ComplexHessLine]:
        """Derive the lines from a laser cross image.

        Args:
            image: image to process

        Returns:
            tuple of two line objects

        """
        image_center = np.array(image.shape[::-1]) / 2

        angle_0, angle_1 = self.calc_angles(arr=image, method=self.optimization_method)

        # calculate radius
        radius_0 = self.calc_radius(image, angle_0)
        radius_1 = self.calc_radius(image, angle_1)

        return (
            ComplexHessLine.from_degrees(radius_0, angle_0, center=image_center),
            ComplexHessLine.from_degrees(radius_1, angle_1, center=image_center),
        )

    def __call__(
        self,
        image: NDArray,
        *args,
        **kwargs,
    ) -> NDArray:
        """Calculates the point of intersection of two beams in images
        containing both beams.

        Args:
            image (NDArray): image to process

        Returns:
            NDArray: point of intersection
        """
        line0, line1 = self.calc_lines(image)
        intersection = line0.intersect(line1)
        return intersection

    def calc_angles(self, arr: NDArray, method="Brent") -> tuple[float, float]:
        """Calculates the angles of two beams present in a image.

        Args:
            arr (NDArray): image containing the beams

        Returns:
            Tuple[float, float]: angles of the beams in degrees
        """
        # estimate angles
        guessed_angles, properties = self.guess_angles(arr)
        sort_by_prominence = np.argsort(properties["prominences"])[::-1]
        guess = guessed_angles[sort_by_prominence][:2]
        if method in [
            "Nelder-Mead",
            "Powell",
            "COBYLA",
            "COBYQA",
        ]:
            # optimize estimation by minimizing cost function
            res_0 = scipy.optimize.minimize(
                fun=optimization_loss_function,
                x0=(guess[0],),
                # note: comma it is needed since scipy expects a one element tuple
                bounds=[
                    (
                        guess[0] - self.angle_step_size / 2,
                        guess[0] + self.angle_step_size / 2,
                    )
                ],
                args=arr,
                method=method,
            )
            angle_0 = res_0.x[0]

            res_1 = scipy.optimize.minimize(
                fun=optimization_loss_function,
                x0=(guess[1],),
                bounds=[
                    (
                        guess[1] - self.angle_step_size / 2,
                        guess[1] + self.angle_step_size / 2,
                    )
                ],
                args=arr,
                method=method,
            )
            angle_1 = res_1.x[0]
            return angle_0, angle_1

        elif method in ["Brent", "Golden"]:
            res_0 = scipy.optimize.minimize_scalar(
                fun=lambda angle: optimization_loss_function_scalar(angle, arr),
                bracket=[
                    guess[0] - self.angle_step_size / 2,
                    guess[0],
                    guess[0] + self.angle_step_size / 2,
                ],
                method=method,
            )
            angle_0 = res_0.x

            res_1 = scipy.optimize.minimize_scalar(
                fun=lambda angle: optimization_loss_function_scalar(angle, arr),
                bracket=[
                    guess[1] - self.angle_step_size / 2,
                    guess[1],
                    guess[1] + self.angle_step_size / 2,
                ],
                method=method,
            )
            angle_1 = res_1.x
            return angle_0, angle_1

        else:
            raise NotImplementedError()

    def calc_radius(self, arr: NDArray, angle: float) -> float:
        """Calculates the radius (distance from the center) of a beam with
        known angle.

        Args:
            arr (NDArray): image of the beam
            angle (float): angle of the beam in degrees

        Returns:
            float: radius of the beam, e. g. the distance from the center
        """
        intensity_profile = self.collapse_arr(arr, angle)
        projection_axis = np.arange(intensity_profile.size) - intensity_profile.size / 2
        maximum_index = np.argmax(intensity_profile)

        # estimate peak width
        background = np.percentile(intensity_profile, 10)
        leveled_intensity_profile = intensity_profile - background

        # Calculate peak widths at the relative height
        peak_widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
            leveled_intensity_profile,
            np.array([maximum_index]),
            rel_height=0.5,
        )

        width_at_10percent = peak_widths[0] * sqrt(log(1 / 0.1) / log(2))
        beam_width = max(ceil(width_at_10percent), 7)
        if beam_width % 2 == 0:
            beam_width += 1

        half_beam_width = beam_width // 2

        pad_width = max(half_beam_width, 20)
        padded_intensity_profile = np.pad(
            leveled_intensity_profile, pad_width, mode="reflect"
        )

        padded_maximum_index = maximum_index + pad_width

        padded_fitting_window = padded_maximum_index + np.arange(
            -half_beam_width, half_beam_width + 1
        )  # +1 because else last idx not included

        padded_projection_axis = np.arange(padded_intensity_profile.size) - (
            pad_width + projection_axis.size / 2
        )

        initial_params = self.beam_model.guess(
            data=padded_intensity_profile[padded_fitting_window],
            x=padded_projection_axis[padded_fitting_window],
        )
        fitting_result = self.beam_model.fit(
            data=padded_intensity_profile[padded_fitting_window],
            params=initial_params,
            x=padded_projection_axis[padded_fitting_window],
            method=self.profile_fit_method,
        )
        # the center of the beam is the radius and it is the center of the
        # Gaussian distribution
        peak_position = fitting_result.params["center"].value
        return peak_position

    def collapse_arr(self, arr: NDArray, angle: float = 0.0) -> NDArray:
        """Rotates an image by the specified amount and reduces the 2d image
        to a 1d vector by averaging columns.

        Args:
            arr (NDArray): 2d image to process
            angle (float, optional): angle to rotate in degrees.
                Defaults to 0.0.

        Returns:
            NDArray: 1d averaged vector
        """

        if angle == 0:
            col = np.mean(arr, axis=0).flatten()
        else:
            col = np.mean(
                rotate_image(
                    image=arr, angle=angle, impl=Implementations.OPENCV, order=1
                ),
                axis=0,
            ).flatten()
        return col

    def calc_angle_space(
        self,
        arr: NDArray,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Performs the collapse_arr operation on a linear space of angles.

        Args:
            arr (NDArray): image to process

        Returns:
            NDArray: accumulated result of the collapse_arr operation
                (referred to as "angle space")
        """
        angles = np.linspace(
            self.angle_space_dim.start,
            self.angle_space_dim.start + self.angle_space_dim.range,
            self.angle_space_dim.steps,
            endpoint=False,
        )

        from concurrent.futures import ThreadPoolExecutor

        def rotate_single(args):
            return self.collapse_arr(arr=args[0], angle=args[1])

        with ThreadPoolExecutor() as executer:
            result = list(
                executer.map(rotate_single, [(arr, angle) for angle in angles])
            )

        return angles, np.array(result)

    def guess_angles(
        self,
        arr: NDArray,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Estimating the orientation of two beams in an image to provide a
        good initial guess used as starting point for optimization. Image is
        rotated in discrete steps. Angles where the beams best align with the
        first image axis are returned.

        Args:
            arr (NDArray): image to process

        Returns:
            tuple[float, float]: estimation of the angles of the two beams in
                degrees
        """
        angles, accumulator = self.calc_angle_space(arr)
        accumulator = accumulator**2
        projection = np.max(accumulator, axis=1)
        projection_wrapped = np.tile(projection, 3)

        peaks, properties = scipy.signal.find_peaks(projection_wrapped, prominence=0.0)

        mask = (peaks >= projection.size) & (peaks < 2 * projection.size)
        peaks = peaks[mask] - projection.size
        for key in properties:
            properties[key] = properties[key][mask]

        return angles[peaks], properties


def optimization_loss_function(angle: NDArray[np.floating], im: NDArray) -> float:
    """Cost function used for accurate estimation of the alignment of a
    straight beam in an image with the first axis of the image. Suitable for
    scipy.optimize.minimize.

    Args:
        angle (float): angle to rotate the image in degree
        im (NDArray): image to check

    Returns:
        float: score of the alignment
    """
    neg_maximum = -np.max(
        np.mean(
            rotate_image(im, angle=angle[0], impl=Implementations.OPENCV, order=1),
            axis=0,
        )
    )
    return neg_maximum.item()


def optimization_loss_function_scalar(angle: float, im: NDArray) -> float:
    """Cost function used for accurate estimation of the alignment of a
    straight beam in an image with the first axis of the image. Suitable for
    scipy.optimize.minimize_scalar.

    Args:
        angle (float): angle to rotate the image in degree
        im (NDArray): image to check

    Returns:
        float: score of the alignment
    """
    neg_maximum = -np.max(
        np.mean(
            rotate_image(im, angle=angle, impl=Implementations.OPENCV, order=1),
            axis=0,
        )
    )
    return neg_maximum
