import numpy as np
import numpy.typing as nptyping
import lmfit
import scipy
import skimage as ski

from collections import namedtuple
from typing import Union, Tuple

from .hess_normal_line import HessNormalLine
from .detection_abc import DetectionMethodABC
from ..utils import image_utils

from math import ceil, sqrt, log

AngleSpaceDimension = namedtuple("AngleSpaceDimension", "start range steps")


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
        beam_model: lmfit.Model = lmfit.models.GaussianModel(),
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
        self.beam_model = beam_model
        self.optimization_method = optimization_method
        self.profile_fit_method = profile_fit_method

    def __call__(
        self,
        image: nptyping.NDArray,
        return_beam_params: bool = False,
        *args,
        **kwds,
    ) -> nptyping.NDArray:
        """Calculates the point of intersection of two beams in images
        containing both beams.

        Args:
            image (nptyping.NDArray): image to process

        Returns:
            nptyping.NDArray: point of intersection
        """
        image_center = np.array(image.shape[::-1]) / 2

        beam_params = dict(beam0=dict(), beam1=dict())
        angle_0, angle_1 = self.calc_angles(arr=image, method=self.optimization_method)
        beam_params["beam0"]["angle"] = angle_0
        beam_params["beam1"]["angle"] = angle_1

        # calculate radius
        radius_0 = self.calc_radius(image, angle_0)
        radius_1 = self.calc_radius(image, angle_1)
        beam_params["beam0"]["radius"] = radius_0
        beam_params["beam1"]["radius"] = radius_1

        beam0 = HessNormalLine.from_degrees(radius_0, angle_0, center=image_center)
        beam1 = HessNormalLine.from_degrees(radius_1, angle_1, center=image_center)

        intersection = beam0.intersect_crossprod(beam1)
        if return_beam_params:
            return intersection, beam_params
        else:
            return intersection

    def calc_angles(self, arr: nptyping.NDArray, method="Brent") -> Tuple[float, float]:
        """Calculates the angles of two beams present in a image.

        Args:
            arr (nptyping.NDArray): image containing the beams

        Returns:
            Tuple[float, float]: angles of the beams in degrees
        """
        # estimate angles
        guess = self.estimate_global_maxima(arr)
        if method in [
            "Nelder-Mead",
            "Powell",
            "COBYLA",
            "COBYQA",
        ]:
            # optimize estimation by minimizing coast function
            res_0 = scipy.optimize.minimize(
                fun=optimization_loss_function,
                x0=(guess[0],),
                bounds=(
                    (
                        guess[0] - self.angle_step_size / 2,
                        guess[0] + self.angle_step_size / 2,
                    ),  # note the comma it is needed since scipy expects a one element tuple!
                ),
                args=arr,
                method=method,
            )
            angle_0 = res_0.x[0]

            res_1 = scipy.optimize.minimize(
                fun=optimization_loss_function,
                x0=(guess[1]),
                bounds=(
                    (
                        guess[1] - self.angle_step_size / 2,
                        guess[1] + self.angle_step_size / 2,
                    ),  # note the comma it is needed since scipy expects a one element tuple!
                ),
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

    def calc_radius(self, arr: nptyping.NDArray, angle: float) -> float:
        """Calculates the radius (distance from the center) of a beam with
        known angle.

        Args:
            arr (nptyping.NDArray): image of the beam
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
            leveled_intensity_profile, np.array([maximum_index]), rel_height=0.5
        )

        width_at_10percent = peak_widths[0] * sqrt(log(1 / 0.1) / log(2))
        beam_width = ceil(width_at_10percent)
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

    def collapse_arr(
        self, arr: nptyping.NDArray, angle: float = 0.0
    ) -> nptyping.NDArray:
        """Rotates an image by the specified amount and reduces the 2d image
        to a 1d vector by averaging aling the first axis.

        Args:
            arr (nptyping.NDArray): 2d image to process
            angle (float, optional): angle to rotate in degrees.
                Defaults to 0.0.

        Returns:
            nptyping.NDArray: 1d averaged vector
        """

        if angle == 0:
            col = np.mean(arr, axis=0).flatten()
        else:
            col = np.mean(
                image_utils.rotate_image(image=arr, angle=angle, impl="cv2", order=1),
                axis=0,
            ).flatten()
        return col

    def calc_angle_space(
        self, arr: nptyping.NDArray, offset: float = 0
    ) -> nptyping.NDArray:
        """Performs the collapse_arr operation on a linear space of angles.
        When the searched beams align with the start/end point of the range of
        angles, a single peak may gets splitted and creates two peaks. For this
        case the offset parameter exists which offsets the angle range by the
        specified amount.

        Args:
            arr (nptyping.NDArray): image to process
            offset (float, optional): offset to the angle range in degrees.
                Defaults to 0.

        Returns:
            nptyping.NDArray: accumulated result of the collapse_arr operation
                (referred to as "angle space")
        """
        angles = np.linspace(
            self.angle_space_dim.start + offset,
            self.angle_space_dim.start + offset + self.angle_space_dim.range,
            self.angle_space_dim.steps,
            endpoint=False,
        )
        return angles, np.array([self.collapse_arr(arr, angle) for angle in angles])

    def estimate_global_maxima(
        self, arr: nptyping.NDArray, wrap_length=5
    ) -> Tuple[float, float]:
        """Estimating the orientation of two beams in an image to provide a
        good initial guess used as starting point for optimization. Image is
        rotated in discrete steps. Angles where the beams best align with the
        first image axis are returned.

        Args:
            arr (nptyping.NDArray): image to process
            min_angle (Union[float  |  None], optional): minimum angle between
                the beams. Defaults to None (2 angle steps).

        Returns:
            Tuple[float, float]: estimation of the angles of the two beams in
                degrees
        """

        angles, angle_space = self.calc_angle_space(arr)
        projection = np.max(angle_space, axis=1)
        wrapped_projection = np.pad(projection, wrap_length, mode="wrap")

        signal_min = np.min(wrapped_projection)
        signal_max = np.max(wrapped_projection)
        signal_mean = np.mean(wrapped_projection)
        signal_std = np.std(wrapped_projection)

        peaks, properties = scipy.signal.find_peaks(
            wrapped_projection,
            height=(signal_mean + 1.0 * signal_std, None),
            distance=1,
            prominence=2.0 * signal_std,
        )

        if len(peaks) < 2:
            peaks, properties = scipy.signal.find_peaks(
                wrapped_projection,
                height=(signal_mean + 0.5 * signal_std, None),
                distance=1,
                prominence=1.0 * signal_std,
            )
        assert len(peaks) > 1, f"Not enough peaks caught ({len(peaks)})"
        adjusted_peaks = (peaks - wrap_length) % len(projection)
        sorted_indices = np.argsort(-properties["prominences"])
        ranked_peaks = adjusted_peaks[sorted_indices]

        # check if peaks 'wrap around'
        unique_peaks = []
        for peak in ranked_peaks:
            peak_angle = angles[peak]
            # Check if this peak is sufficiently different from already selected peaks
            # Using our complex number-based angle_diff function
            if all(
                angle_diff(peak_angle, angle) >= self.angle_step_size
                for angle in angles[unique_peaks]
            ):
                unique_peaks.append(peak)
                if len(unique_peaks) >= 2:
                    break

        # print(angles[unique_peaks[0]], angles[unique_peaks[1]])
        return angles[unique_peaks[0]], angles[unique_peaks[1]]


def optimization_loss_function(
    angle: nptyping.NDArray[float], im: nptyping.NDArray
) -> float:
    """Coast function used for accurate estimation of the alignment of a
    straight beam in an image with the first axis of the image. Suitable for
    scipy.optimize.minimize.

    Args:
        angle (float): angle to rotate the image in degree
        im (nptyping.NDArray): image to check

    Returns:
        float: score of the alignment
    """
    neg_maximum = -np.max(
        np.mean(
            image_utils.rotate_image(im, angle=angle[0], impl="cv2", order=1),
            axis=0,
        )
    )
    return neg_maximum


def optimization_loss_function_scalar(angle: float, im: nptyping.NDArray) -> float:
    """Coast function used for accurate estimation of the alignment of a
    straight beam in an image with the first axis of the image. Suitable for
    scipy.optimize.minimize_scalar.

    Args:
        angle (float): angle to rotate the image in degree
        im (nptyping.NDArray): image to check

    Returns:
        float: score of the alignment
    """
    neg_maximum = -np.max(
        np.mean(
            image_utils.rotate_image(im, angle=angle, impl="cv2", order=1),
            axis=0,
        )
    )
    return neg_maximum
