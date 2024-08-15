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

AngleSpaceDimension = namedtuple("AngleSpaceDimension", "start range steps")


class Kluwe(DetectionMethodABC):
    def __init__(
        self,
        cvt_uint8: bool = False,
        beam_width: int = 20,
        start_angle: float = 0,
        angle_range: float = 180,
        angle_steps: int = 180,
        beam_model: lmfit.Model = lmfit.models.GaussianModel(),
        interpolation_order: int = 3,
    ) -> None:
        """Method to detect the intersection of two light beams in 2d images.

        Args:
            cvt_uint8 (bool, optional): If the image should be converted to
                uint8. Defaults to False.
            beam_width (int, optional): Hint of the expected beam width.
                Defaults to 20.
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
        self.cvt_uint8 = cvt_uint8
        self.angle_space_dim = AngleSpaceDimension(
            start_angle, angle_range, angle_steps
        )
        self.beam_model = beam_model
        self.beam_width = beam_width
        self.interpolation_order = interpolation_order

    @property
    def half_beam_width(self) -> float:
        """Half of the user guessed beam width.

        Returns:
            float: half of the user guessed beam width
        """
        return self.beam_width // 2

    def __call__(
        self, image: nptyping.NDArray, *args, **kwds
    ) -> nptyping.NDArray:
        """Calculates the point of intersection of two beams in images
        containing both beams.

        Args:
            image (nptyping.NDArray): image to process

        Returns:
            nptyping.NDArray: point of intersection
        """
        if self.cvt_uint8:
            image = ski.util.img_as_ubyte(image)

        image_center = np.array(image.shape[::-1]) / 2

        angle_0, angle_1 = self.calc_angles(arr=image)

        # calculate radius
        radius_0 = self.calc_radius(image, angle_0)
        radius_1 = self.calc_radius(image, angle_1)

        beam0 = HessNormalLine.from_degrees(
            radius_0, angle_0, center=image_center
        )
        beam1 = HessNormalLine.from_degrees(
            radius_1, angle_1, center=image_center
        )

        intersection = beam0.intersect_crossprod(beam1)

        return intersection

    def calc_angles(self, arr: nptyping.NDArray) -> Tuple[float, float]:
        """Calculates the angles of two beams present in a image.

        Args:
            arr (nptyping.NDArray): image containing the beams

        Returns:
            Tuple[float, float]: angles of the beams in degrees
        """
        # estimate angles
        guess = self.estimate_global_maxima(arr)
        # optimize estimation by minimizing coast function
        res_0 = scipy.optimize.minimize(
            fun=optimization_loss_function,
            x0=(guess[0],),
            bounds=((guess[0] - 5, guess[0] + 5),),
            args=arr,
            method="Powell",
        )
        angle_0 = res_0.x[0]
        res_1 = scipy.optimize.minimize(
            fun=optimization_loss_function,
            x0=(guess[1]),
            bounds=((guess[1] - 5, guess[1] + 5),),
            args=arr,
            method="Powell",
        )
        angle_1 = res_1.x[0]
        return angle_0, angle_1

    def calc_radius(self, arr: nptyping.NDArray, angle: float) -> float:
        """Calculates the radius (distance from the center) of a beam with
        known angle.

        Args:
            arr (nptyping.NDArray): image of the beam
            angle (float): angle of the beam in degrees

        Returns:
            float: radius of the beam, e. g. the distance from the center
        """
        col = self.collapse_arr(arr, angle)
        x = np.arange(col.size) - col.size / 2
        idx = np.argmax(col)

        window = idx + np.arange(
            -self.half_beam_width, self.half_beam_width + 1
        )  # +1 because else last idx not included

        params = self.beam_model.guess(data=col[window], x=x[window])
        result = self.beam_model.fit(
            data=col[window], params=params, x=x[window]
        )
        # the center of the beam is the radius and it is the center of the
        # Gaussian distribution
        peak_position = result.params["center"].value
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
                image_utils.rotate_image(image=arr, angle=angle, impl="cv2"),
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
        return angles, np.array(
            [self.collapse_arr(arr, angle) for angle in angles]
        )

    def estimate_global_maxima(
        self, arr: nptyping.NDArray, min_angle: Union[float | None] = None
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
        if min_angle is None:
            min_angle = (
                5 * self.angle_space_dim.range / self.angle_space_dim.steps
            )
        for i in range(3):
            angles, angle_space = self.calc_angle_space(arr, offset=i * 15)
            min_distance = int(
                min_angle
                / (self.angle_space_dim.range / self.angle_space_dim.steps)
            )
            maxima = ski.feature.peak_local_max(
                angle_space,
                threshold_abs=np.percentile(angle_space, 50),
                num_peaks=2,
                min_distance=min_distance,
                exclude_border=False,
            )
            assert maxima.shape == (
                2,
                2,
            ), f"Expected 2 Peaks, found {maxima.T[0].shape}"
            est_angle0, est_angle1 = angles[maxima.T[0]]
            if abs(est_angle0 - est_angle1) < 179.5:
                break
        return est_angle0, est_angle1


def optimization_loss_function(angle: float, im: nptyping.NDArray) -> float:
    """Coast function used for accurate estimation of the alignment of a
    straight beam in an image with the first axis of the image.

    Args:
        angle (float): angle to rotate the image in degree
        im (nptyping.NDArray): image to check

    Returns:
        float: score of the alignment
    """
    neg_maximum = -np.max(
        np.mean(
            image_utils.rotate_image(im, angle=angle[0], impl="skimage"),
            axis=0,
        )  # opencv is faster for rotation but has some problems with
        # interpolation TODO: investigate this further
    )
    return neg_maximum
