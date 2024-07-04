import numpy as np
import lmfit
import scipy
import cv2
from collections import namedtuple
import skimage
import skimage.feature

from .hess_normal_line import HessNormalLine
from .detection_abc import DetectionMethodABC
from ..utils import image_utils

AngleSpaceDimension = namedtuple("AngleSpaceDimension", "start range steps")


class Kluwe(DetectionMethodABC):
    def __init__(
        self,
        cvt_uint8=False,
        beam_width=20,
        start_angle=0,
        angle_range=180,
        angle_steps=180,
        beam_model=lmfit.models.GaussianModel(),
        interpolation_order: int = 3,
    ) -> None:
        self.cvt_uint8 = cvt_uint8
        self.angle_space_dim = AngleSpaceDimension(
            start_angle, angle_range, angle_steps
        )
        self.beam_model = beam_model
        self.beam_width = beam_width
        self.interpolation_order = interpolation_order

    @property
    def half_beam_width(self):
        return self.beam_width // 2

    def __call__(self, image, *args, **kwds):

        if self.cvt_uint8:
            image = cv2.convertScaleAbs(image)

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

    def calc_angles(self, arr):
        guess = self.estimate_global_maxima(arr)

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

    def calc_radius(self, arr, angle):
        # window size: per side -> 3 means a full window of 7 indizes
        # radius about the center of the array
        col = self.collapse_arr(arr, angle)  # cv.INTER_CUBIC)
        x = np.arange(col.size) - col.size / 2
        idx = np.argmax(col)

        window = idx + np.arange(
            -self.half_beam_width, self.half_beam_width + 1
        )  # +1 bc else last idx not included

        params = self.beam_model.guess(data=col[window], x=x[window])
        result = self.beam_model.fit(
            data=col[window], params=params, x=x[window]
        )
        # the center of the beam is the radius and it is the center of the gauss distribution
        peak_position = result.params["center"].value
        return peak_position

    def collapse_arr(self, arr, angle: float = 0.0):  # cv.INTER_LINEAR
        if angle == 0:
            col = np.mean(arr, axis=0).flatten()
        else:
            # col = np.sum(skimage.transform.rotate(arr, angle=angle, preserve_range=True), axis=0)
            col = np.mean(
                image_utils.rotate_image(image=arr, angle=angle, impl="cv2"),
                axis=0,
            ).flatten()
        return col

    def calc_angle_space(self, arr, offset=0):
        start, ang_range, nsamples = self.angle_space_dim
        start += offset
        angles = np.linspace(
            self.angle_space_dim.start + offset,
            self.angle_space_dim.start + offset + self.angle_space_dim.range,
            self.angle_space_dim.steps,
            endpoint=False,
        )
        return angles, np.array(
            [self.collapse_arr(arr, angle) for angle in angles]
        )

    def estimate_global_maxima(self, arr):
        for i in range(3):
            angles, angle_space = self.calc_angle_space(arr, offset=i * 15)
            min_angle = 5  # minimum angle between the lines
            min_distance = int(
                (self.angle_space_dim.range / self.angle_space_dim.steps)
                * min_angle
            )
            maxima = skimage.feature.peak_local_max(
                angle_space,
                threshold_abs=np.percentile(angle_space, 80),
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


def optimization_loss_function(angle, im):
    neg_maximum = -np.max(
        np.mean(
            image_utils.rotate_image(im, angle=angle[0], impl="skimage"),
            axis=0,
        )  # falls es Probleme gibt hier auf skimage oder ndimage wechseln
    )
    # neg_maximum = -np.percentile(
    #     np.mean(
    #         rotate_image(im, angle=angle[0], interp_flag=cv.INTER_CUBIC),
    #         axis=0,
    #     ),
    #     99.5,
    # )
    return neg_maximum
