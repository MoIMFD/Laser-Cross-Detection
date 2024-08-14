import numpy as np
import numpy.typing as nptyping
import lmfit
import scipy.optimize as sopt
import skimage as ski

from typing import Union, Tuple

from .hess_normal_line import HessNormalLine
from .detection_abc import DetectionMethodABC
from ..utils import image_utils
from ..test import make_beam_image


class Gunady(DetectionMethodABC):
    """ """

    def __call__(
        self, arr: nptyping.NDArray, *args, p0, **kwargs
    ) -> nptyping.NDArray:

        height, width = arr.shape
        arr = arr.copy()
        arr[arr < 50] = 0

        def fit_function(xy, theta, rho, beam_width, scale):
            x, y = xy
            theta = np.deg2rad(theta)
            return (
                scale
                * np.exp(
                    -(
                        (
                            (x - width / 2) * np.cos(theta)
                            + (y - height / 2) * np.sin(theta)
                            - rho
                        )
                        ** 2
                    )
                    / ((beam_width / 3) ** 2)
                ).ravel()
            )

        x = np.arange(width)
        y = np.arange(height)

        xx, yy = np.meshgrid(x, y)
        popt, pcov = sopt.curve_fit(
            fit_function,
            (xx, yy),
            arr.ravel(),
            p0=p0,
            # bounds=([-width / 2, 0, 5, 10], [width / 2, 359, width / 10, 255]),
        )

        theta1, rho1, *_ = popt
        beam1 = HessNormalLine.from_degrees(
            angle=theta1, distance=rho1, center=(width / 2, height / 2)
        )
        first_beam_image = fit_function((xx, yy), *popt).reshape(
            (height, width)
        )

        residual = arr - first_beam_image
        residual[residual < 50] = 0

        popt, pcov = sopt.curve_fit(
            fit_function,
            (xx, yy),
            residual.ravel(),
            p0=p0 + [90, 0, 0, 0],
        )
        theta2, rho2, *_ = popt
        beam2 = HessNormalLine.from_degrees(
            angle=theta2, distance=rho2, center=(width / 2, height / 2)
        )
        return beam1.intersect_crossprod(beam2)
