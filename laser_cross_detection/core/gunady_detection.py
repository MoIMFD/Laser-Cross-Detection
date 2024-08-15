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
    """Laser Cross detection based on the method described by Gunady et al.
    A 2d dimensional gaussian beam is fitted to the image, yielding information
    on the first beam. This beam is subtracted from the image and the procedure
    is repeated, yielding the remaining beam. The method is sensitive to initial
    conditions since many parameter need to be fitted.

        Ian E Gunady et al 2024 Meas. Sci. Technol. 35 105901
        DOI: 10.1088/1361-6501/ad574d
    """

    def __call__(self, arr: nptyping.NDArray, *args, p0, **kwargs) -> nptyping.NDArray:

        height, width = arr.shape
        arr = arr.copy()
        arr[arr < 50] = 0

        def fit_function(
            xy: Tuple[nptyping.NDArray, nptyping.NDArray],
            theta: float,
            rho: float,
            beam_width: float,
            scale: float,
        ):
            """Function describing a 2d gaussian beam. Result is flattened (.ravel) since
            scipy.optimize.curve_fit expects a 1d array to be returned. The flattening does
            not effect the fitting process."""
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
        # fit the first beam
        popt, pcov = sopt.curve_fit(
            fit_function,
            (xx, yy),
            arr.ravel(),
            p0=p0,
        )

        theta1, rho1, *_ = popt
        beam1 = HessNormalLine.from_degrees(
            angle=theta1, distance=rho1, center=(width / 2, height / 2)
        )
        first_beam_image = fit_function((xx, yy), *popt).reshape((height, width))
        # create residual image
        residual = arr - first_beam_image
        residual[residual < 50] = 0
        # fit the second beam
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
