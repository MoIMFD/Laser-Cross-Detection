import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt

from .detection_abc import DetectionMethodABC
from .hess_normal_line import HessNormalLine


class Gunady(DetectionMethodABC):
    """Laser Cross detection based on the method described by Gunady et al.
    A 2d dimensional gaussian beam is fitted to the image, yielding information
    on the first beam. This beam is subtracted from the image and the procedure
    is repeated, yielding the remaining beam. The method is sensitive to initial
    conditions since many parameter need to be fitted.

        Ian E Gunady et al 2024 Meas. Sci. Technol. 35 105901
        DOI: 10.1088/1361-6501/ad574d
    """

    def __call__(
        self,
        arr: nptyping.NDArray,
        *args,
        p01: tuple[float, float, float, float],
        p02: tuple[float, float, float, float] | None = None,
        threshold: float = 100,
        **kwargs,
    ) -> nptyping.NDArray:
        """Estimates the intersection point of two gaussian beams in a 2d
        image by fitting gaussian beams using least squares method. The method
        is sensitive regarding starting points. p01 and p02 should be chosen
        as close as possible to the actual beam parameter.

        Args:
            arr (nptyping.NDArray): image to process
            p01 (Tuple[float, float, float, float]): starting beam parameter
                set for the first beam used for optimization: angle,
                distance from center, beam width, peak intensity
            p02 (Tuple[float, float, float, float]): starting beam parameter
                set for the second beam used for optimization: angle,
                distance from center, beam width, peak intensity
                ->  If not specified the set for beam one is used but rotated
                    by 90 degrees
            threshold (int): intensity threshold used to remove dark objects
                not belonging to the beams

        Returns:
            nptyping.NDArray: 1d array containing the coordinates (x, y) of the
                intersection
        """

        height, width = arr.shape
        arr = arr.copy()
        arr[arr < threshold] = 0

        def fit_function(
            xy: tuple[nptyping.NDArray, nptyping.NDArray],
            theta: float,
            rho: float,
            beam_width: float,
            scale: float,
        ):
            """Function describing a 2d gaussian beam. Result is flattened
            (.ravel) since scipy.optimize.curve_fit expects a 1d array to be
            returned. The flattening does not effect the fitting process.
            """
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
        popt, _pcov = sopt.curve_fit(
            fit_function,
            (xx, yy),
            arr.ravel(),
            p0=p01,
        )

        theta1, rho1, *_ = popt
        beam1 = HessNormalLine.from_degrees(
            angle=theta1, distance=rho1, center=(width / 2, height / 2)
        )
        first_beam_image = fit_function((xx, yy), *popt).reshape((height, width))
        # create residual image
        residual = arr - first_beam_image
        residual[residual < threshold] = 0
        # fit the second beam
        if p02 is None:
            p02 = np.add(p01, [90, 0, 0, 0])
        popt, pcov = sopt.curve_fit(
            fit_function,
            (xx, yy),
            residual.ravel(),
            p0=p02,
        )
        theta2, rho2, *_ = popt
        beam2 = HessNormalLine.from_degrees(
            angle=theta2, distance=rho2, center=(width / 2, height / 2)
        )
        return beam1.intersect_crossprod(beam2)
