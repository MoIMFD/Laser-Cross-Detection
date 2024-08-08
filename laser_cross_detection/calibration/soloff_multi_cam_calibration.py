from dataclasses import dataclass, field
from typing import List

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt

from . import SoloffCamCalibration


@dataclass
class SoloffMultiCamCalibration:
    """Container for all single camera calibrations in a multi camera setup.
    Provides a method to predict points in world space (x, y, z) based on
    image coordinates (u, v) for all cameras.

    Returns:
        _type_: SoloffMultiCamCalibration
    """

    single_cam_calibrations: List[SoloffCamCalibration] = field(
        default_factory=set()
    )
    opt_method: str = "Powell"

    def add_calibration(self, calibration: SoloffCamCalibration):
        """Adds a single camera calibration to the multi camera calibration
        object.

        Args:
            calibration (SoloffCamCalibration): single camera calibration

        Returns:
            _type_: self
        """
        self.single_cam_calibrations.add(calibration)
        return self

    def __call__(
        self, xyz: nptyping.NDArray[np.float64]
    ) -> List[nptyping.NDArray[np.float64]]:
        """Calculates the pixel coordinates for each single camera calibration
        in the multi camera set for given points in world space (x, y, z)

        Args:
            xyz (nptyping.NDArray[np.float64]): points in world space (x, y, z)

        Returns:
            List[nptyping.NDArray[np.float64]]: respective point in image
                space (u, v) of each camera
        """
        return [calib(xyz) for calib in self.single_cam_calibrations]

    def calculate_point(
        self, *, us: List[float], vs: List[float], x0: List[float]
    ) -> nptyping.NDArray[np.float64]:
        """Calculates the point in world space best fitting the given image
        coordinates for each camera in the multi camera set. Calculation is
        based on minimizing the sum of the reprojection errors.

        Args:
            us (List[float]): list of u image coordinates for each camera
            vs (List[float]): list of v image coordinates for each camera
            x0 (List[float]): initial guess in world coordinates (x, y, z)

        Returns:
            nptyping.NDArray[np.float64]: _description_
        """
        assert (
            len(us) == len(vs) == len(self.single_cam_calibrations)
        ), "Number of image coordinates does not match number of cameras in calibration"

        def opt_fun(
            xyz: nptyping.NDArray[np.float64],
        ) -> float:
            xyz = [xyz]
            return np.sqrt(
                np.sum(
                    [
                        (calibration.soloff_u(xyz) - u) ** 2
                        + (calibration.soloff_v(xyz) - v) ** 2
                        for calibration, u, v in zip(
                            self.single_cam_calibrations, us, vs
                        )
                    ]
                )
            )

        res = sopt.minimize(opt_fun, x0=x0, method=self.opt_method)
        return res.x
