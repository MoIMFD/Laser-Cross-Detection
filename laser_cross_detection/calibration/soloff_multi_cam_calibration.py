from dataclasses import dataclass, field
from typing import List

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt

from . import SoloffCamCalibration


@dataclass
class SoloffMultiCamCalibration:
    single_cam_calibrations: List[SoloffCamCalibration] = field(
        default_factory=lambda: []
    )
    opt_method: str = "Powell"

    def add_calibration(self, calibration: SoloffCamCalibration):
        self.single_cam_calibrations.append(calibration)
        return self

    def __call__(
        self, xyz: nptyping.NDArray[np.float64]
    ) -> List[nptyping.NDArray[np.float64]]:
        return [calib(xyz) for calib in self.single_cam_calibrations]

    def calculate_point(
        self, *, us: List[float], vs: List[float], x0: List[float]
    ) -> nptyping.NDArray[np.float64]:
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
