from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as nptyping

from . import SoloffPolynom


@dataclass
class SoloffCamCalibration:
    """Container holding two Soloff polynomials for each image coordinate u
    and v.

    Returns:
        _type_: SoloffCamCalibration
    """

    soloff_u: SoloffPolynom
    soloff_v: SoloffPolynom

    @classmethod
    def from_clibration_points(
        cls,
        xyz: nptyping.NDArray[np.float64],
        u: nptyping.NDArray[np.float64],
        v: nptyping.NDArray[np.float64],
        soloff_type: Tuple[int, int, int],
    ):
        """Creates a single camera calibration based on Soloff polynomials
        using a set of points with known real world positions (x, y, z) and
        image positions (u, v)

        Args:
            xyz (nptyping.NDArray[np.float64]): real world coordinates of
                calibration points
            u (nptyping.NDArray[np.float64]): u image coordinates of
                calibration points
            v (nptyping.NDArray[np.float64]): v image coordinates of
                 calibration points
            soloff_type (Tuple[int, int, int]): type of the Soloff polynomial
                to use in x, y and z direction

        Returns:
            _type_: SoloffCamCalibration
        """
        soloff_u = SoloffPolynom(*soloff_type)
        soloff_u.fit_least_squares(xyz, u)
        soloff_v = SoloffPolynom(*soloff_type)
        soloff_v.fit_least_squares(xyz, v)
        return cls(soloff_u, soloff_v)

    def __call__(
        self, xyz: nptyping.NDArray[np.float64]
    ) -> nptyping.NDArray[np.float64]:
        """Returns the image coordinates (u, v) of a given points in world
        space (x, y, z).

        Args:
            xyz (nptyping.NDArray[np.float64]): world coordinates

        Returns:
        nptyping.NDArray[np.float64]: image coordinates
        """
        return self.soloff_u(xyz), self.soloff_v(xyz)
