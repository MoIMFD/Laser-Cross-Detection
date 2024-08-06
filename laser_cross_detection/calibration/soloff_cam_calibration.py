from dataclasses import dataclass
from typing import Tuple

from . import SoloffPolynom


@dataclass
class SoloffCamCalibration:
    soloff_u: SoloffPolynom
    soloff_v: SoloffPolynom

    @classmethod
    def from_clibration_points(
        cls, xyz, u, v, soloff_type: Tuple[int, int, int]
    ):
        soloff_u = SoloffPolynom(*soloff_type)
        soloff_u.fit_least_squares(xyz, u)
        soloff_v = SoloffPolynom(*soloff_type)
        soloff_v.fit_least_squares(xyz, v)
        return cls(soloff_u, soloff_v)

    def __call__(self, xyz):
        return self.soloff_u(xyz), self.soloff_v(xyz)
