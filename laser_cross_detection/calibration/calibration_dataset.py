from dataclasses import dataclass

import pandas as pd
import numpy as np
import numpy.typing as nptyping


@dataclass
class CameraCalibrationSet:
    """Container to store camera calibration data containing world coordiantes
    xyz and the repspective image coordinates (u, v)
    """

    xyz: nptyping.NDArray[np.float64]
    uv: nptyping.NDArray[np.float64]

    @classmethod
    def from_path(
        cls,
        calibration_path,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
    ):
        data = pd.read_csv(
            calibration_path, comment="#", names="image u v x y z".split()
        )
        order = np.lexsort(data[["z", "y", "x"]].values.T)
        data = data.iloc[order, :]
        data["x"] = data["x"] * x_scale + x_offset
        data["y"] = data["y"] * y_scale + y_offset
        data["z"] = data["z"] * z_scale + z_offset
        return cls(
            data[["x", "y", "z"]].values.astype(np.float64),
            data[["u", "v"]].values.astype(np.float64),
        )

    @property
    def x(self) -> nptyping.NDArray[np.float64]:
        """Returns the x world coordinate

        Returns:
            nptyping.NDArray[np.float64]: x world coordinate
        """
        return self.xyz[:, 0]

    @property
    def y(self) -> nptyping.NDArray[np.float64]:
        """Returns the y world coordinate

        Returns:
            nptyping.NDArray[np.float64]: y world coordinate
        """
        return self.xyz[:, 1]

    @property
    def z(self) -> nptyping.NDArray[np.float64]:
        """Returns the z world coordinate

        Returns:
            nptyping.NDArray[np.float64]: z world coordinate
        """
        return self.xyz[:, 2]

    @property
    def u(self) -> nptyping.NDArray[np.float64]:
        """Returns the u image coordinate

        Returns:
            nptyping.NDArray[np.float64]: u image coordinate
        """
        return self.uv[:, 0]

    @property
    def v(self) -> nptyping.NDArray[np.float64]:
        """Returns the v image coordinate

        Returns:
            nptyping.NDArray[np.float64]: v image coordinate
        """
        return self.uv[:, 1]
