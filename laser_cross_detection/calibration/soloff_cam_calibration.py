from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Callable

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt
from functools import cached_property
import time

from . import SoloffPolynom


@dataclass
class SoloffCamCalibration:
    """Enhanced container holding two Soloff polynomials for camera calibration.

    This class manages the u and v polynomial mappings for a single camera and provides
    methods for projection, reprojection error calculation, and visualization.
    """

    soloff_u: SoloffPolynom
    soloff_v: SoloffPolynom
    camera_id: str = "cam0"  # Added camera identifier for multi-camera systems

    # Added metadata for calibration quality assessment
    _calibration_stats: Dict = field(default_factory=dict)

    @classmethod
    def from_calibration_points(
        cls,
        xyz: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        soloff_type: Tuple[int, int, int],
        camera_id: str = "cam0",
        regularization: float = 0.0,
    ):
        """Creates a single camera calibration with improved fitting and validation.

        Args:
            xyz: Real world coordinates of calibration points (n_points x 3)
            u: u image coordinates (n_points)
            v: v image coordinates (n_points)
            soloff_type: Polynomial orders (x_order, y_order, z_order)
            camera_id: Camera identifier
            regularization: Regularization strength for polynomial fitting (ridge regression)

        Returns:
            SoloffCamCalibration: Calibrated camera
        """
        # Validation
        if u.shape[0] != v.shape[0] or u.shape[0] != xyz.shape[0]:
            raise ValueError("Number of points must match across all coordinates")

        # Time the calibration process
        start_time = time.time()

        # Create and fit polynomials
        soloff_u = SoloffPolynom(*soloff_type)
        soloff_v = SoloffPolynom(*soloff_type)

        # Handle regularization if requested
        EPSILON = 1e-10
        if abs(regularization) > EPSILON:
            # Fit with regularization (ridge regression)
            soloff_u.fit_regularized(xyz, u, alpha=regularization)
            soloff_v.fit_regularized(xyz, v, alpha=regularization)
        else:
            # Standard direct fit
            soloff_u.fit_direct(xyz, u)
            soloff_v.fit_direct(xyz, v)

        # Create the calibration object
        calibration = cls(soloff_u, soloff_v, camera_id)

        # Calculate calibration statistics
        u_pred, v_pred = calibration(xyz)

        # Compute reprojection errors
        u_errors = u - u_pred
        v_errors = v - v_pred
        reprojection_errors = np.sqrt(u_errors**2 + v_errors**2)

        # Store calibration statistics
        calibration._calibration_stats = {
            "num_points": len(xyz),
            "rmse_u": np.sqrt(np.mean(u_errors**2)),
            "rmse_v": np.sqrt(np.mean(v_errors**2)),
            "rmse_total": np.sqrt(np.mean(reprojection_errors**2)),
            "max_error": np.max(reprojection_errors),
            "median_error": np.median(reprojection_errors),
            "calibration_time": time.time() - start_time,
            "polynomial_orders": soloff_type,
            "num_terms_u": len(soloff_u.a),
            "num_terms_v": len(soloff_v.a),
        }

        return calibration

    def __call__(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D world coordinates to 2D image coordinates (vectorized operation).

        Args:
            xyz: World coordinates - can be (3,), (n,3) or (3,n) shaped

        Returns:
            Tuple of u and v image coordinates
        """
        # Handle different input shapes efficiently
        orig_shape = xyz.shape

        if xyz.ndim == 1 and len(xyz) == 3:
            # Single point as flat array
            xyz = xyz.reshape(1, 3)
        elif xyz.ndim == 2:
            if xyz.shape[0] == 3 and xyz.shape[1] != 3:
                # Shape (3,n) - transpose to (n,3)
                xyz = xyz.T

        # Compute projections
        u = self.soloff_u(xyz)
        v = self.soloff_v(xyz)

        return u, v

    def reprojection_error(
        self, xyz: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Calculate reprojection error for given 3D points and their measured 2D coordinates.

        Args:
            xyz: 3D world coordinates
            u: Measured u coordinates
            v: Measured v coordinates

        Returns:
            Array of reprojection errors (Euclidean distance in pixels)
        """
        u_pred, v_pred = self(xyz)
        return np.sqrt((u - u_pred) ** 2 + (v - v_pred) ** 2)

    @property
    def calibration_stats(self) -> Dict:
        """Get the calibration statistics and quality metrics."""
        return self._calibration_stats

    def get_report(self) -> str:
        """Generate a formatted report of calibration quality."""
        if not self._calibration_stats:
            return "No calibration statistics available."

        s = self._calibration_stats
        report = [
            f"Camera: {self.camera_id}",
            f"Polynomial orders: {s.get('polynomial_orders', 'Unknown')}",
            f"Calibration points: {s.get('num_points', 'Unknown')}",
            f"RMSE u: {s.get('rmse_u', 0):.3f} pixels",
            f"RMSE v: {s.get('rmse_v', 0):.3f} pixels",
            f"Total RMSE: {s.get('rmse_total', 0):.3f} pixels",
            f"Maximum error: {s.get('max_error', 0):.3f} pixels",
            f"Median error: {s.get('median_error', 0):.3f} pixels",
            f"Number of terms: u={s.get('num_terms_u', 0)}, v={s.get('num_terms_v', 0)}",
            f"Calibration time: {s.get('calibration_time', 0):.3f} seconds",
        ]
        return "\n".join(report)
