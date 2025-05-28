from dataclasses import dataclass, field

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt

from .soloff_cam_calibration import SoloffCamCalibration


@dataclass
class SoloffMultiCamCalibration:
    """Enhanced container for managing a multi-camera calibration system.

    Provides improved methods for triangulation, uncertainty estimation, and
    optimization of 3D reconstruction from multiple camera views.
    """

    single_cam_calibrations: list[SoloffCamCalibration] = field(default_factory=list)

    # Optimization configuration with sensible defaults
    optimization_config: dict = field(
        default_factory=lambda: {
            "method": "Powell",
            "options": {"maxiter": 1000, "ftol": 1e-6, "xtol": 1e-6},
            "bounds": None,
        }
    )

    def add_calibration(self, calibration: SoloffCamCalibration):
        """Add a single camera calibration to the system.

        Args:
            calibration: Single camera calibration to add

        Returns:
            self: For method chaining
        """
        self.single_cam_calibrations.append(calibration)
        return self

    def __call__(
        self, xyz: nptyping.NDArray
    ) -> list[tuple[nptyping.NDArray, nptyping.NDArray]]:
        """Project 3D points to all cameras (vectorized operation).

        Args:
            xyz: World coordinates to project

        Returns:
            List of (u,v) tuples for each camera
        """
        return [calibration(xyz) for calibration in self.single_cam_calibrations]

    def calculate_point(
        self,
        us: list[float | nptyping.NDArray],
        vs: list[float | nptyping.NDArray],
        x0: nptyping.NDArray | None = None,
        weights: nptyping.NDArray | None = None,
        method: str | None = None,
        full_output: bool = False,
    ) -> nptyping.NDArray | tuple[nptyping.NDArray, dict]:
        """Calculate 3D position from multiple camera views with improved optimization.

        Args:
            us: List of u coordinates for each camera
            vs: List of v coordinates for each camera
            x0: Initial guess (defaults to [0,0,0] if None)
            weights: Optional weights for each camera (for handling reliability
                differences)
            method: Override optimization method (uses config default if None)
            full_output: Whether to return optimization details

        Returns:
            Reconstructed 3D point (and optimization details if full_output=True)
        """
        if len(us) != len(vs) or len(us) != len(self.single_cam_calibrations):
            raise ValueError("Number of coordinates must match number of cameras")

        # Default initial guess at origin if not provided
        if x0 is None:
            x0 = np.zeros(3)

        # Default weights to equal if not provided
        if weights is None:
            weights = np.ones(len(us))
        elif len(weights) != len(us):
            raise ValueError("Number of weights must match number of cameras")

        # Create vectorized cost function for better performance
        def reprojection_cost(xyz_flat):
            xyz = xyz_flat.reshape(1, 3)  # Reshape for vectorized operations

            # Calculate squared reprojection errors for all cameras at once
            total_error = 0.0
            for i, (calibration, u, v, weight) in enumerate(
                zip(self.single_cam_calibrations, us, vs, weights)
            ):
                # Project 3D point to camera
                u_pred, v_pred = calibration(xyz)

                # Calculate weighted squared error
                cam_error = ((u_pred - u) ** 2 + (v_pred - v) ** 2) * weight
                total_error += cam_error

            return np.sqrt(total_error)

        # Get optimization method (use override or config)
        opt_method = method if method else self.optimization_config["method"]

        # Run optimization
        result = sopt.minimize(
            reprojection_cost,
            x0=x0,
            method=opt_method,
            bounds=self.optimization_config.get("bounds"),
            options=self.optimization_config.get("options", {}),
        )

        # Return result based on full_output flag
        if full_output:
            return result.x, {
                "success": result.success,
                "reprojection_error": result.fun,
                "iterations": result.nit if hasattr(result, "nit") else None,
                "message": result.message,
            }
        else:
            return result.x

    def triangulate_points(
        self,
        point_correspondences: list[tuple[list[float], list[float]]],
        initial_guess: nptyping.NDArray | None = None,
        weights: list[nptyping.NDArray] | None = None,
    ) -> nptyping.NDArray:
        """Triangulate multiple points from corresponding image coordinates.

        Args:
            point_correspondences: List of (us, vs) tuples where each us and vs
                                   is a list of coordinates across all cameras
            initial_guess: Optional initial guess for all points
            weights: Optional weights for each point's cameras

        Returns:
            Array of 3D points
        """
        n_points = len(point_correspondences)
        result = np.zeros((n_points, 3))

        # Process each point
        for i, (us, vs) in enumerate(point_correspondences):
            # Get initial guess for this point
            x0 = initial_guess[i] if initial_guess is not None else None

            # Get weights for this point
            w = weights[i] if weights is not None else None

            # Triangulate point
            result[i] = self.calculate_point(us, vs, x0=x0, weights=w)

        return result

    def estimate_uncertainty(
        self,
        point_3d: nptyping.NDArray,
        us: list[float],
        vs: list[float],
        confidence: float = 0.95,
    ) -> tuple[nptyping.NDArray, nptyping.NDArray]:
        """Estimate uncertainty of the reconstructed 3D point.

        Uses local Jacobian approximation to estimate uncertainty ellipsoid.

        Args:
            point_3d: Reconstructed 3D point
            us, vs: Image coordinates used for reconstruction
            confidence: Confidence level (e.g., 0.95 for 95% confidence)

        Returns:
            Tuple of (principal_axes, axis_lengths) of uncertainty ellipsoid
        """
        # Convert point to proper shape
        point_3d = np.array(point_3d).reshape(3)

        # Calculate Jacobian matrix using finite differences
        epsilon = 1e-5
        jacobian = np.zeros((2 * len(self.single_cam_calibrations), 3))

        # For each dimension (x, y, z)
        for i in range(3):
            # Perturb in this dimension
            delta = np.zeros(3)
            delta[i] = epsilon

            # Forward difference
            point_plus = point_3d + delta
            projections_plus = self(point_plus.reshape(1, 3))

            # Extract all u, v coordinates
            uv_plus = []
            for u_plus, v_plus in projections_plus:
                uv_plus.extend([u_plus[0], v_plus[0]])
            uv_plus = np.array(uv_plus)

            # Backward difference
            point_minus = point_3d - delta
            projections_minus = self(point_minus.reshape(1, 3))

            # Extract all u, v coordinates
            uv_minus = []
            for u_minus, v_minus in projections_minus:
                uv_minus.extend([u_minus[0], v_minus[0]])
            uv_minus = np.array(uv_minus)

            # Calculate derivative
            jacobian[:, i] = (uv_plus - uv_minus) / (2 * epsilon)

        # Flatten observed coordinates
        uv_observed = []
        for u, v in zip(us, vs):
            uv_observed.extend([u, v])
        uv_observed = np.array(uv_observed)

        # Project current estimate
        projections = self(point_3d.reshape(1, 3))
        uv_estimated = []
        for u_est, v_est in projections:
            uv_estimated.extend([u_est[0], v_est[0]])
        uv_estimated = np.array(uv_estimated)

        # Calculate residuals
        residuals = uv_observed - uv_estimated

        # Estimate measurement error (variance in image space)
        sigma_sq = np.mean(residuals**2)

        # Calculate covariance matrix in 3D space
        # Using pseudo-inverse for potentially under-constrained systems
        J_pinv = np.linalg.pinv(jacobian)
        cov_matrix = sigma_sq * (J_pinv @ J_pinv.T)

        # Eigendecomposition to get uncertainty ellipsoid
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Scale for confidence interval
        from scipy import stats

        chi2_val = stats.chi2.ppf(confidence, df=3)
        axis_lengths = np.sqrt(eigenvalues * chi2_val)

        return eigenvectors, axis_lengths

    def get_system_report(self) -> str:
        """Generate a comprehensive report of the multi-camera system."""
        if not self.single_cam_calibrations:
            return "No cameras in the system."

        num_cameras = len(self.single_cam_calibrations)

        # System overview
        report = [
            "=== Soloff Multi-Camera System Report ===",
            f"Number of cameras: {num_cameras}",
            f"Optimization method: {self.optimization_config['method']}",
            "",
        ]

        # Add individual camera reports
        for i, cam in enumerate(self.single_cam_calibrations):
            report.append(f"--- Camera {i + 1}: {cam.camera_id} ---")
            report.append(cam.get_report())
            report.append("")

        return "\n".join(report)

    def save_calibration(self, filename: str):
        """Save the multi-camera calibration to a file.

        Args:
            filename: Path to save the calibration
        """
        import pickle

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_calibration(cls, filename: str) -> "SoloffMultiCamCalibration":
        """Load a multi-camera calibration from a file.

        Args:
            filename: Path to the saved calibration

        Returns:
            Loaded calibration object
        """
        import pickle

        with open(filename, "rb") as f:
            return pickle.load(f)
