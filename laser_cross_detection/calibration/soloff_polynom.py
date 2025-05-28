from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt

from .polynom_basis import PolynomialBasis


@dataclass
class SoloffPolynom:
    """
    Enhanced Soloff polynomial using NumPy's polynomial capabilities.

    This implementation leverages NumPy's efficient polynomial operations
    and provides a clean mathematical representation of the polynomial.
    """

    x_order: int
    y_order: int
    z_order: int
    a: Optional[nptyping.NDArray] = None

    def __post_init__(self):
        # Initialize the polynomial basis
        self.basis = PolynomialBasis(self.x_order, self.y_order, self.z_order)

        # Initialize coefficients
        if self.a is None:
            self.a = np.zeros(len(self.basis))
            self.a[0] = 1.0  # Set constant term to 1
        else:
            self.a = np.asarray(self.a).flatten()

            # Ensure coefficients match basis size
            if len(self.a) != len(self.basis):
                raise ValueError(
                    f"Expected {len(self.basis)} coefficients, got {len(self.a)}"
                )

    def __call__(self, *args) -> nptyping.NDArray:
        """
        Evaluate the polynomial at given points.

        Args can be:
            - x, y, z as separate arrays
            - A single array of shape [3, n_points]
            - A single array of shape [n_points, 3]

        Returns:
            nptyping.NDArray: Polynomial values
        """
        # Process input coordinates
        if len(args) == 3:
            # Individual x, y, z arrays
            x, y, z = args
        elif len(args) == 1:
            # Single array containing xyz coordinates
            xyz = np.asarray(args[0])

            if xyz.ndim == 1 and len(xyz) == 3:
                # Single point [x, y, z]
                x = np.array([xyz[0]])
                y = np.array([xyz[1]])
                z = np.array([xyz[2]])
            elif xyz.shape[0] == 3:
                # Array of shape [3, n_points]
                x, y, z = xyz
            else:
                # Array of shape [n_points, 3]
                x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        else:
            raise ValueError("Invalid input format")

        # Evaluate basis functions
        design_matrix = self.basis.evaluate(x, y, z)

        # Compute polynomial value
        return design_matrix @ self.a

    def fit_direct(self, xyz: nptyping.NDArray, u: nptyping.NDArray) -> "SoloffPolynom":
        """
        Fit polynomial coefficients using direct linear solving.

        Args:
            xyz: Coordinates, shape [n_points, 3] or [3, n_points]
            u: Target values, shape [n_points]

        Returns:
            self: Updated instance with fitted coefficients
        """
        # Extract coordinates
        if xyz.shape[0] == 3:
            x, y, z = xyz
        else:
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Build design matrix
        design_matrix = self.basis.evaluate(x, y, z)

        # Solve linear system directly using NumPy's least squares
        self.a, residuals, rank, s = np.linalg.lstsq(design_matrix, u, rcond=None)

        return self

    def fit_regularized(
        self, xyz: nptyping.NDArray, u: nptyping.NDArray, alpha: float = 0.1
    ) -> "SoloffPolynom":
        """
        Fit polynomial coefficients using direct linear solving.

        Args:
            xyz: Coordinates, shape [n_points, 3] or [3, n_points]
            u: Target values, shape [n_points]
            alpha:

        Returns:
            self: Updated instance with fitted coefficients
        """
        if alpha < 0:
            import warnings

            warnings.warn(
                f"Negative regularization parameter ({alpha}) is invalid and may cause "
                f"numerical instability. Using abs({alpha}) instead."
            )
            alpha = abs(alpha)

        EPSILON = 1e-10

        if abs(alpha) < EPSILON:
            return self.fit_direct(xyz, u)

        # Extract coordinates
        if xyz.shape[0] == 3:
            x, y, z = xyz
        else:
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Build design matrix
        A = self.basis.evaluate(x, y, z)

        ATA = A.T @ A
        reg_matrix = alpha * np.eye(ATA.shape[0])

        # don't regularize the constant term
        reg_matrix[0, 0] = 0
        ATu = A.T @ u
        self.a = np.linalg.solve(ATA + reg_matrix, ATu)

        return self

    def fit_curve_fit(
        self, xyz: nptyping.NDArray, u: nptyping.NDArray
    ) -> "SoloffPolynom":
        """
        Fit polynomial coefficients using scipy's curve_fit.

        Args:
            xyz: Coordinates, shape [n_points, 3] or [3, n_points]
            u: Target values, shape [n_points]

        Returns:
            self: Updated instance with fitted coefficients
        """

        def model_func(xyz, *params):
            self.a = np.array(params)
            return self(xyz)

        initial_params = self.a if self.a is not None else np.ones(len(self.basis))
        popt, pcov = sopt.curve_fit(model_func, xyz, u, p0=initial_params)

        self.a = popt
        return self

    def get_symbolic_representation(self) -> str:
        """
        Get a human-readable representation of the polynomial.

        Returns:
            str: Symbolic formula with coefficients
        """
        terms = self.basis.get_symbolic_terms(["x", "y", "z"])
        result = []

        for i, (term, coef) in enumerate(zip(terms, self.a)):
            if abs(coef) < 1e-10:  # Skip near-zero coefficients
                continue

            if i == 0:  # Constant term
                result.append(f"{coef:.4f}")
            elif coef > 0:
                result.append(f"+ {coef:.4f}·{term}")
            else:
                result.append(f"- {abs(coef):.4f}·{term}")

        if not result:
            return "0"

        return " ".join(result)

    def derive(self, dimension: int) -> "SoloffPolynom":
        """
        Compute the derivative of the polynomial with respect to a dimension.

        Args:
            dimension: 0 for x, 1 for y, 2 for z

        Returns:
            SoloffPolynom: New polynomial representing the derivative
        """
        if dimension < 0 or dimension >= 3:
            raise ValueError("Dimension must be 0 (x), 1 (y), or 2 (z)")

        # Create new polynomial with same orders
        deriv_poly = SoloffPolynom(self.x_order, self.y_order, self.z_order)
        deriv_coeffs = np.zeros_like(self.a)

        # For each term in the original polynomial
        for i, powers in enumerate(self.basis.powers()):
            power = powers[dimension]

            if power > 0:
                # Find the corresponding term with power-1 in the target dimension
                new_powers = list(powers)
                new_powers[dimension] -= 1
                new_powers = tuple(new_powers)

                # Find this term in the basis
                try:
                    j = self.basis.powers().index(new_powers)
                    deriv_coeffs[j] += self.a[i] * power
                except ValueError:
                    # Term not in basis (shouldn't happen with proper basis)
                    pass

        deriv_poly.a = deriv_coeffs
        return deriv_poly

    def integrate(self, dimension: int) -> "SoloffPolynom":
        """
        Compute the indefinite integral of the polynomial with respect to a dimension.

        Args:
            dimension: 0 for x, 1 for y, 2 for z

        Returns:
            SoloffPolynom: New polynomial representing the integral
        """
        if dimension < 0 or dimension >= 3:
            raise ValueError("Dimension must be 0 (x), 1 (y), or 2 (z)")

        # We may need higher order for integration
        new_orders = list(self.basis.orders)
        new_orders[dimension] += 1

        # Create new polynomial with increased order in the integration dimension
        int_poly = SoloffPolynom(*new_orders)
        int_coeffs = np.zeros(len(int_poly.basis))

        # For each term in the original polynomial
        for i, powers in enumerate(self.basis.powers()):
            # New power in the integration dimension
            new_powers = list(powers)
            new_powers[dimension] += 1
            new_powers = tuple(new_powers)

            # Find this term in the new basis
            try:
                j = int_poly.basis.powers().index(new_powers)
                int_coeffs[j] = self.a[i] / new_powers[dimension]
            except ValueError:
                # Term not in new basis (shouldn't happen with proper basis)
                pass

        int_poly.a = int_coeffs
        return int_poly
