import numpy as np
import numpy.polynomial.polynomial as poly
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from itertools import combinations_with_replacement


class PolynomialBasis:
    """
    A class to handle multivariate polynomial basis functions using NumPy's polynomial capabilities.
    """

    def __init__(self, *orders: int):
        """
        Initialize a multivariate polynomial basis.

        Args:
            *orders: Maximum order for each dimension
        """
        self.orders = orders
        self.n_dims = len(orders)
        self.max_order = max(orders)
        self._powers = self._generate_powers()

    def _generate_powers(self) -> List[Tuple[int, ...]]:
        """
        Generate power tuples for multivariate polynomial terms.

        Returns:
            List of tuples where each tuple contains powers for each dimension
        """
        all_powers = []

        # Add constant term (all zeros)
        all_powers.append(tuple([0] * self.n_dims))

        # For each possible total order
        for total_order in range(1, self.max_order + 1):
            # Generate all possible combinations of dimensions with replacement
            for combo in combinations_with_replacement(range(self.n_dims), total_order):
                # Count occurrences of each dimension
                powers = [0] * self.n_dims
                for dim_idx in combo:
                    powers[dim_idx] += 1

                # Check if any dimension exceeds its max order
                if all(powers[i] <= self.orders[i] for i in range(self.n_dims)):
                    all_powers.append(tuple(powers))

        return all_powers

    def __len__(self) -> int:
        """Return the number of terms in the basis."""
        return len(self._powers)

    def powers(self) -> List[Tuple[int, ...]]:
        """Return the power tuples for all terms."""
        return self._powers

    def evaluate(self, *coords: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at given coordinates.

        Args:
            *coords: Arrays of coordinates for each dimension

        Returns:
            Matrix where each column is a basis function evaluated at all points
        """
        if len(coords) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} coordinate arrays, got {len(coords)}"
            )

        # Initialize design matrix
        n_points = len(coords[0])
        n_terms = len(self._powers)
        design_matrix = np.ones((n_points, n_terms))

        # Fill design matrix with evaluated basis functions
        for i, powers in enumerate(self._powers):
            for dim, power in enumerate(powers):
                if power > 0:
                    design_matrix[:, i] *= np.power(coords[dim], power)

        return design_matrix

    def evaluate_single_term(self, term_idx: int, *coords: np.ndarray) -> np.ndarray:
        """
        Evaluate a single basis function at given coordinates.

        Args:
            term_idx: Index of the term to evaluate
            *coords: Arrays of coordinates for each dimension

        Returns:
            Array of evaluated basis function values
        """
        if len(coords) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} coordinate arrays, got {len(coords)}"
            )

        powers = self._powers[term_idx]
        result = np.ones_like(coords[0])

        for dim, power in enumerate(powers):
            if power > 0:
                result *= np.power(coords[dim], power)

        return result

    def get_symbolic_terms(self, var_names: List[str] = None) -> List[str]:
        """
        Get symbolic representation of each basis function.

        Args:
            var_names: Optional list of variable names (defaults to x1, x2, ...)

        Returns:
            List of strings representing each term
        """
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(self.n_dims)]

        if len(var_names) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} variable names, got {len(var_names)}"
            )

        symbolic_terms = []

        for powers in self._powers:
            # Start with 1 for constant term
            term_parts = []

            for dim, power in enumerate(powers):
                if power == 0:
                    continue
                elif power == 1:
                    term_parts.append(f"{var_names[dim]}")
                else:
                    term_parts.append(f"{var_names[dim]}^{power}")

            if not term_parts:
                symbolic_terms.append("1")  # Constant term
            else:
                symbolic_terms.append("*".join(term_parts))

        return symbolic_terms
