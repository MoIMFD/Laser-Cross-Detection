from dataclasses import dataclass
from typing import Tuple, List
from itertools import combinations_with_replacement
from collections import Counter
import functools

import numpy as np
import numpy.typing as nptyping
import scipy.optimize as sopt


def make_3d_polynom(*orders: int) -> List[Tuple[str]]:
    """Creates a mapping of all combinations of parameters for the 3d
    polynomial. For example orders = (3, 3, 2), e. g. cubic order in x,
    cubic order in y and quadratic order in z  :
                [('x1',),
                 ('x2',),
                 ('x3',),
                 ('x1', 'x1'),
                 ('x1', 'x2'),
                 ('x1', 'x3'),
                 ('x2', 'x2'),
                 ('x2', 'x3'),
                 ('x3', 'x3'),
                 ('x1', 'x1', 'x1'),
                 ('x1', 'x1', 'x2'),
                 ('x1', 'x1', 'x3'),
                 ('x1', 'x2', 'x2'),
                 ('x1', 'x2', 'x3'),
                 ('x1', 'x3', 'x3'),
                 ('x2', 'x2', 'x2'),
                 ('x2', 'x2', 'x3'),
                 ('x2', 'x3', 'x3')]
    Using a mapping (dict) these can be replaced by actual values. The
    variable names are x1 to xn, where n is the number of provided arguments.

    Args:
        orders (int): polynomial order in respective dimension

    Returns:
        List[Tuple[str]]: Terms of the polynomial with the specified orders
    """
    max_order = max(orders)
    x = [f"x{i+1}" for i, _ in enumerate(orders)]
    params = [
        c
        for i in range(max_order)
        for c in combinations_with_replacement(x, r=i + 1)
    ]
    return [
        param
        for param in params
        if all(
            [
                Counter(param)[f"x{i+1}"] <= order
                for i, order in enumerate(orders)
            ]
        )
    ]


@dataclass
class SoloffPolynom:
    x_order: int
    y_order: int
    z_order: int
    a: nptyping.NDArray = None

    def __post_init__(self):
        if self.a is None:
            self.a = np.ones(len(self.polynom) + 1)
        else:
            self.a = np.asarray(self.a).ravel()

    def __call__(self, *args: nptyping.NDArray):
        if len(args) == 3:
            M = self.build_m(*args)
        elif len(args) == 1:
            xyz = np.array(args)
            try:
                M = self.build_m(*xyz)
            except TypeError:
                M = self.build_m(*xyz.T)
        else:
            raise ValueError("Invalid number of Arguments")
        return np.matmul(M, self.a)

    def fit_curve_fit(self, xyz, u):
        popt, pcov = sopt.curve_fit(
            self.fn_opt, xyz, u, p0=self.a, method="lm"
        )
        self.a = np.asarray(popt)
        return self

    def fit_least_squares(self, xyz, u):
        result = sopt.least_squares(self.fn_opt_ls(xyz, u), x0=self.a)
        self.a = np.asarray(result.x)
        return self

    @functools.cached_property
    def polynom(self):
        return make_3d_polynom(self.x_order, self.y_order, self.z_order)

    def fn_opt_ls(self, xyz, u):
        def f(a):
            s = self.fn_opt(xyz, *a)
            return s - u

        return f

    def fn_opt(self, xyz, *a):
        self.a = np.asarray(a)
        return self.__call__(xyz)

    def build_m(
        self, x1: nptyping.NDArray, x2: nptyping.NDArray, x3: nptyping.NDArray
    ) -> nptyping.NDArray:
        mapping = {"x1": x1, "x2": x2, "x3": x3}
        m = [
            np.prod([mapping[param] for param in params], axis=0)
            for params in self.polynom
        ]
        return np.hstack([np.ones_like(x1), *m])
