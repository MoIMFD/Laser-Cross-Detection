from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nptyping

PI = np.pi
TWO_PI = np.pi * 2
PI_HALF = np.pi / 2
THREE_PI_HALF = 3 * np.pi / 2


def norm_vector(v: nptyping.NDArray) -> nptyping.NDArray:
    """Normalize a vector to length 1.

    Args:
        v (nptyping.NDArray): vector to normalize

    Returns:
        nptyping.NDArray: normalized vector
    """
    return v / np.linalg.norm(v)


def distance_line_point(
    p1: nptyping.NDArray, p2: nptyping.NDArray, p: nptyping.NDArray
) -> float:
    """Calculates the shortest distance between a line defined by p1 and p2
    and a point p

    Args:
        p1 (nptyping.NDArray): first point on line
        p2 (nptyping.NDArray): second point on line
        p (nptyping.NDArray): point to calculate distance from line

    Returns:
        float: distance between line (p1, p2) and point p
    """
    p1, p2, p = np.array([p1, p2, p])  # make sure all inputs are numpy arrays
    return np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and
    b2,b1. Method is based on the cross product in homogenous coordinate
    space. A detailed explanation is provided in this blog post:
        https://imois.in/posts/line-intersections-with-cross-products/

    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return (x / z, y / z)


@dataclass
class HessNormalLine:
    distance: float
    angle: float
    center: tuple[float, float] = 0, 0

    def __post_init__(self) -> None:
        self.center = np.array(self.center)
        # converte angles (including negative angles)
        # to the range (0, TWOPI]
        self.angle = (TWO_PI + self.angle) % TWO_PI
        if self.angle < PI:
            self.distance = -self.distance
            self.angle = self.angle % PI

    @classmethod
    def from_degrees(cls, distance, angle, center=(0, 0)):
        return cls(distance, np.deg2rad(angle), center=center)

    @classmethod
    def from_intercept_and_slope(cls, intercept, slope, center=(0, 0)):
        """Define a line from intercept and slope. The distance is calculated
        from the triangle formed by the intercept, origin and angel (slope)."""
        angle = np.arctan(slope) - PI / 2
        distance = np.sin(angle) * intercept
        return cls(distance, angle, center=center)

    @classmethod
    def from_direction(cls, p1, direction, center=(0, 0)):
        p2 = np.add(p1, direction)
        p1_center = np.subtract(p1, center)

        # check if slope point line is above or below slope
        # important to calculate correct angle
        if np.cross(direction, p1_center) < 0:
            angle = np.arctan2(direction[1], direction[0]) - PI_HALF
        else:
            angle = np.arctan2(direction[1], direction[0]) + PI_HALF
        distance = distance_line_point(p1, p2, center)
        return cls(distance, angle, center=center)

    @classmethod
    def from_normal(cls, normal, center=(0, 0)):
        direction = -normal[1], normal[0]
        point = np.add(center, normal)
        return cls.from_direction(point, direction, center)

    @classmethod
    def from_two_points(cls, p1, p2, center=(0, 0)):
        p1, p2 = np.array(p1), np.array(p2)
        d = p2 - p1
        return cls.from_direction(p1, d, center)

    @property
    def normal_point(self) -> nptyping.NDArray:
        return self.center + self.distance * self.normal_vector

    @property
    def slope(self) -> float:
        return np.tan(self.angle + PI_HALF)

    @property
    def normal_vector(self) -> nptyping.NDArray:
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def direction_vector(self) -> nptyping.NDArray:
        return np.array([np.cos(self.angle + PI_HALF), np.sin(self.angle + PI_HALF)])

    def plot_slope(self, axis, *args, **kwds):
        return axis.axline(self.normal_point, slope=self.slope, *args, **kwds)

    def interscet_nplinalg(self, other) -> nptyping.NDArray:
        """Calculate the intersection of two instances solving the linear
        equations defining the lines. Currently both lines need
        to share the same origin.

        Args:
            other (HessNormalLine): other instance

        Returns:
            nptyping.NDArray: point of intersection
        """
        assert all(np.isclose(self.center, other.center))
        A = np.vstack([self.normal_vector, other.normal_vector])
        r = np.array([self.distance, other.distance])
        return np.linalg.solve(A, r) + self.center

    def intersect_crossprod(self, other) -> nptyping.NDArray:
        """Calculate the intersection of two instances using the method of
        cross products in homogenous coordinates. Currently both lines need
        to share the same origin.

        Adapted from: https://stackoverflow.com/a/42727584
        Theory described: https://imois.in/posts/line-intersections-with-cross-products/

        Args:
            other (HessNormalLine): other instance

        Returns:
            nptyping.NDArray: point of intersection
        """
        assert all(np.isclose(self.center, other.center))
        a1 = self.normal_point
        a2 = self.normal_point + self.direction_vector

        b1 = other.normal_point
        b2 = other.normal_point + other.direction_vector
        return np.array(get_intersect(a1, a2, b1, b2))


@dataclass
class ComplexHessLine:
    """
    Represents a line using complex numbers in Hessian normal form.

    The complex number z encodes both the distance and angle:
    - |z| (magnitude) is the absolute distance from the center to the line
    - arg(z) (angle) is the direction of the normal vector

    This representation simplifies calculations and handles edge cases better.
    """

    z: complex
    center: tuple[float, float] = (0, 0)

    def __post_init__(self) -> None:
        """Initialize the object and convert center to numpy array"""
        self.center = np.array(self.center)

        # Normalize to have angle in [0, 2π) and positive distance
        angle = np.angle(self.z)
        if angle < 0:
            angle += TWO_PI

        # Ensure positive distance
        distance = max(abs(self.z), 1e-8)

        # Reconstruct the complex number with normalized values
        self.z = distance * np.exp(1j * angle)

    @property
    def distance(self) -> float:
        """Get the distance parameter rho (always positive)"""
        return abs(self.z)

    @property
    def z_normed(self):
        return self.z / abs(self.z)

    @property
    def angle(self) -> float:
        """Get the angle parameter theta in [0, 2π)"""
        angle = np.angle(self.z) % TWO_PI
        return angle if angle >= 0 else angle + TWO_PI

    @property
    def normal_vector(self) -> nptyping.NDArray:
        """Get the unit normal vector pointing from center to line"""
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def direction_vector(self) -> nptyping.NDArray:
        """Get the direction vector along the line (perpendicular to normal)"""
        return np.array([-np.sin(self.angle), np.cos(self.angle)])

    @property
    def normal_point(self) -> nptyping.NDArray:
        """Get the point on the line closest to center"""
        return self.center + self.distance * self.normal_vector

    @property
    def slope(self) -> float:
        """Get the slope of the line"""
        return (
            -np.cos(self.angle) / np.sin(self.angle)
            if np.sin(self.angle) != 0
            else float("inf")
        )

    def rotate(self, angle):
        angle = (self.angle + angle) % TWO_PI
        self.z = self.distance * np.exp(1j * angle)
        return self

    @classmethod
    def from_distance_angle(cls, distance, angle, center=(0, 0)):
        """Create line from distance and angle parameters

        Args:
            distance: Distance from center to line (always positive)
            angle: Angle in radians [0, 2π) of the normal vector
            center: Reference center point
        """
        # Normalize angle to [0, 2π)
        # angle = angle % TWO_PI

        z = max(abs(distance), 1e-8) * np.exp(1j * (angle + (distance < 0) * PI))

        # Ensure positive and finite distance
        # if distance < 0:
        #     angle = (angle - PI) % TWO_PI
        #     distance = abs(distance)

        # distance = max(abs(distance), 1e-8)

        # z = distance * np.exp(1j * angle)
        return cls(z, center=center)

    @classmethod
    def from_degrees(cls, distance, angle_deg, center=(0, 0)):
        """Create line from distance and angle in degrees"""
        angle = np.deg2rad(angle_deg)
        return cls.from_distance_angle(distance, angle, center=center)

    @classmethod
    def from_intercept_and_slope(cls, intercept, slope, center=(0, 0)):
        """Create line from y-intercept and slope in image coordinates

        Args:
            intercept: y-intercept of the line (b in y = m*x + b)
            slope: slope of the line (m in y = m*x + b)
            center: reference center point, usually image center

        Returns:
            ComplexHessLine instance
        """
        # Note: In image coordinates, the y-axis points down
        # This requires special handling for the angle calculation

        if np.isclose(slope, 0):
            angle = PI_HALF  # 90 degrees
            distance = intercept
            return cls.from_distance_angle(distance, angle, center=center)

        elif np.isinf(slope):
            raise ValueError()

        else:
            normal = np.array((-slope, 1))
            normal = normal / np.linalg.norm(normal)
            distance = np.dot([0, intercept], normal)
            angle = np.arctan2(normal[1], normal[0])
            return cls.from_distance_angle(distance, angle, center=center)

    @classmethod
    def from_two_points(cls, p1, p2, center=(0, 0)):
        """Create line from two points"""
        p1, p2 = np.array(p1), np.array(p2)
        if np.all(p1 == p2):
            raise ValueError("Points must be different")

        # Direction vector of the line
        direction = p2 - p1

        # Normal vector (perpendicular to direction)
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)  # normalize

        # Calculate distance
        distance = np.dot(normal, p1 - center)

        # Calculate angle of normal vector
        angle = np.arctan2(normal[1], normal[0])

        # If distance is negative, flip the normal vector (add π to angle)
        if distance < 0:
            angle = (angle - PI) % TWO_PI
            distance = abs(distance)

        return cls.from_distance_angle(distance, angle, center=center)

    @classmethod
    def from_normal(cls, normal, center=(0, 0)):
        """Create line from normal vector"""
        normal = np.array(normal)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Normal vector cannot be zero")

        # Normalize the normal vector
        normal = normal / norm

        # Calculate angle
        angle = np.arctan2(normal[1], normal[0])

        # Calculate distance (projection of normal onto itself)
        distance = norm

        return cls.from_distance_angle(distance, angle, center=center)

    @classmethod
    def from_averaged_lines(cls, lines: List["ComplexHessLine"]):
        if not lines:
            raise ValueError("Cannot calculate average of empty line list")

        # Check if all centers are the same
        centers = np.array([line.center for line in lines])
        if not np.allclose(centers, centers[0]):
            raise ValueError("All lines must share the same center for averaging")

        # Get the complex numbers
        z_values = [line.z for line in lines]

        # Direct averaging
        z_avg = sum(z_values) / len(z_values)

        # Ensure minimum magnitude
        magnitude = abs(z_avg)
        if magnitude < 1e-8:
            # Preserve angle but set minimum magnitude
            angle = np.angle(z_avg)
            z_avg = 1e-8 * np.exp(1j * angle)

        return cls(z_avg, center=centers[0])

    def plot_slope(self, axis, *args, **kwds):
        return axis.axline(self.normal_point, slope=self.slope, *args, **kwds)

    def intersect(self, other):
        """Calculate intersection point with another ComplexHessLine or subclass"""
        if not isinstance(
            other, type(self).__mro__[0]
        ):  # Get the direct class, not base classes
            raise TypeError(f"Can only intersect with another {type(self).__name__}")

        # Check if centers match
        if not np.allclose(self.center, other.center):
            raise ValueError(
                f"Lines must share the same center for intersection calculation. Line1 ({self.center}), Line2 ({other.center})"
            )

        # Extract angles and ensure they're normalized to [0, 2π)
        theta1 = self.angle
        theta2 = other.angle

        # Check if lines are parallel (angles differ by 0 or π)
        angle_diff = abs((theta1 - theta2) % PI)
        if np.isclose(angle_diff, 0) or np.isclose(angle_diff, PI):
            return np.array([float("inf"), float("inf")])

        # Get distances (always positive in this implementation)
        r1 = self.distance
        r2 = other.distance

        # Calculate intersection using Cramer's rule with normal form equations
        det = np.sin(theta2 - theta1)

        # Handle near-zero determinant (nearly parallel lines)
        if abs(det) < 1e-10:
            return np.array([float("inf"), float("inf")])

        x = (r1 * np.sin(theta2) - r2 * np.sin(theta1)) / det
        y = (r2 * np.cos(theta1) - r1 * np.cos(theta2)) / det

        return np.array([x, y]) + self.center

    # For backward compatibility with the original HessNormalLine
    def intersect_crossprod(self, other):
        """Alias for the intersect method, for compatibility with HessNormalLine"""
        return self.intersect(other)

    def to_general_form(self):
        """
        Convert to general form ax + by + c = 0
        Returns coefficients (a, b, c)
        """
        # From Hessian normal form:
        # x*cos(θ) + y*sin(θ) = r
        # This translates to: cos(θ)*x + sin(θ)*y - r = 0

        a = np.cos(self.angle)
        b = np.sin(self.angle)
        c = -self.distance

        # Adjust for center if it's not at origin
        if not np.allclose(self.center, [0, 0]):
            # Translate equation: a(x-x0) + b(y-y0) = r
            # Expands to: ax + by - a*x0 - b*y0 = r
            # Rearranged: ax + by = r + a*x0 + b*y0
            c -= a * self.center[0] + b * self.center[1]

        return a, b, c

    def distance_to_point(self, point):
        """Calculate the shortest distance from a point to this line"""
        point = np.array(point)
        # Distance formula for ax + by + c = 0 is |ax0 + by0 + c|/√(a² + b²)
        a, b, c = self.to_general_form()
        return abs(a * point[0] + b * point[1] + c) / np.sqrt(a * a + b * b)


if __name__ == "__main__":
    fig, ax = plt.subplots()
