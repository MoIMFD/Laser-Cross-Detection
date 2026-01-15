from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from laser_cross_detection.core import ComplexHessLine, Kluwe
from laser_cross_detection.test import (
    BeamConfig,
    ImageConfig,
    NoiseConfig,
    SyntheticImageGenerator,
)
from laser_cross_detection.utils.image_utils import ImageDimension

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GeneratedTestCase:
    config: ImageConfig
    image: NDArray
    expected_angles: NDArray
    expected_intersection: NDArray
    raw_angles: tuple[float, float]


@st.composite
def beam_image_noise_free(draw, dimension: tuple[int, int] = (800, 800)):
    # draw angles with minimum separation
    angle1 = draw(st.floats(min_value=0.0, max_value=180, exclude_max=True))
    offset = draw(st.floats(min_value=20, max_value=160))
    angle2 = (angle1 + offset) % 180

    # draw beam properties
    width1 = draw(st.floats(min_value=3.0, max_value=25.0))
    width2 = draw(st.floats(min_value=3.0, max_value=25.0))
    intensity1 = draw(st.floats(min_value=0.1, max_value=1.0))
    intensity2 = draw(st.floats(min_value=0.1, max_value=1.0))

    # draw dtype
    dtype = draw(st.sampled_from(["uint8", "uint16", "float32", "float64"]))

    # distace
    distance1 = draw(st.floats(min_value=0.0, max_value=min(dimension) / 2.0))
    distance2 = draw(st.floats(min_value=0.0, max_value=min(dimension) / 2.0))

    # build config
    line1 = ComplexHessLine.from_degrees(angle_deg=angle1, distance=distance1)
    line2 = ComplexHessLine.from_degrees(angle_deg=angle2, distance=distance2)

    config = ImageConfig(
        dimension=ImageDimension(*dimension),
        beams=[
            BeamConfig(
                angle=line1.angle_deg,
                rho=line1.distance,
                width=width1,
                intensity=intensity1,
            ),
            BeamConfig(
                angle=line2.angle_deg,
                rho=line2.distance,
                width=width2,
                intensity=intensity2,
            ),
        ],
        noise=NoiseConfig.none(),
        dtype=dtype,
    )

    image = SyntheticImageGenerator(config).generate()

    # compute expected angles in detector frame
    expected_angles = np.sort((90 - np.array([angle1, angle2])) % 180)

    # compute expected point of intersection
    expected_intersection = np.array(line1.intersect(line2)) + np.array(dimension) / 2
    return GeneratedTestCase(
        config=config,
        image=image,
        expected_angles=expected_angles,
        expected_intersection=expected_intersection[::-1],
        raw_angles=(angle1, angle2),
    )


def is_intersection_inside_image(
    test_case: GeneratedTestCase, margin: float = 25.0
) -> bool:
    margin = max(margin, 0.0)
    return (
        margin < test_case.expected_intersection[0] < test_case.image.shape[0] - margin
        and margin
        < test_case.expected_intersection[1]
        < test_case.image.shape[1] - margin
    )


class TestDetectionMethodKluwe:
    @settings(deadline=2000)
    @given(beam_image_noise_free().filter(is_intersection_inside_image))
    def test_image(self, test_case: GeneratedTestCase):
        detector = Kluwe()
        prediction = detector(test_case.image)
        assert_allclose(prediction, test_case.expected_intersection, rtol=0.02)

    @settings(deadline=2000)
    @given(beam_image_noise_free().filter(is_intersection_inside_image))
    def test_angles_guess(self, test_case: GeneratedTestCase):
        """Test if the initial guess for angle estimation is accurate."""
        detector = Kluwe()
        guess = np.array(detector.guess_angles(test_case.image))
        prediction = np.sort(guess % 180)
        # absolute tolerance of 2 degree
        assert_allclose(
            np.cos(np.deg2rad(2 * prediction)),
            np.cos(np.deg2rad(2 * test_case.expected_angles)),
            atol=2,
        )
