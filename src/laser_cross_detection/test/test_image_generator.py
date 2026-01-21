from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from perlin_numpy import perlin2d

from laser_cross_detection.utils.image_utils import ImageDimension

OutputDType = Literal["uint8", "uint16", "float32", "float64"]


@dataclass(frozen=True)
class BeamConfig:
    """Configuration for a single Gaussian beam.

    Attributes:
        angle: Angle of the beam normal in degrees (0-360).
        rho: Distance from the image center in pixels.
        width: Width of the Gaussian beam profile in pixels.
        intensity: Relative intensity of this beam (0.0-1.0).
    """

    angle: float = 0.0
    rho: float = 0.0
    width: float = 10.0
    intensity: float = 1.0


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Configuration for Gaussian noise added to beam profiles.

    Attributes:
        level: Standard deviation of the Gaussian noise relative to beam intensity.
        mean_factor: Factor to compute mean from level (mean = level * mean_factor).
    """

    level: float = 0.05
    mean_factor: float = 0.5


@dataclass(frozen=True)
class SaltPepperNoiseConfig:
    """Configuration for salt and pepper noise.

    Attributes:
        amount: Fraction of total pixels to affect (0.0-1.0).
        salt_ratio: Ratio of salt (white) to pepper (black) pixels (0.0-1.0).
    """

    amount: float = 0.04
    salt_ratio: float = 0.5


@dataclass(frozen=True)
class PerlinNoiseConfig:
    """Configuration for Perlin noise generation and application.

    Attributes:
        resolution: Resolution tuple for Perlin noise generation.
        octaves: Number of octaves for fractal noise.
        seed: Random seed for Perlin noise generation.
        add_threshold: Threshold for adding Perlin noise to image.
        mask_threshold: Threshold for masking with Perlin noise.
    """

    resolution: tuple[int, int] = (64, 64)
    octaves: int = 3
    seed: int = 0
    add_threshold: float = 0.6
    mask_threshold: float = 0.35


@dataclass(frozen=True)
class GaussianBlurConfig:
    """Configuration for Gaussian blur post-processing.

    Attributes:
        kernel_size: Size of the blur kernel (must be odd).
        sigma_x: Standard deviation in X direction.
        sigma_y: Standard deviation in Y direction.
    """

    kernel_size: tuple[int, int] = (5, 5)
    sigma_x: float = 1.5
    sigma_y: float = 1.5


@dataclass
class NoiseConfig:
    """Composite configuration for all noise types.

    Set any noise config to None to disable that noise type.
    """

    gaussian: GaussianNoiseConfig | None = None
    salt_pepper: SaltPepperNoiseConfig | None = None
    perlin: PerlinNoiseConfig | None = None
    blur: GaussianBlurConfig | None = None

    @classmethod
    def none(cls) -> NoiseConfig:
        """Create a noise-free configuration."""
        return cls(gaussian=None, salt_pepper=None, perlin=None, blur=None)

    @classmethod
    def light(cls) -> NoiseConfig:
        """Create a light noise configuration for testing."""
        return cls(
            gaussian=GaussianNoiseConfig(level=0.02),
            salt_pepper=SaltPepperNoiseConfig(amount=0.01),
            perlin=None,
            blur=GaussianBlurConfig(kernel_size=(3, 3), sigma_x=1.0, sigma_y=1.0),
        )

    @classmethod
    def heavy(cls) -> NoiseConfig:
        """Create a heavy noise configuration for robustness testing."""
        return cls(
            gaussian=GaussianNoiseConfig(level=0.1),
            salt_pepper=SaltPepperNoiseConfig(amount=0.08),
            perlin=PerlinNoiseConfig(add_threshold=0.5, mask_threshold=0.4),
            blur=GaussianBlurConfig(kernel_size=(7, 7), sigma_x=2.0, sigma_y=2.0),
        )


@dataclass
class ImageConfig:
    """Complete configuration for test image generation.

    Attributes:
        dimension: Image dimensions (height, width).
        beams: List of beam configurations. Supports 1 to N beams.
        noise: Noise configuration. Use NoiseConfig.none() for clean images.
        dtype: Output data type.
        seed: Master random seed for reproducibility.
    """

    dimension: ImageDimension = field(default_factory=lambda: ImageDimension(512, 512))
    beams: list[BeamConfig] = field(
        default_factory=lambda: [
            BeamConfig(angle=0.0, width=10.0),
            BeamConfig(angle=90.0, width=10.0),
        ]
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig.none)
    dtype: OutputDType = "uint8"
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.dimension, tuple):
            self.dimension = ImageDimension(*self.dimension)
        if not self.beams:
            raise ValueError("At least one beam configuration is required")


class NoiseLayer(ABC):
    """Abstract base class for composable noise layers."""

    @abstractmethod
    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        """Apply noise transformation to image.

        Args:
            image: Input image as float64 array normalized to [0, 1].
            rng: NumPy random generator for reproducibility.

        Returns:
            Transformed image as float64 array.
        """
        pass


class GaussianNoiseLayer(NoiseLayer):
    """Applies Gaussian noise to beam regions."""

    def __init__(self, config: GaussianNoiseConfig, beam_threshold: float = 1e-4):
        self.config = config
        self.beam_threshold = beam_threshold

    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        image = image.copy()
        beam_mask = image > self.beam_threshold
        noise = rng.normal(
            loc=self.config.level * self.config.mean_factor,
            scale=np.sqrt(self.config.level * 0.25),
            size=beam_mask.sum(),
        )
        image[beam_mask] += noise
        return np.clip(image, 0, np.inf)


class SaltPepperNoiseLayer(NoiseLayer):
    """Applies salt and pepper noise."""

    def __init__(self, config: SaltPepperNoiseConfig):
        self.config = config

    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        image = image.copy()
        num_salt = int(
            np.ceil(self.config.amount * image.size * self.config.salt_ratio)
        )
        num_pepper = int(
            np.ceil(self.config.amount * image.size * (1 - self.config.salt_ratio))
        )

        salt_coords = rng.choice(image.size, num_salt, replace=False)
        salt_idx = np.unravel_index(salt_coords, image.shape)
        image[salt_idx] = 1.0

        pepper_coords = rng.choice(image.size, num_pepper, replace=False)
        pepper_idx = np.unravel_index(pepper_coords, image.shape)
        image[pepper_idx] = 0.0

        return image


class PerlinNoiseLayer(NoiseLayer):
    """Applies Perlin noise for masking and additive noise."""

    def __init__(
        self, config: PerlinNoiseConfig, cache_shape: tuple[int, int] = (2048, 2048)
    ):
        self.config = config
        self.cache_shape = cache_shape

    def _generate_noise(self) -> NDArray:
        old_state = np.random.get_state()
        np.random.seed(self.config.seed)
        noise = perlin2d.generate_fractal_noise_2d(
            self.cache_shape, self.config.resolution, self.config.octaves
        )
        np.random.set_state(old_state)
        return noise

    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        image = image.copy()
        noise = self._generate_noise()
        noise = noise[: image.shape[0], : image.shape[1]]

        add_mask = np.abs(noise) > self.config.add_threshold
        image[add_mask] = np.abs(noise[add_mask])

        mask = np.abs(noise) < self.config.mask_threshold
        image[mask] = 0

        return image


class GaussianBlurLayer(NoiseLayer):
    """Applies Gaussian blur."""

    def __init__(self, config: GaussianBlurConfig):
        self.config = config

    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        return cv2.GaussianBlur(
            image,
            self.config.kernel_size,
            self.config.sigma_x,
            self.config.sigma_y,
        )


class NoisePipeline:
    """Composable pipeline of noise layers."""

    def __init__(self, layers: list[NoiseLayer] | None = None):
        self.layers: list[NoiseLayer] = layers or []

    def add(self, layer: NoiseLayer) -> NoisePipeline:
        """Add a noise layer to the pipeline. Returns self for chaining."""
        self.layers.append(layer)
        return self

    def apply(self, image: NDArray, rng: np.random.Generator) -> NDArray:
        """Apply all noise layers in sequence."""
        for layer in self.layers:
            image = layer.apply(image, rng)
        return image

    @classmethod
    def from_config(cls, config: NoiseConfig) -> NoisePipeline:
        """Build a pipeline from a NoiseConfig."""
        pipeline = cls()
        if config.gaussian is not None:
            pipeline.add(GaussianNoiseLayer(config.gaussian))
        if config.perlin is not None:
            pipeline.add(PerlinNoiseLayer(config.perlin))
        if config.salt_pepper is not None:
            pipeline.add(SaltPepperNoiseLayer(config.salt_pepper))
        if config.blur is not None:
            pipeline.add(GaussianBlurLayer(config.blur))
        return pipeline


def solve_for_intersection(
    rho1: float,
    theta1: float,
    rho2: float,
    theta2: float,
    offset: tuple[float, float] = (0, 0),
) -> NDArray:
    """Solves the linear system of equations for two lines in Hess normal form.

    Args:
        rho1: Distance of first line from origin.
        theta1: Angle of first line in degrees.
        rho2: Distance of second line from origin.
        theta2: Angle of second line in degrees.
        offset: Offset to add to the result.

    Returns:
        x and y coordinate of the point of intersection.
    """
    if np.isclose(theta1, theta2):
        return np.array([np.nan, np.nan])
    theta1_rad, theta2_rad = np.deg2rad([theta1, theta2])
    A = np.array(
        [
            [np.cos(theta1_rad), np.sin(theta1_rad)],
            [np.cos(theta2_rad), np.sin(theta2_rad)],
        ]
    )
    b = np.array([rho1, rho2])
    return np.linalg.solve(A, b) + offset


class SyntheticImageGenerator:
    """Generates synthetic test images with configurable beams and noise.

    Example:
        generator = SyntheticImageGenerator()
        image = generator.generate()

        config = ImageConfig(
            dimension=ImageDimension(1024, 1024),
            beams=[
                BeamConfig(angle=0, width=15),
                BeamConfig(angle=60, width=15),
                BeamConfig(angle=120, width=15),
            ],
            noise=NoiseConfig.light(),
            dtype="uint16",
        )
        generator = SyntheticImageGenerator(config)
        image = generator.generate()
    """

    def __init__(self, config: ImageConfig | None = None):
        self.config = config or ImageConfig()
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def center(self) -> tuple[float, float]:
        """Image center coordinates (x, y)."""
        return (
            self.config.dimension.width / 2,
            self.config.dimension.height / 2,
        )

    def _make_single_beam(self, beam: BeamConfig) -> NDArray[np.float64]:
        """Generate a single Gaussian beam image."""
        width = self.config.dimension.width
        height = self.config.dimension.height

        x, y = np.mgrid[:width, :height]
        theta = np.deg2rad(beam.angle)

        distance = (
            (x - width / 2) * np.cos(theta)
            + (y - height / 2) * np.sin(theta)
            - beam.rho
        )
        sigma = beam.width / 3
        image = np.exp(-(distance**2) / (sigma**2))

        return beam.intensity * (image / image.max())

    def _combine_beams(self, beams: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Combine multiple beam images using maximum operation."""
        if len(beams) == 1:
            return beams[0]
        return np.maximum.reduce(beams)

    def _convert_dtype(self, image: NDArray[np.float64]) -> NDArray:
        """Convert float64 image to target dtype."""
        if image.max() > 0:
            image = image / image.max()

        dtype_map: dict[OutputDType, tuple[type, float]] = {
            "uint8": (np.uint8, np.iinfo(np.uint8).max),
            "uint16": (np.uint16, np.iinfo(np.uint16).max),
            "float32": (np.float32, 1.0),
            "float64": (np.float64, 1.0),
        }

        dtype, scale = dtype_map[self.config.dtype]
        return (image * scale).astype(dtype)

    def generate(self) -> NDArray:
        """Generate a test image according to configuration.

        Returns:
            Generated image with configured dtype and noise.
        """
        self._rng = np.random.default_rng(self.config.seed)

        beam_images = [self._make_single_beam(beam) for beam in self.config.beams]
        image = self._combine_beams(beam_images)

        pipeline = NoisePipeline.from_config(self.config.noise)
        image = pipeline.apply(image, self._rng)

        return self._convert_dtype(image)

    def generate_clean(self) -> NDArray:
        """Generate a noise-free version of the configured image.

        Returns:
            Clean image without any noise applied.
        """
        beam_images = [self._make_single_beam(beam) for beam in self.config.beams]
        image = self._combine_beams(beam_images)
        return self._convert_dtype(image)

    def get_intersection_points(self) -> list[tuple[float, float]]:
        """Calculate intersection points of all beam pairs.

        Returns:
            List of (x, y) intersection coordinates relative to image origin.
        """
        intersections = []
        beams = self.config.beams
        center = self.center

        for i in range(len(beams)):
            for j in range(i + 1, len(beams)):
                point = solve_for_intersection(
                    beams[i].rho,
                    beams[i].angle,
                    beams[j].rho,
                    beams[j].angle,
                    offset=center,
                )
                if not np.any(np.isnan(point)):
                    intersections.append((point[0], point[1]))

        return intersections
