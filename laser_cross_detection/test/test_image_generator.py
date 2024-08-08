import numpy as np
import cv2

from dataclasses import dataclass
from collections import namedtuple
from functools import lru_cache
from perlin_numpy import perlin2d


from ..utils.image_utils import ImageDimension


def solve_for_intersection(rho1, theta1, rho2, theta2, offset=(0, 0)):
    """calculates the inter section of two lines given by rho and theta"""
    if np.isclose(theta1, theta2):
        return np.nan, np.nan
    theta1, theta2 = np.deg2rad([theta1, theta2])
    A = np.array(
        [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
    )
    b = np.array([rho1, rho2])
    return np.linalg.solve(A, b) + offset


def gaussian(x, x0, width):
    return np.exp(
        -((x - x0) ** 2) / ((width / 6) ** 2)
    )  # width is divided by 6 sigma to obtain pixel values


def salt_and_pepper_noise(image, s_vs_p, amount, scale=1.0):
    num_salt = np.ceil(amount * image.size * s_vs_p)
    salt_idx = np.random.randint(0, image.size, int(num_salt)).astype(bool)
    salt_idx = np.unravel_index(salt_idx, image.shape)

    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    pepper_idx = np.random.randint(0, image.size, int(num_pepper)).astype(bool)
    pepper_idx = np.unravel_index(pepper_idx, image.shape)

    image = image.copy()
    image[salt_idx] = scale
    image[pepper_idx] = 0
    return image


def make_beam_image(
    width: int,
    height: int,
    theta: float,
    rho: float,
    beam_width: float,
    scale: float = 1.0,
):
    """Creates an image of with dimension width times height containing a line
    with a gaussian profile. The line is specified in Hess-Normal-Form, angle
    theta and radius rho (distance from center). The width of the beam is
    defined via beam_width. Per default the intensity values of the returned
    image are between 0 and 1, but can be scaled via the scale argument.

    Implementation based on https://doi.org/10.1016/j.jvcir.2013.09.007 Eq. 7
    """
    x, y = np.mgrid[:width, :height]  # build coordinates
    theta = np.deg2rad(theta)
    image = np.exp(
        -(
            (
                (x - width / 2) * np.cos(theta)
                + (y - height / 2) * np.sin(theta)
                - rho
            )
            ** 2
        )
        / ((beam_width / 3) ** 2)
    )
    return scale * (image / image.max())


@dataclass
class BeamImageGenerator:
    dimension: ImageDimension

    def __post_init__(self):
        self.dimension = ImageDimension(*self.dimension)
        self.extend = int(2**0.5 * self.dimension.width // 2 + 1)
        self.extended_dimension = ImageDimension(
            self.dimension.height, self.dimension.width + 2 * self.extend
        )

    @property
    def center(self):
        return self.dimension.height / 2, self.dimension.width / 2

    def make_beam_image(self, angle, rho, beam_width):
        return make_beam_image(
            width=self.dimension.width,
            height=self.dimension.height,
            theta=angle,
            rho=rho,
            beam_width=beam_width,
        )

    def make_crossing_beams(
        self,
        angle1,
        rho1,
        beam_width1,
        angle2,
        rho2,
        beam_width2,
        gaussian_noise_level=0,
        seed=0,
    ):
        np.random.seed(seed)
        beam_image1 = self.make_beam_image(angle1, rho1, beam_width1)
        beam_image2 = self.make_beam_image(angle2, rho2, beam_width2)
        beam_image = np.maximum(beam_image1, beam_image2)
        beam_mask = beam_image > 1e-4
        beam_image[beam_mask] += np.random.normal(
            loc=gaussian_noise_level / 2,
            scale=np.sqrt(gaussian_noise_level * 0.25),
            size=beam_mask.sum(),
        )
        return np.clip(beam_image, 0, np.inf)


def mask_perlin_noise(image, noise, threshold=0.35):
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    mask = np.abs(noise) < threshold
    image[mask] = 0
    return image


def add_perlin_noise(image, noise, threshold=0.6):
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    image[abs(noise) > threshold] = abs(noise[abs(noise) > threshold])
    return image


@lru_cache
def perlin_noise(seed=0, shape=(2048, 2048), res=(64, 64), octaves=5):
    np.random.seed(seed)
    return perlin2d.generate_fractal_noise_2d(shape, res, octaves)


def make_noisy_image(
    width,
    height,
    angle1,
    rho1=0,
    beam_width1=1,
    angle2=0,
    rho2=0,
    beam_width2=1,
    beam_nosie=0.05,
    seed=0,
    add_threshold=0.6,
    mask_threshold=0.35,
):
    b = BeamImageGenerator((height, width))
    image = b.make_crossing_beams(
        angle1=angle1,
        rho1=rho1,
        angle2=angle2,
        beam_width1=beam_width1,
        rho2=rho2,
        beam_width2=beam_width2,
        gaussian_noise_level=beam_nosie,
    )
    noise = perlin_noise(seed=seed, res=(256, 256), octaves=3)
    image = add_perlin_noise(image, noise, threshold=add_threshold)
    noise = perlin_noise(seed=seed, res=(256, 256), octaves=3)
    image = mask_perlin_noise(image, noise, threshold=mask_threshold)
    image = salt_and_pepper_noise(
        image,
        0.5,
        0.04,
    )
    image = cv2.GaussianBlur(image, (5, 5), 1.5, 1.5)
    return (image / image.max() * 255).astype(np.uint8)


def make_noisefree_image(
    width,
    height,
    angle1,
    rho1=0,
    beam_width1=1,
    angle2=0,
    rho2=0,
    beam_width2=2,
):
    b = BeamImageGenerator((height, width))
    image = b.make_crossing_beams(
        angle1=angle1,
        rho1=rho1,
        beam_width1=beam_width1,
        angle2=angle2,
        rho2=rho2,
        beam_width2=beam_width2,
    )
    return (image / image.max() * 255).astype(np.uint8)
