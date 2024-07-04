import numpy as np
import cv2
from skimage import transform
import matplotlib.pyplot as plt

from dataclasses import dataclass
from collections import namedtuple
from functools import lru_cache
from perlin_numpy import perlin2d


from ..utils import image_utils

ImageDimension = namedtuple("ImageDimension", "height width")


def gaussian(x, x0, width):
    return np.exp(
        -((x - x0) ** 2) / ((width / 6) ** 2)
    )  # width is devided by 6 sigma to obtain pixel values


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


@dataclass
class BeamImageGenerator:
    dimension: ImageDimension
    # profile_function: callable

    def __post_init__(self):
        self.dimension = ImageDimension(*self.dimension)
        assert self.dimension.height % 2 != 0, "dimension must be odd"
        assert self.dimension.width % 2 != 0, "dimension must be odd"
        self.extend = int(2**0.5 * self.dimension.width // 2 + 1)
        self.extended_dimension = ImageDimension(
            self.dimension.height, self.dimension.width + 2 * self.extend
        )

    @property
    def center(self):
        return self.dimension.height // 2, self.dimension.width // 2

    def make_beam_image(self, angle, beam_profile):
        beam_image = beam_profile[:, np.newaxis] * np.ones(
            self.extended_dimension
        )
        if angle == 0:
            return beam_image[:, self.extend : -self.extend]
        else:
            # return rotate_image(beam_image, angle)[
            #     :, self.extend : -self.extend
            # ]
            return image_utils.rotate_image(beam_image, angle, impl="skimage")[
                :, self.extend : -self.extend
            ]

    def make_crossing_beams(
        self,
        angle1,
        beam_profile1,
        angle2,
        beam_profile2,
        gaussian_noise_level=0,
        seed=0,
    ):
        np.random.seed(seed)
        beam_image1 = self.make_beam_image(angle1, beam_profile1)
        beam_image2 = self.make_beam_image(angle2, beam_profile2)
        beam_image = np.maximum(beam_image1, beam_image2)
        beam_mask = beam_image > 1e-4
        beam_image[beam_mask] += np.random.normal(
            loc=gaussian_noise_level / 2,
            scale=np.sqrt(gaussian_noise_level * 0.25),
            size=beam_mask.sum(),
        )
        return np.clip(beam_image, a_min=0, a_max=1)

    def gaussian_beam_profile(self, width, x0=None):
        if not x0:
            x0 = self.dimension.height // 2
        x = np.mgrid[: self.dimension.height]
        return gaussian(x, x0, width)


def mask_perlin_noise(image, noise, threshold=0.8):
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    mask = np.abs(noise) < 0.2
    image[mask] = 0
    return image


def add_perlin_noise(image, noise, threshold=0.8):
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    image[abs(noise) > threshold] = abs(noise[abs(noise) > threshold])
    return image


@lru_cache
def perlin_noise(seed=0, shape=(2048, 2048), res=(64, 64), octaves=5):
    np.random.seed(seed)
    return perlin2d.generate_fractal_noise_2d(shape, res, octaves)
