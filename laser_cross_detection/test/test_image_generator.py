from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import numpy.typing as nptyping
from perlin_numpy import perlin2d

from laser_cross_detection.utils.image_utils import ImageDimension


def solve_for_intersection(
    rho1: float,
    theta1: float,
    rho2: float,
    theta2: float,
    offset: tuple[float, float] = (0, 0),
) -> nptyping.NDArray:
    """Solves the linear system of equations

        cos(theta1) x + sin(theta1) y = rho1
        cos(theta2) x + sin(theta2) y = rho2

        to find the intersection of both lines.

        If used in context of laser cross images, the center of the image
        needs to be passed as offset.

    Args:
        rho1 (float): radius of first line
        theta1 (float): angle of first line
        rho2 (float): radius of second line
        theta2 (float): angle of second line
        offset (Tuple[float, float], optional): Offset to add to the result.
            Defaults to (0, 0).

    Returns:
        nptyping.NDArray[float, float]: x and y coordinate of the point of intersection
    """

    # checks if lines are parallel and returns (np.nan, np.nan) in this case
    if np.isclose(theta1, theta2):
        return np.array([np.nan, np.nan])
    theta1, theta2 = np.deg2rad([theta1, theta2])
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.solve(A, b) + offset


def salt_and_pepper_noise(
    image: nptyping.NDArray, s_vs_p: float, amount: float, scale: float = 1.0
) -> nptyping.NDArray:
    """Adds salt and pepper noise to an image.

    Args:
        image (nptyping.NDArray): initial image
        s_vs_p (float): ratio between salt and pepper
        amount (float): total amount of pixels to alter
        scale (float, optional): pixel value set to salt pixels (white).
            Defaults to 1.0.

    Returns:
        nptyping.NDArray: copy of image with added salt and pepper noise
    """
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
) -> nptyping.NDArray[np.float64]:
    """Creates an image of with dimension width times height containing a line
    with a gaussian profile. The line is specified in Hess-Normal-Form, angle
    theta and radius rho (distance from center). The width of the beam is
    defined via beam_width. Per default the intensity values of the returned
    image are between 0 and 1, but can be scaled via the scale argument.

    Implementation based on https://doi.org/10.1016/j.jvcir.2013.09.007 Eq. 7

    Args:
        width (int): width of the gaussian shaped beam
        height (int): height of the image in pixel
        theta (float): angle of the beam
        rho (float): distance of the beam from the image center
        beam_width (float): width of the gaussian shaped beam
        scale (float, optional): value to scale the final image which is
            in the range [0, 1]. Defaults to 1.0.

    Returns:
        nptyping.NDArray[np.float64]: image with gaussian beam
    """
    x, y = np.mgrid[:width, :height]  # build coordinates
    theta = np.deg2rad(theta)
    image = np.exp(
        -(
            ((x - width / 2) * np.cos(theta) + (y - height / 2) * np.sin(theta) - rho)
            ** 2
        )
        / ((beam_width / 3) ** 2)
    )
    return scale * (image / image.max())


@dataclass
class BeamImageGenerator:
    """Wrapper class for creating images with two gaussian beams

    Returns:
        _type_: BeamImageGenerator
    """

    dimension: ImageDimension

    def __post_init__(self):
        self.dimension = ImageDimension(*self.dimension)

    @property
    def center(self) -> tuple[float, float]:
        """Returns the center of the image

        Returns:
            Tuple[float, float]: center of the image
        """
        return self.dimension.height / 2, self.dimension.width / 2

    def make_beam_image(
        self, angle: float, rho: float, beam_width: float
    ) -> nptyping.NDArray:
        """Creates an image of a single Gaussian beam

        Args:
            angle (float): angle of the beam in degree
            rho (float): distance from the image center in pixel
            beam_width (float): width of the beam in pixel

        Returns:
            nptyping.NDArray: image with single Gaussian beam
        """
        return make_beam_image(
            width=self.dimension.width,
            height=self.dimension.height,
            theta=angle,
            rho=rho,
            beam_width=beam_width,
        )

    def make_crossing_beams(
        self,
        angle1: float,
        rho1: float,
        beam_width1: float,
        angle2: float,
        rho2: float,
        beam_width2: float,
        gaussian_noise_level: float = 0.0,
        seed: int = 0,
    ) -> nptyping.NDArray:
        """Creates an image with two Gaussian beams. First a single image for
        each beam is generated. Both images are combined by taking the max
        value for each pixel using np.maximum.

        Args:
            angle1 (float): angle of the first beam in degree
            rho1 (float): distance from the center for the first beam in pixel
            beam_width1 (float): width of the first beam in pixel
            angle2 (float): angle of the second beam in degree
            rho2 (float): distance from the center for the second beam in pixel
            beam_width2 (float): width of the second beam in pixel
            gaussian_noise_level (float, optional): amount of noise used to
                alter the gaussian profiles of the beams. Defaults to 0.0.
            seed (int, optional): seed for random number generator.
                Defaults to 0.

        Returns:
            nptyping.NDArray: image with two gaussian beams
        """
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


def mask_perlin_noise(
    image: nptyping.NDArray, noise: nptyping.NDArray, threshold: float = 0.35
) -> nptyping.NDArray:
    """Masks an image using a 2d Perlin noise array. The mask is defined
    by thresholding the absolute value of the Perlin noise.

    Args:
        image (nptyping.NDArray): image to mask
        noise (nptyping.NDArray): 2d Perlin noise array
        threshold (float, optional): threshold value to select mask regions
            from Perlin noise array. Defaults to 0.35.

    Returns:
        nptyping.NDArray: copy of image with mask applied
    """
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    mask = np.abs(noise) < threshold
    image[mask] = 0
    return image


def add_perlin_noise(
    image: nptyping.NDArray, noise: nptyping.NDArray, threshold: float = 0.6
) -> nptyping.NDArray:
    """Adds Perlin noise to an image based on mask created via thresholding the
    noise at a certain value.

    Args:
        image (nptyping.NDArray): image to add noise to
        noise (nptyping.NDArray): 2d Perlin noise array
        threshold (float, optional): threshold value to select mask regions
            from Perlin noise array. Defaults to 0.6.

    Returns:
        nptyping.NDArray: copy of image with added noise
    """
    image = image.copy()
    noise = noise[: image.shape[0], : image.shape[1]]
    image[abs(noise) > threshold] = abs(noise[abs(noise) > threshold])
    return image


@lru_cache
def perlin_noise(
    seed: int = 0,
    shape: tuple[int, int] = (2048, 2048),
    res: tuple[int, int] = (64, 64),
    octaves: int = 5,
) -> nptyping.NDArray[np.float64]:
    """Creates an image containing Perlin noise.

    Args:
        seed (int, optional): random seed used for numpy random module.
            Defaults to 0.
        shape (Tuple[int, int], optional): size of the returned Perlin noise
            array. Defaults to (2048, 2048).
        res (Tuple[int, int], optional): resolution of the Perlin noise.
            Defaults to (64, 64).
        octaves (int, optional): number of octaves in Perlin noise.
            Defaults to 5.

    Returns:
        nptyping.NDArray[np.float64]: 2d array of Perlin noise
    """
    np.random.seed(seed)
    return perlin2d.generate_fractal_noise_2d(shape, res, octaves)


def make_noisy_image(
    width: int,
    height: int,
    angle1: float = 0,
    rho1: float = 0,
    beam_width1: float = 1,
    angle2: float = 90,
    rho2: float = 0,
    beam_width2: float = 1,
    beam_noise: float = 0.05,
    seed: int = 0,
    add_threshold: float = 0.6,
    mask_threshold: float = 0.35,
) -> nptyping.NDArray[np.uint8]:
    """Creates a noisy image of two gaussian beams.

    Args:
        width (int): width of the returned image
        height (int): height of the returned image
        angle1 (float): angle of the first beam in degrees. Defaults to 0.
        rho1 (float, optional): distance from the image center of the first
            beam in pixel. Defaults to 0.
        beam_width1 (float, optional): width of the first beam in pixel.
            Defaults to 1.
        angle2 (float, optional): angle of the second beam in degrees.
            Defaults to 90.
        rho2 (float, optional):  distance from the image center of the second
            beam in pixel. Defaults to 0.
        beam_width2 (float, optional): width of the second beam in pixel.
            Defaults to 1.
        beam_noise (float, optional): noise altering the gaussian shape of the
            beam profiles. Defaults to 0.05.
        seed (int, optional): seed for random number generator. Defaults to 0.
        add_threshold (float, optional): threshold used for adding Perlin
            noise. Defaults to 0.6.
        mask_threshold (float, optional): threshold used for masking with
            Perlin noise. Defaults to 0.35.

    Returns:
        nptyping.NDArray[np.uint8]: Noisy image with to gaussian beams
    """
    b = BeamImageGenerator(ImageDimension(height, width))
    image = b.make_crossing_beams(
        angle1=angle1,
        rho1=rho1,
        angle2=angle2,
        beam_width1=beam_width1,
        rho2=rho2,
        beam_width2=beam_width2,
        gaussian_noise_level=beam_noise,
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
    width: int,
    height: int,
    angle1: float = 0,
    rho1: float = 0,
    beam_width1: float = 1,
    angle2: float = 90,
    rho2: float = 0,
    beam_width2: float = 1,
) -> nptyping.NDArray[np.uint8]:
    """Creates a noise free image of two gaussian beams

    Args:
        width (int):  width of the returned image
        height (int):  height of the returned image
        angle1 (float): angle of the first beam in degrees. Defaults to 0.
        rho1 (float, optional): distance from the image center of the first
            beam in pixel. Defaults to 0.
        beam_width1 (float, optional):  width of the first beam in pixel.
            Defaults to 1.
        angle2 (flaot, optional): angle of the second beam in degrees.
            Defaults to 90.
        rho2 (float, optional): distance from the image center of the second
            beam in pixel. Defaults to 0.
        beam_width2 (float, optional):  width of the second beam in pixel.
            Defaults to 1.

    Returns:
        nptyping.NDArray[np.uint8]: noise free image of two gaussian beams
    """
    b = BeamImageGenerator(ImageDimension(height, width))
    image = b.make_crossing_beams(
        angle1=angle1,
        rho1=rho1,
        beam_width1=beam_width1,
        angle2=angle2,
        rho2=rho2,
        beam_width2=beam_width2,
    )
    return (image / image.max() * 255).astype(np.uint8)
