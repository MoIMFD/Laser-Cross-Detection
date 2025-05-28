import lmfit
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nptyping
import skimage as ski
from matplotlib.widgets import RectangleSelector

from .detection_abc import DetectionMethodABC


class TemplateMatching(DetectionMethodABC):
    def __init__(
        self, template: nptyping.NDArray, intersec_offset: tuple[float, float]
    ) -> None:
        """Detecting the intersection of two light beams in an image based
        on template matching.

        Args:
            template (nptyping.NDArray): template of the intersection
            intersec_offset (Tuple[float, float]): location of the intersection
                within the template
        """
        super().__init__()
        self.template = template
        self.intersec_offset = intersec_offset

    def __call__(
        self,
        image: nptyping.NDArray,
        *args,
        fit_window: int | None = None,
        **kwargs,
    ) -> nptyping.NDArray[np.float64]:
        """Uses the specified template to identify the intersection of two
        light beams in a new image. The process consists of the generation
        of a correlation map, finding the discrete maximum, extracting a
        quadratic sub map with size fit_window around the maximum, fitting
        a 2d gaussian to the sub map and returning the center of the fitted
        2d gaussian distribution.

        Args:
            image (nptyping.NDArray): _description_
            fit_window ( int | None, optional): Number of pixels used for
                the 2d window to fit a two dimensional gaussian to the maximum
                of the correlation map. Defaults to None which uses
                max(width, height) / 10 as the fit_window.

        Returns:
            nptyping.NDArray[np.float64]: detected coordinates of the
                intersection
        """
        image = image.copy()
        result = ski.feature.match_template(image=image, template=self.template)

        max_index = np.unravel_index(np.argmax(result), shape=result.shape)

        if fit_window is None:
            fit_window = max(image.shape) // 25

        x_slice = slice(max_index[0] - fit_window, max_index[0] + fit_window + 1)
        x = np.mgrid[x_slice]
        y_slice = slice(max_index[1] - fit_window, max_index[1] + fit_window + 1)
        y = np.mgrid[y_slice]
        result_section = result[x_slice, y_slice]

        xx, yy = np.meshgrid(x, y)
        model = lmfit.models.Gaussian2dModel()
        params = model.guess(data=result_section.ravel(), y=xx.ravel(), x=yy.ravel())
        fit_result = model.fit(
            data=result_section.ravel(),
            x=xx.ravel(),
            y=yy.ravel(),
            params=params,
        )

        return (
            np.add(
                (
                    fit_result.best_values["centerx"],
                    fit_result.best_values["centery"],
                ),
                self.intersec_offset,
            )
            + self.half_template_shape
        )

    @property
    def half_template_shape(self):
        return np.divide(self.template.shape, 2)

    def update_template(
        self, template: nptyping.NDArray, intersec_offset: tuple[float, float]
    ):
        """Set a new template and offset

        Args:
            template (nptyping.NDArray): template of the intersection
            intersec_offset (Tuple[float, float]): location of the intersection
                within the template

        Returns:
            _type_: self
        """
        self.template = template
        self.intersec_offset = intersec_offset
        return self

    @staticmethod
    def select_template(
        ref_image: nptyping.NDArray,
    ) -> tuple[nptyping.NDArray, float, float]:
        """Static method to interactively select a template and the offset
        using matplotlib widgets.

        Args:
            ref_image (nptyping.NDArray): image to select the template from

        Returns:
            Tuple[nptyping.NDArray, float, float]: selected template and x, y
                coordinates of the intersection
        """
        fig, ax = plt.subplots()
        ax.imshow(ref_image, cmap="gray")

        so = SelectorObject()
        _selector = RectangleSelector(
            ax=ax, onselect=so.__call__, ignore_event_outside=True
        )
        plt.show(block=True)

        template = ref_image[so.yslice, so.xslice]
        fig, ax = plt.subplots()

        co = CenterObject()
        ax.imshow(template, cmap="gray")

        _cid = fig.canvas.mpl_connect("button_press_event", co.__call__)

        plt.show(block=True)

        return template, co.x, co.y


class SelectorObject:
    """Object to store result from matplotlib RectangleSelector"""

    def __init__(self):
        self.p1 = (None, None)
        self.p2 = (None, None)

    def __call__(self, eclick, erelease):
        self.p1 = int(eclick.xdata), int(eclick.ydata)
        self.p2 = int(erelease.xdata), int(erelease.ydata)
        plt.close()

    @property
    def xslice(self):
        x0, x1 = sorted((self.p1[0], self.p2[0]))
        return slice(x0, x1)

    @property
    def yslice(self):
        y0, y1 = sorted((self.p1[1], self.p2[1]))
        return slice(y0, y1)


class CenterObject:
    """Objetc to store result from interactive center selection"""

    def __init__(self):
        self.x = None
        self.y = None
        self.scatter = plt.scatter(None, None)

    def __call__(self, event):
        self.x = event.xdata
        self.y = event.ydata

        self.scatter.remove()
        self.scatter = plt.scatter(self.x, self.y, c="red", alpha=0.3)
        plt.gcf().canvas.draw_idle()
