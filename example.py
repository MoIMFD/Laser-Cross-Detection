import laser_cross_detection as lcd
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from time import time

beam_parameter = namedtuple("BeamParameter", "width rho theta")

if __name__ == "__main__":
    beam1 = beam_parameter(12, 50, 125)  # beam width, rho, beam angle
    beam2 = beam_parameter(27, 100, 27)  # beam width, rho, beam angle

    image = lcd.test.make_noisy_image(
        width=1200,
        height=800,
        beam_width1=beam1.width,
        rho1=beam1.rho,
        angle1=beam1.theta,
        beam_width2=beam2.width,
        rho2=beam2.rho,
        angle2=beam2.theta,
        add_threshold=0.8,
    ).T

    intersection = lcd.test.solve_for_intersection(
        rho1=beam1.rho,
        theta1=beam1.theta,
        rho2=beam2.rho,
        theta2=beam2.theta,
        offset=(600, 400),
    )
    print(
        f"Created test image with intersection point at ({intersection[0]:.4f}, {intersection[1]:.4f})"
    )

    template_center, template_offset = np.divmod(intersection, 1)
    window = 50
    template = image[
        int(template_center[1])
        - window : int(template_center[1])
        + window
        + 1,
        int(template_center[0])
        - window : int(template_center[0])
        + window
        + 1,
    ]

    methods = dict(
        kluwe=lcd.core.Kluwe(),
        ransac=lcd.core.Ransac(),
        hough=lcd.core.Hough(),
        gunady=lcd.core.Gunady(),
        template_matching=lcd.core.TemplateMatching(
            template=template, intersec_offset=template_offset
        ),
    )

    print("Testing Methods...\n")

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.axis(False)
    colors = "red orange green blue cyan".split()
    for i, (name, method) in enumerate(methods.items()):
        tic = time()
        if name == "gunady":
            detected_intersection = method(
                image,
                p01=(beam1.rho, beam1.width, beam1.theta, 255),
                p02=(beam2.rho, beam2.width, beam2.theta, 255),
                threshold=100,
            )
        else:
            detected_intersection = method(image)
        toc = time()

        ax.scatter(
            *detected_intersection,
            label=name,
            fc="none",
            ec=colors[i],
            s=(i + 1) * 100,
        )
        print(
            f"\t{name:<18} - "
            + f"({detected_intersection[0]:.4f}, {detected_intersection[1]:.4f}) - "
            + f"error: {np.linalg.norm(np.subtract(detected_intersection, intersection)):.4f} pixel - "
            + f"took {toc - tic:.4f} ms"
        )

    ax.legend()
    plt.show()
    print("\nDone")
