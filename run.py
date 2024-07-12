from laser_cross_detection.core import Kluwe, Ransac, Hough
from laser_cross_detection.test import make_noisefree_image
import matplotlib.pyplot as plt


if __name__ == "__main__":
    methods = dict(kluwe=Kluwe(), ransac=Ransac(), hough=Hough())

    image = make_noisefree_image(
        width=500,
        height=500,
        beam_width=12,
        angle1=45,
        rho1=0,
        angle2=135,
        rho2=0,
    )

    print("Created test image with intersection point at (250, 250)")

    print("Testing Methods...\n")

    fig, ax = plt.subplots()
    ax.imshow(image)
    for name, method in methods.items():
        intersection = method(image)
        ax.scatter(*intersection, label=name)
        print(f"\t{name} - {intersection}")

    ax.legend()
    plt.show()
    print("\nDone")
