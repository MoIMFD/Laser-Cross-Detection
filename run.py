from laser_cross_detection.core import Kluwe, Ransac, Hough
from laser_cross_detection.test import make_noisefree_image


if __name__ == "__main__":
    methods = dict(kluwe=Kluwe(), ransac=Ransac(), hough=Hough())

    image = make_noisefree_image(501, 501, 12, 45, -45)
    print("Createdt test image with intersection point at (250, 250)")

    print("Testing Methods...\n")
    for name, method in methods.items():
        intersection = method(image)

        print(f"\t{name} - {intersection}")

    print("\nDone")
