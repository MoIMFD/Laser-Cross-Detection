import laser_cross_detection as lcd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    methods = dict(
        kluwe=lcd.core.Kluwe(),
        ransac=lcd.core.Ransac(),
        hough=lcd.core.Hough(),
    )

    beam1 = (12, 50, 125)  # beam width, rho, beam angle
    beam2 = (27, 100, 27)  # beam width, rho, beam angle

    image = lcd.test.make_noisy_image(
        width=1200,
        height=800,
        beam_width1=beam1[0],
        rho1=beam1[1],
        angle1=beam1[2],
        beam_width2=beam2[0],
        rho2=beam2[1],
        angle2=beam2[2],
        add_threshold=0.8,
    ).T

    intersection = lcd.test.solve_for_intersection(
        rho1=beam1[1],
        theta1=beam1[2],
        rho2=beam2[1],
        theta2=beam2[2],
        offset=(600, 400),
    )

    print(f"Created test image with intersection point at {intersection}")

    print("Testing Methods...\n")

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.axis(False)
    colors = "red orange green".split()
    for i, (name, method) in enumerate(methods.items()):
        intersection = method(image)
        ax.scatter(
            *intersection, label=name, fc="none", ec=colors[i], s=(i + 1) * 100
        )
        print(f"\t{name} - {intersection}")

    ax.legend()
    plt.show()
    print("\nDone")
