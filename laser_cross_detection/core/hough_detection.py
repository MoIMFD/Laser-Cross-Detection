class Hough:
    def compute(self, arr, *args, **kwargs):
        arr_copy = np.copy(arr)
        arr = self.__preprocess(arr=arr)
        lines = probabilistic_hough_line(arr, threshold=100)
        lines_array = np.array(lines)

        angles = []
        for line in lines:
            p0, p1 = line
            d_x, d_y = p1[0] - p0[0], p1[1] - p0[1]
            angles.append(atan2(d_y, d_x))
        angles = np.array(angles)
        # angles, close to pi, get pi subtracted to be close to 0
        # -> that is a problem when there are horizontal lines, as these can either be 0 or pi
        angles[
            np.logical_and(
                angles > pi - pi / 180 * 5, angles < pi + pi / 180 * 5
            )
        ] -= pi
        angles = np.abs(angles)
        div_value = (np.min(angles) + np.max(angles)) / 2

        lines_1 = lines_array[angles < div_value]
        lines_2 = lines_array[angles >= div_value]

        intrsctn_pnts = []
        for line_1 in lines_1:
            for line_2 in lines_2:
                x1, y1 = line_1[0]
                x2, y2 = line_1[1]
                x3, y3 = line_2[0]
                x4, y4 = line_2[1]
                # px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                # py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                #        # source: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                px, py = self.__calc_intersection(
                    x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4
                )
                intrsctn_pnts.append([px, py])

        intrsctn_pnts = np.array(intrsctn_pnts, dtype=np.float64)

        # self.__plot(arr=arr, lines=lines, intersection_point=np.mean(intrsctn_pnts, axis=0))

        hor = np.mean(intrsctn_pnts, axis=0)[0]
        ver = np.mean(intrsctn_pnts, axis=0)[1]

        if (
            np.isnan(hor)
            or np.isnan(ver)
            or not CommonFunctions.test_reasonable_position(
                arr=arr_copy, x=hor, y=ver
            )
        ):
            hor, ver = -9, -9  # is removed later in the core routines
        return hor, ver

    @classmethod
    def __calc_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Calculate the slope of the first line

        return calculate_intersection_of_lines(
            (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        )

    @classmethod
    def __preprocess(self, arr):
        blur = cv.GaussianBlur(arr, (5, 5), 0)
        _, arr = cv.threshold(
            np.array(blur, dtype=np.uint16),
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        arr = arr.astype(bool)
        return arr

    # @classmethod
    def __plot(
        self,
        arr,
        lines,
        intersection_point=None,
        figsize=(8, 5),
        colormap="viridis",
    ):
        fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

        ax.imshow(arr, cmap=colormap)
        for line in lines:
            p0, p1 = line
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]))
        if intersection_point is not None:
            ax.scatter(
                intersection_point[0],
                intersection_point[1],
                c="black",
                zorder=999,
            )
        ax.set_xlim((0, arr.shape[1]))
        ax.set_ylim((arr.shape[0], 0))
        plt.show()
