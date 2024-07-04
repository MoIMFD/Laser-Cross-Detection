class TemplateMatching:
    def __init__(self, images):
        # from .core import ImageArray
        # prepare images for choosing the one with the cross in the middle for every view
        images_summed = [[] for _ in range(len(images))]
        for i_pos in range(len(images)):
            for i_cam in range(len(images[0][0])):
                # sum all images of a loop
                imList = [None for _ in range(len(images[0]))]
                for i_loop in range(len(images[0])):
                    imList[i_loop] = images[i_pos][i_loop][i_cam]
                arr = imList[0]
                for i_loop in range(1, len(images[0])):
                    arr = np.maximum(arr, imList[i_loop])
                images_summed[i_pos].append(arr)
        # ImageArray.plot(images_summed[0][0])

        # choose best images by center of weight -> as close as possible to center of image
        best_images_idx = [None for _ in range(len(images_summed[0]))]
        distances = [np.inf for _ in range(len(images_summed[0]))]
        for i_cam in range(len(images_summed[0])):
            for i_pos in range(len(images_summed)):
                arr = images_summed[i_pos][i_cam]
                com = ndimage.center_of_mass(arr)
                d = sqrt(
                    (com[0] - arr.shape[0] / 3) ** 2
                    + (com[1] - arr.shape[1] / 3) ** 2
                )
                if distances[i_cam] > d:
                    distances[i_cam] = d
                    best_images_idx[i_cam] = i_pos
        # ImageArray.plot(images_summed[best_images_idx_byAbs[0]][0])
        # ImageArray.plot(images_summed[best_images_idx_byAbs[1]][1])
        # ImageArray.plot(images_summed[best_images_idx_byAbs[2]][2])

        # display images; let user cut out cross; let user mark center of the cross
        self.templates = [None for _ in range(len(best_images_idx))]
        self.centers = [None for _ in range(len(best_images_idx))]
        for i_cam in range(len(best_images_idx)):
            # user interactive part (template and center selection)
            im = images_summed[best_images_idx[i_cam]][i_cam]
            im = ((im - np.min(im)) / np.max(im) * 255).astype(np.uint8)
            cv.namedWindow("Template Selection", cv.WINDOW_GUI_EXPANDED)
            x0, y0, w0, h0 = cv.selectROI(
                windowName="Template Selection",
                img=im,
                showCrosshair=True,
                fromCenter=True,
            )
            cv.destroyWindow("Template Selection")
            cv.namedWindow("Center Selection", cv.WINDOW_GUI_EXPANDED)
            im = im[y0 : y0 + h0, x0 : x0 + w0]
            x1, y1, w1, h1 = cv.selectROI(
                windowName="Center Selection",
                img=im,
                showCrosshair=True,
                fromCenter=True,
            )
            cv.destroyWindow("Center Selection")
            # save template and center information
            self.centers[i_cam] = [
                x1 + w1 / 2,
                y1 + h1 / 2,
            ]  # in opencv, coordinates are transposed to array nomenclature
            self.templates[i_cam] = im

    def compute(self, arr, *args, **kwargs):  # *args -> idx, ...
        # from .core import ImageArray
        idx = kwargs["i_cam"]
        # irgendso eine scikit funktion -> J. P. Lewis, “Fast Normalized Cross-Correlation”, Industrial Light and Magic.
        mt = match_template(arr, self.templates[idx])
        # ImageArray.plot(mt)

        # max idx
        x_y = np.unravel_index(mt.argmax(), mt.shape)

        # subpixel peak detection: -> lmfit 2d gauss model
        gaussWinSize: int = 12  # px squared -> should be odd number
        x, y, z = ([] for _ in range(3))
        for yy in range(
            x_y[1] - int(gaussWinSize / 2), x_y[1] + int(gaussWinSize / 2) + 1
        ):
            for xx in range(
                x_y[0] - int(gaussWinSize / 2),
                x_y[0] + int(gaussWinSize / 2) + 1,
            ):
                x.append(xx)
                y.append(yy)
                try:
                    z.append(mt[xx, yy])
                except:
                    return None, None
        model = lmfit.models.Gaussian2dModel()
        params = model.guess(z, x, y)
        result = model.fit(z, x=x, y=y, params=params)

        hor = self.centers[idx][0] + result.best_values["centery"]
        ver = self.centers[idx][1] + result.best_values["centerx"]

        if not CommonFunctions.test_reasonable_position(arr=arr, x=hor, y=ver):
            hor, ver = -9, -9  # is removed later in the core routines
        return hor, ver
        # y -> horizontale Komponente, x -> vertikale Komponente
