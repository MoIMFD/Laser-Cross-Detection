import numpy as np


class DLT:  # Direct Linear Transformation
    @staticmethod
    def compute(xyz, uv, nd: int = 3):
        """
        Camera calibration by DLT using known object points and their image points.

        Adapted from: https://github.com/JD-Canada/Tracker3D/blob/master/DLT.py

        Input
        -----
        nd: dimensions of the object space, 3 here.
        xyz: coordinates in the object 3D space.
        uv: coordinates in the image 2D space.

        The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

        There must be at least 6 calibration points for the 3D DLT.

        Output
        ------
        H: camera projection matrix.
        err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
        """
        if nd != 3:
            raise ValueError("%dD DLT unsupported." % (nd))

        # Converting all variables to numpy array
        xyz = np.asarray(xyz)
        uv = np.asarray(uv)

        # if the matrices are different length, make them same length (in dir -> 0)
        min_len = np.min([xyz.shape[0], uv.shape[0]])  # shortest length only
        xyz = xyz[:min_len, :]
        uv = uv[:min_len, :]

        n = xyz.shape[0]

        # Validating the parameters:
        if uv.shape[0] != n:
            raise ValueError(
                "Object (%d points) and image (%d points) have different number of points."
                % (n, uv.shape[0])
            )

        if xyz.shape[1] != 3:
            raise ValueError(
                "Incorrect number of coordinates (%d) for %dD DLT (it should be %d)."
                % (xyz.shape[1], nd, nd)
            )

        if n < 6:
            raise ValueError(
                "%dD DLT requires at least %d calibration points. Only %d points were entered."
                % (nd, 2 * nd, n)
            )

        # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
        # This is relevant when there is a considerable perspective distortion.
        # Normalization: mean position at origin and mean distance equals to 1 at each direction.
        Txyz, xyzn = DLT.normalization(nd, xyz)
        Tuv, uvn = DLT.normalization(2, uv)

        A = []

        for i in range(n):
            x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

        # Convert A to array
        A = np.asarray(A)

        # Find the 11 parameters:
        U, S, V = np.linalg.svd(A)

        # The parameters are in the last line of Vh and normalize them
        L = V[-1, :] / V[-1, -1]
        # print(f"L\n{L}\n")
        # Camera projection matrix
        H = L.reshape(3, nd + 1)
        # print(f"H\n{H}\n")

        # Denormalization
        # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
        H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
        # print(f"H\n{H}\n")
        H = H / H[-1, -1]
        # print(f"H\n{H}\n")
        L = H.flatten()
        # print(f"L\n{L}\n")

        # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
        uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
        uv2 = uv2 / uv2[2, :]
        # Mean distance:
        err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))
        err_list = np.sum((uv2[0:2, :].T - uv) ** 2, 1)

        return H, err, err_list

    @staticmethod
    def compute_multiple_cams(c3d, c2d):
        """Camera calibration for multiple cameras by DLT using known object points and their image points.

        Parameters
        ----------
        c3d : array
            coordinates in the object 3D space. n x 3 matrix with n points
        c2d : array
            coordinates in the image 2D space. n x 2*c matrix with n points and c cams

        Returns
        -------
        Hs : list
            list of camera calibration matrices according to cam num in c2d.
        errs : list
            list of errors of the DLT (mean residual of the DLT transformation in units of camera coordinates)
            according to cam num in c2d.
        """
        Hs, errs, errs_lists = [], [], []
        numCams = int(c2d.shape[1] / 2)
        min_len = np.min([c3d.shape[0], c2d.shape[0]])  # shortest length only
        for i in range(numCams):
            _Hs, _errs, _errs_list = DLT.compute(
                c3d[:min_len, :], c2d[:min_len, i * 2 : i * 2 + 2]
            )
            Hs.append(_Hs)
            errs.append(_errs)
            errs_lists.append(_errs_list)
        return Hs, errs, errs_lists

    # implemented by https://github.com/acvictor/DLT/blob/master/DLT.py
    @classmethod
    def normalization(self, nd, x):
        """
        Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

        Input
        -----
        nd: number of dimensions, 3 here
        x: the data to be normalized (directions at different columns and points at rows)
        Output
        ------
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
        """

        x = np.asarray(x)
        m, s = np.mean(x, 0), np.std(x)
        if nd == 2:
            Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = np.array(
                [
                    [s, 0, 0, m[0]],
                    [0, s, 0, m[1]],
                    [0, 0, s, m[2]],
                    [0, 0, 0, 1],
                ]
            )

        Tr = np.linalg.inv(Tr)
        x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
        x = x[0:nd, :].T

        return Tr, x
