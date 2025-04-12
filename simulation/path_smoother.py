import numpy as np
from scipy.interpolate import CubicSpline

class PathSmoother:
    def __init__(self, resolution=0.4):
        self.resolution = resolution  # spacing between interpolated points

    def smooth(self, path_x, path_y):
        if len(path_x) < 2:
            return path_x, path_y

        # Use arc-length parameterization
        points = np.array(list(zip(path_x, path_y)))
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        t = np.insert(np.cumsum(distances), 0, 0)

        # Normalize t to [0, 1]
        t /= t[-1]

        # Cubic splines for x(t) and y(t)
        cs_x = CubicSpline(t, path_x)
        cs_y = CubicSpline(t, path_y)

        # Combine original t with interpolated values to preserve all waypoints
        t_interp = np.linspace(0, 1, int(1 / self.resolution * len(path_x)))
        t_combined = np.union1d(t, t_interp)  # ensures original points are included

        # Evaluate smoothed path
        smooth_x = cs_x(t_combined)
        smooth_y = cs_y(t_combined)

        return smooth_x.tolist(), smooth_y.tolist()