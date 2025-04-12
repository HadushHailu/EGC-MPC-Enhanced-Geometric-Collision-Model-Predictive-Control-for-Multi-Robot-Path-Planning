import numpy as np
from scipy.interpolate import CubicSpline

class PathSmoother:
    def __init__(self, num_points=500):
        self.num_points = num_points  # Number of interpolated points

    def smooth(self, path):
        """
        Smooths a path using cubic splines.

        Parameters:
            path (List[Tuple[float, float]]): The input path as (x, y) points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The smoothed path (x_smooth, y_smooth).
        """
        if len(path) < 3:
            x_path, y_path = zip(*path)
            return np.array(x_path), np.array(y_path)  # Not enough points to smooth

        path = np.array(path)
        x_path = path[:, 0]
        y_path = path[:, 1]

        t = np.linspace(0, 1, len(x_path))
        t_smooth = np.linspace(0, 1, self.num_points)
        spline_x = CubicSpline(t, x_path)
        spline_y = CubicSpline(t, y_path)

        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        return list(zip(x_smooth, y_smooth))
