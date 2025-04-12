import logging
from simulation.point_robot_mpc import PointRobotMPC

logger = logging.getLogger("MPCController")

class MPCController:
    def __init__(self, robot_radius, horizon=15, dt=0.1):
        self.robot_radius = robot_radius
        self.mpc = PointRobotMPC(N=horizon, dt=dt)

    def get_control(self, state, path_x, path_y, other_robots):
        """
        Compute optimal control given current state and path.

        Args:
            state (tuple): Current robot state (x, y, theta)
            path_x (list): Smoothed x positions of global path
            path_y (list): Smoothed y positions of global path
            other_robots (dict): Dictionary of other robots' states

        Returns:
            v (float): Linear velocity
            omega (float): Angular velocity
        """
        if len(path_x) < 2:
            logger.warning("Insufficient path points for MPC")
            return 0.0, 0.0

        x, y, theta = state
        max_points = self.mpc.N
        ref_path_x = path_x[:max_points]
        ref_path_y = path_y[:max_points]

        v, omega = self.mpc.solve(x, y, theta, ref_path_x, ref_path_y)
        return v, omega
