# simulation/controller.py

import numpy as np

class Controller:
    def __init__(self, robot):
        """Initialize the controller with a reference to the robot."""
        self.robot = robot

    def compute(self, global_path):
        """
        Compute the velocity and orientation (theta) needed to move toward the first waypoint
        in the global path.
        """
        if not global_path or len(global_path[0]) < 2:
            # No path or only one point (just the current position)
            return 0.0, self.robot.theta

        # Get the first waypoint (skip index 0 because it's the current position)
        target_x = global_path[0][1]
        target_y = global_path[1][1]
        current_x, current_y = self.robot.position

        # Desired angle to the target
        theta = np.arctan2(target_y - current_y, target_x - current_x)

        # Difference between current and target orientation
        angle_diff = np.arctan2(np.sin(theta - self.robot.theta), np.cos(theta - self.robot.theta))

        # Distance to the waypoint
        distance = np.linalg.norm([target_x - current_x, target_y - current_y])

        # Direction and velocity logic
        if abs(angle_diff) > np.pi / 2:
            velocity = -self.robot.max_vel
            theta += np.pi  # Flip orientation
        else:
            velocity = self.robot.max_vel

        # Normalize theta between [-π, π]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        # Stop if very close to the waypoint
        if distance < 0.5:
            velocity = 0.0

        return velocity, theta
