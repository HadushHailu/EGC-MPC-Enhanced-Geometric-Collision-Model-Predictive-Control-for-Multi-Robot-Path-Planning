# simulation/robot.py

import threading
import numpy as np
from simulation.rate import Rate
from simulation.controller import Controller
from simulation.avoidance import CollisionAvoidance

class Robot(threading.Thread):
    def __init__(self, robot_id, config, robot_dim, robot_states, stop_event):
        super().__init__()
        self.robot_id = robot_id
        self.robot_states = robot_states
        self.stop_event = stop_event
        self.position = np.array([config["start"]["x"], config["start"]["y"]], dtype=float)
        self.goal = np.array([config["goal"]["x"], config["goal"]["y"]], dtype=float)
        self.color = config["color"]
        self.max_vel = config["max_vel"]
        self.controller_frequency = config["controller_frequency"]
        self.length = robot_dim["length"]
        self.width = robot_dim["width"]
        self.safety_margin = robot_dim["safety_margin"]
        
        self.velocity = np.array([0.0, 0.0])
        self.theta = 0.0
        self.x_traj = [self.position[0]]
        self.y_traj = [self.position[1]]
        
        self.controller = Controller(self)
        self.avoidance = CollisionAvoidance()

    def run(self):
        rate = Rate(self.controller_frequency)
        while not self.stop_event.is_set()  and not self._reached_goal():
            other_robots = self.robot_states.get_others(self.robot_id)

            # Plan path and compute control
            global_path = self.avoidance.plan(self, other_robots)
            velocity, theta = self.controller.compute(global_path)

            # Update position
            dt = 1.0 / self.controller_frequency
            self.position[0] += velocity * np.cos(theta) * dt
            self.position[1] += velocity * np.sin(theta) * dt
            self.velocity = np.array([velocity * np.cos(theta), velocity * np.sin(theta)])
            self.theta = theta

            # Update trajectory
            self.x_traj.append(self.position[0])
            self.y_traj.append(self.position[1])

            # Publish state
            self.robot_states.update(self.robot_id, {
                "position": tuple(self.position),
                "velocity": tuple(self.velocity),
                "theta": self.theta,
                "goal": tuple(self.goal),
                "global_path": global_path,
                "trajectory": (list(self.x_traj), list(self.y_traj)),
                "color": self.color,
                "safety_margin": self.safety_margin,
                "max_vel": self.max_vel
            })

            rate.sleep()

    def _reached_goal(self, tolerance=1.0):
        return np.linalg.norm(self.goal - self.position) < tolerance
