# simulation/egc_mpc_simulation.py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("avoidance_debug.log"),
        logging.StreamHandler()
    ]
)

class EgcMpcSimulation:
    def __init__(self, robot_states, config, stop_event):

        self.robot_states = robot_states
        self.robot_dim = config["robot_dim"]
        self.stop_event = stop_event

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._configure_plot()

        self.robot_visuals = {}

    def _configure_plot(self):
        """Configure the plot with grid, labels, and appearance."""
        self.ax.set_xlim(-35, 35)
        self.ax.set_ylim(-35, 35)
        self.ax.set_title("Multiple Robots Simulation with Waypoints & Paths")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_aspect('equal', adjustable='datalim')

        self.ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.8)
        self.ax.minorticks_on()
        self.ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.9)
        self.ax.set_facecolor("#F5F5F5")

        self.ax.axhline(0, color="black", linewidth=0.3, alpha=0.7)
        self.ax.axvline(0, color="black", linewidth=0.3, alpha=0.7)

    def _init_robot_visual(self, robot_id, state):
        color = state["color"]

        robot_rect = Rectangle(
            (0, 0), self.robot_dim["length"], self.robot_dim["width"],
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.8
        )
        self.ax.add_patch(robot_rect)

        safety_circle = Circle(
            (0, 0), radius=state["safety_margin"],
            edgecolor=color, facecolor="none", linewidth=1.2, alpha=0.6
        )
        self.ax.add_patch(safety_circle)

        traj_line, = self.ax.plot([], [], linestyle="-", color=color, label=robot_id)
        path_line, = self.ax.plot([], [], linestyle="--", color=color, alpha=0.4)
        waypoint_scatter = self.ax.scatter([], [], color=color, edgecolor="black", s=50, alpha=0.6)

        self.robot_visuals[robot_id] = (robot_rect, safety_circle, traj_line, path_line, waypoint_scatter)
        self.ax.legend()

    def _update_plot(self, frame):
        states = self.robot_states.get_all()

        for robot_id, state in states.items():
            if robot_id not in self.robot_visuals:
                self._init_robot_visual(robot_id, state)

            rect, circle, traj, path, scatter = self.robot_visuals[robot_id]

            # Update rectangle position and rotation
            rect.set_xy((state["position"][0] - self.robot_dim["length"] / 2,
                         state["position"][1] - self.robot_dim["width"] / 2))
            rect.angle = np.degrees(state["theta"])

            # Update safety circle
            circle.center = state["position"]

            # Update trajectory
            x_traj, y_traj = state["trajectory"]
            traj.set_data(x_traj, y_traj)

            # Draw full global path (dashed line)
            x_path, y_path = state.get("global_path", ([], []))
            if x_path and y_path:
                logging.info(f"[VIS] Robot {robot_id} global_path x: {x_path[:3]} y: {y_path[:3]} (len={len(x_path)})")
                path.set_data(x_path, y_path)
            else:
                logging.info(f"[VIS] Robot {robot_id} global_path is empty or malformed → x: {x_path}, y: {y_path}")

            # Waypoints: show all intermediate points except start and goal
            if len(x_path) > 2:
                via_x = x_path[1:-1]
                via_y = y_path[1:-1]
                scatter.set_offsets(np.column_stack((via_x, via_y)))
            else:
                scatter.set_offsets([[state["goal"][0], state["goal"][1]]])


        return sum(self.robot_visuals.values(), ())

    def run(self):
        self.anim = FuncAnimation(self.fig, self._update_plot, interval=100, blit=False)
        try:
            plt.show()
        finally:
            self.stop_event.set()
