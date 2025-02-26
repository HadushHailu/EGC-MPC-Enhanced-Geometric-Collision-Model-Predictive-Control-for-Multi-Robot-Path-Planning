import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import yaml

dt = 0.1  # Time step


class Robot:
    def __init__(self, config, global_dims):
        """Initialize the robot using YAML configuration with nested x and y."""
        self.position = np.array([config["start"]["x"], config["start"]["y"]], dtype=float)
        self.goal_position = np.array([config["goal"]["x"], config["goal"]["y"]], dtype=float)
        self.color = config["color"]
        self.max_vel = config["max_vel"]  # Read max velocity from YAML
        
        # Set global robot dimensions
        self.length = global_dims["length"]
        self.width = global_dims["width"]
        self.safety_margin = global_dims["safety_margin"]  # Store safety margin
        
        self.theta = 0.0  # Orientation

        # Trajectory tracking
        self.x_traj = [self.position[0]]
        self.y_traj = [self.position[1]]

    def update(self):
        """Move the robot forward using its assigned velocity."""
        self.position[0] += self.max_vel * np.cos(self.theta) * dt
        self.position[1] += self.max_vel * np.sin(self.theta) * dt

        # Update trajectory
        self.x_traj.append(self.position[0])
        self.y_traj.append(self.position[1])

    def get_position(self):
        return self.position

    def get_trajectory(self):
        return self.x_traj, self.y_traj


class RobotPlot:
    def __init__(self):
        """Initialize the Matplotlib figure and configure the plot."""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.configure_plot()
        self.robots_visuals = []

    def configure_plot(self):
        """Configure the plot with grid, labels, and appearance."""
        self.ax.set_xlim(-35, 35)
        self.ax.set_ylim(-35, 35)
        self.ax.set_title("Multiple Robots Simulation with Safety Margin")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_aspect('equal', adjustable='datalim')  # Fix aspect ratio

        self.ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.8)
        self.ax.minorticks_on()
        self.ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.9)
        self.ax.set_facecolor("#F5F5F5")

        self.ax.axhline(0, color="black", linewidth=0.3, alpha=0.7)
        self.ax.axvline(0, color="black", linewidth=0.3, alpha=0.7)

    def add_robot(self, robot):
        """Add robot visualization to the plot."""
        robot_rect = Rectangle(
            (robot.get_position()[0] - robot.length / 2, robot.get_position()[1] - robot.width / 2),
            robot.length,
            robot.width,
            angle=np.degrees(robot.theta),
            facecolor=robot.color,
            edgecolor="black",
            linewidth=2,
            alpha=0.8
        )
        self.ax.add_patch(robot_rect)

        # Safety margin circle
        safety_circle = Circle(
            robot.get_position(),
            radius=robot.safety_margin,
            edgecolor=robot.color,
            facecolor="none",
            linewidth=1.5,
            alpha=0.7
        )
        self.ax.add_patch(safety_circle)

        # Add trajectory line
        trajectory_line, = self.ax.plot([], [], linestyle="-", color=robot.color)

        self.robots_visuals.append((robot, robot_rect, safety_circle, trajectory_line))
        self.ax.legend()

    def update_plot(self):
        """Update the positions and trajectories of all robots."""
        updated_elements = []
        for robot, robot_rect, safety_circle, trajectory_line in self.robots_visuals:
            robot_rect.set_xy((robot.get_position()[0] - robot.length / 2, robot.get_position()[1] - robot.width / 2))
            robot_rect.angle = np.degrees(robot.theta)

            # Update safety margin circle position
            safety_circle.center = robot.get_position()

            # Update trajectory line
            x_traj, y_traj = robot.get_trajectory()
            trajectory_line.set_data(x_traj, y_traj)

            updated_elements.extend([robot_rect, safety_circle, trajectory_line])

        return updated_elements


class RobotSimulation:
    def __init__(self, config_file="robots_config.yaml"):
        """Initialize the simulation by reading YAML configurations."""
        self.plot = RobotPlot()
        self.robots = []

        # Load configurations from YAML
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)

        print("[] Config_data: {}".format(config_data))
        # Read global robot dimensions
        global_dims = config_data["robot_dim"]

        # Create robots based on the YAML file
        for robot_config in config_data["robots"]:
            robot = Robot(robot_config, global_dims)
            self.robots.append(robot)
            self.plot.add_robot(robot)

        # Create animation
        self.anim = FuncAnimation(self.plot.fig, self.update, frames=500, interval=100, blit=False)

    def update(self, frame):
        """Update function for the animation."""
        for robot in self.robots:
            robot.update()
        return self.plot.update_plot()

    def run(self):
        """Run the Matplotlib animation."""
        plt.show()


# Run the simulation
if __name__ == "__main__":
    simulation = RobotSimulation()
    simulation.run()
