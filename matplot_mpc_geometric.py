import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import yaml
import cvxpy as cp
from scipy.interpolate import CubicSpline


dt = 0.1  # Time step
robot_radius = 1.0  # Assumed robot radius
safety_buffer = 4.0  # Safety margin around the robot


class CubicSplineClass:
    def __init__(self, path_resolution=0.1):
        """
        Initialize the spline generator.
        
        Parameters:
        - path_resolution: Step size for sampling the spline (smaller = smoother path).
        """
        self.path_resolution = path_resolution  # How finely the spline is sampled

    def generate_spline(self, global_path):
        """
        Generate a cubic spline from the global path.
        
        Parameters:
        - global_path: (x_path, y_path), a list of waypoints from the global planner.
        
        Returns:
        - (smooth_x, smooth_y): Smooth trajectory points from the spline.
        """

        x_path, y_path = global_path
        print("[] GENERATE SPLINE: {}".format(x_path))

        # Ensure we have enough points for interpolation
        if len(x_path) < 3 or len(y_path) < 3:
            print("[!] Warning: Not enough waypoints for spline interpolation, returning original path.")
            return np.linspace(x_path[0], x_path[-1], num=5), np.linspace(y_path[0], y_path[-1], num=5)

        # Define time indices for the waypoints
        t = np.linspace(0, len(x_path) - 1, len(x_path))

        # Create cubic spline functions for x and y
        spline_x = CubicSpline(t, x_path)
        spline_y = CubicSpline(t, y_path)

        # Generate new time samples for a smoother path
        t_smooth = np.arange(0, len(x_path) - 1, self.path_resolution)
        smooth_x = spline_x(t_smooth)
        smooth_y = spline_y(t_smooth)

        return smooth_x, smooth_y

class NonLinearMPC:
    def __init__(self, robot, horizon=10, dt=0.1):
        """
        Initialize the NMPC controller.
        
        Parameters:
        - robot: The robot object containing its current state and dynamics.
        - horizon: Prediction horizon (number of future steps to optimize).
        - dt: Time step for each NMPC prediction.
        """
        self.robot = robot
        self.horizon = horizon  # Number of future steps to optimize
        self.dt = dt  # Time step for NMPC

        # Define control limits
        self.max_vel = robot.max_vel
        self.max_steering = np.pi / 4  # Max steering angle (±45 degrees)
        self.max_accel = 1.0  # Max acceleration

    def compute_control(self, spline_path):
        """Solves the NMPC optimization to find the best velocity and steering angle (theta)."""
        print("[!] spline_path: {}".format(spline_path))

        # 🔹 Fix: Ensure NMPC gets `horizon + 1` waypoints
        if len(spline_path[0]) < self.horizon + 1:
            print("[!] Warning: Not enough waypoints for NMPC, increasing resolution.")
            extra_x = np.linspace(spline_path[0][0], spline_path[0][-1], num=self.horizon + 1)
            extra_y = np.linspace(spline_path[1][0], spline_path[1][-1], num=self.horizon + 1)
            spline_path = (extra_x, extra_y)

        x_ref = np.array(spline_path[0][: self.horizon + 1]).reshape(-1, 1)  # 🔹 Fix: Column vector
        y_ref = np.array(spline_path[1][: self.horizon + 1]).reshape(-1, 1)  # 🔹 Fix: Column vector

        # 🔹 Fix: Ensure valid size before solving NMPC
        if len(x_ref) != self.horizon + 1 or len(y_ref) != self.horizon + 1:
            print("[!] Error: x_ref and y_ref do not match horizon length!")
            return 0, self.robot.theta  # Stop robot if data is incorrect

        # Extract current position
        x0, y0 = self.robot.get_position()
        theta0 = self.robot.theta
        v0 = self.robot.velocity  # Current velocity

        # Decision Variables (for each step in the horizon)
        x = cp.Variable((self.horizon + 1, 1))  # X positions
        y = cp.Variable((self.horizon + 1, 1))  # Y positions
        theta = cp.Variable((self.horizon + 1, 1))  # Orientations
        v = cp.Variable((self.horizon, 1))  # Velocities
        steering = cp.Variable((self.horizon, 1))  # Steering angles

        # Constraints and Cost Function
        constraints = []
        cost = 0

        # Define weight matrices
        Q = np.eye(self.horizon + 1)  # Weight matrix for tracking error
        R = 0.1 * np.eye(self.horizon)  # Weight matrix for control effort

        # Initial conditions
        constraints += [x[0] == x0, y[0] == y0, theta[0] == theta0]

        for t in range(self.horizon):
            # Cost Function: Minimize tracking error and control effort
            cost += cp.quad_form(x - x_ref, Q)  # 🔹 Fix: Use proper matrix form
            cost += cp.quad_form(y - y_ref, Q)  # 🔹 Fix: Use proper matrix form
            cost += cp.quad_form(steering, R)  # Minimize steering effort
            cost += cp.quad_form(v - v0, R)  # Minimize acceleration effort

            # 🔹 Fix: Linearize cos(theta) and sin(theta)
            constraints += [
                x[t + 1] == x[t] + v[t] * (1 - theta[t]**2 / 2) * self.dt,  # Approximate cos(theta)
                y[t + 1] == y[t] + v[t] * theta[t] * self.dt,  # Approximate sin(theta)
                theta[t + 1] == theta[t] + (v[t] / self.robot.length) * steering[t] * self.dt,  # Linearized
            ]

            # Control constraints
            constraints += [
                v[t] <= self.max_vel,
                v[t] >= -self.max_vel,  # Allow reversing
                steering[t] <= self.max_steering,
                steering[t] >= -self.max_steering,
            ]

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status != cp.OPTIMAL:
            print("[!] NMPC Failed to find an optimal solution. Stopping robot.")
            return 0, self.robot.theta  # If solver fails, stop robot

        # Extract optimal velocity and steering angle for the first time step
        optimal_velocity = v.value[0, 0]
        optimal_steering = steering.value[0, 0]

        return optimal_velocity, optimal_steering


class Robot:
    def __init__(self, config, global_dims):
        """Initialize the robot using YAML configuration with nested x and y."""
        self.position = np.array([config["start"]["x"], config["start"]["y"]], dtype=float)
        self.goal_position = np.array([config["goal"]["x"], config["goal"]["y"]], dtype=float)
        self.color = config["color"]
        self.max_vel = config["max_vel"]
        self.velocity = self.max_vel 

        # Set global robot dimensions
        self.length = global_dims["length"]
        self.width = global_dims["width"]
        self.safety_margin = global_dims["safety_margin"]  

        self.theta = 0.0  # Orientation

        # Trajectory tracking
        self.x_traj = [self.position[0]]
        self.y_traj = [self.position[1]]

        # Collision Avoidance & Path Planning
        self.avoidance_system = None
        self.spline_generator = CubicSplineClass()  # Spline generator
        self.mpc = NonLinearMPC(self)  # NMPC Controller
        self.spline_path = ([], [])  # Store the smooth spline path

    def update_total_robots(self, all_robots):
        """Assign the Collision Avoidance System after all robots are created."""
        self.avoidance_system = CollisionAvoidanceClass(all_robots)

    def update(self):
        """Move the robot forward using NMPC-based velocity and theta."""
        if self.avoidance_system is None:
            return  

        # Compute the global path dynamically
        global_path = self.avoidance_system.find_global_path(self)  

        # Convert the global path to a smooth spline
        self.spline_path = self.spline_generator.generate_spline(global_path)

        # Compute velocity and theta using NMPC
        self.velocity, self.theta = self.mpc.compute_control(self.spline_path)

        print("[] NMPC Path: {} Velocity: {}".format(self.spline_path, self.velocity))

        # Update position using velocity and new orientation
        self.position[0] += self.velocity * np.cos(self.theta) * dt
        self.position[1] += self.velocity * np.sin(self.theta) * dt

        # Update trajectory
        self.x_traj.append(self.position[0])
        self.y_traj.append(self.position[1])

    def get_position(self):
        return self.position

    def get_velocity(self):
        return np.array([self.velocity * np.cos(self.theta), self.velocity * np.sin(self.theta)])

    def get_trajectory(self):
        return self.x_traj, self.y_traj


class CollisionAvoidanceClass:
    def __init__(self, robots):
        self.robots = robots

    def collision_prediction(self, robot_pos, robot_vel, obstacle_pos, obstacle_vel):
        rel_pos = np.array([obstacle_pos[0] - robot_pos[0], obstacle_pos[1] - robot_pos[1]])
        rel_vel = np.array([obstacle_vel[0] - robot_vel[0], obstacle_vel[1] - robot_vel[1]])
        print("         [] POS: obstacles: {} {} robot: {} {}".format(obstacle_pos[0], obstacle_pos[1], robot_pos[0], robot_pos[1]))
        print("         [] VEL: obstacles: {} {} robot: {} {}".format(obstacle_vel[0], obstacle_vel[1], robot_vel[0], robot_vel[1]))
        print("         [] rel_vel: {}".format(rel_vel))
        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq == 0:
            print("         [] Returning false")
            return False
        TCPA = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        closest_point_robot = np.array(robot_pos) + np.array(robot_vel) * TCPA
        closest_point_obstacle = np.array(obstacle_pos) + np.array(obstacle_vel) * TCPA
        DCPA = np.linalg.norm(closest_point_robot - closest_point_obstacle)
        collision_distance = robot_radius + safety_buffer
        print("     [$] DCPA: {} TCPA: {}".format(DCPA, TCPA))
        return DCPA <= collision_distance and 0 < TCPA < 10

    def compute_tangents(self, robot_pos, obstacle_pos, radius_with_buffer):
        x_p, y_p = robot_pos
        x_c, y_c = obstacle_pos
        Q = np.sqrt((x_c - x_p) ** 2 + (y_c - y_p) ** 2)
        L = radius_with_buffer
        if Q <= robot_radius:
            raise ValueError("Too close to compute tangents!")
        if Q < L:
            print("[!] Warning: Q < L, setting Q = L to prevent sqrt of negative")
            Q = L  # Prevents invalid sqrt calculation
        M = np.sqrt(Q**2 - L**2)
        theta = np.arcsin(L / Q)
        alpha = np.arctan2(y_c - y_p, x_c - x_p)
        x_s1 = x_p + M * np.cos(alpha - theta)
        y_s1 = y_p + M * np.sin(alpha - theta)
        x_s2 = x_p + M * np.cos(alpha + theta)
        y_s2 = y_p + M * np.sin(alpha + theta)
        return (x_s1, y_s1), (x_s2, y_s2)

    def find_global_path(self, robot):
        robot_pos = robot.get_position()
        goal_pos = robot.goal_position

        print("[x] robot_pos: {} goal_pos: {}".format(robot_pos, goal_pos))
        obstacles = [r.get_position() for r in self.robots if r != robot]
        obstacle_velocities = [r.get_velocity() for r in self.robots if r != robot]
        print("[x] obstacles: {} ".format(obstacles))
        current_pos = robot_pos
        full_path_x = [current_pos[0]]
        full_path_y = [current_pos[1]]
        tangent_points = []

        while obstacles:
            nearest_idx = np.argmin([np.linalg.norm(np.array(obs) - np.array(current_pos)) for obs in obstacles])
            nearest_obstacle = obstacles[nearest_idx]
            nearest_velocity = obstacle_velocities[nearest_idx]
            print("     [#] nearest_velocity: {}".format(nearest_velocity))

            collision = self.collision_prediction(
                current_pos, robot.get_velocity(), nearest_obstacle, nearest_velocity
            )
            print("     [] collision: {}".format(collision))

            if collision:
                tangent1, tangent2 = self.compute_tangents(current_pos, nearest_obstacle, robot_radius + safety_buffer)
                tangent_points.extend([tangent1, tangent2])
                path_length1 = np.linalg.norm(np.array(current_pos) - np.array(tangent1)) + np.linalg.norm(np.array(tangent1) - np.array(goal_pos))
                path_length2 = np.linalg.norm(np.array(current_pos) - np.array(tangent2)) + np.linalg.norm(np.array(tangent2) - np.array(goal_pos))
                selected_tangent = tangent1 if path_length1 < path_length2 else tangent2
                full_path_x.append(selected_tangent[0])
                full_path_y.append(selected_tangent[1])
                current_pos = selected_tangent
            else:
                break

            obstacles.pop(nearest_idx)
            obstacle_velocities.pop(nearest_idx)


        mid_x = (full_path_x[-1] + goal_pos[0]) / 2
        mid_y = (full_path_y[-1] + goal_pos[1]) / 2
        full_path_x.append(mid_x)
        full_path_y.append(mid_y)

        full_path_x.append(goal_pos[0])
        full_path_y.append(goal_pos[1])

        print("[] FIND GLOBAL PATh: {}".format(full_path_x))
        return full_path_x, full_path_y

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
        trajectory_line, = self.ax.plot([], [], linestyle="-", color=robot.color, label=f"Robot {len(self.robots_visuals) + 1}")

        # Add global path line (dashed line for visualization)
        global_path_line, = self.ax.plot([], [], linestyle="--", color=robot.color, alpha=0.5)

        # Add waypoints as scatter points
        waypoint_scatter = self.ax.scatter([], [], color=robot.color, marker='o', s=50, alpha=0.6, edgecolor='black')

        self.robots_visuals.append((robot, robot_rect, safety_circle, trajectory_line, global_path_line, waypoint_scatter))
        self.ax.legend()

    def update_plot(self):
        """Update the positions, trajectories, global paths, and waypoints of all robots."""
        updated_elements = []
        for robot, robot_rect, safety_circle, trajectory_line, global_path_line, waypoint_scatter in self.robots_visuals:
            robot_rect.set_xy((robot.get_position()[0] - robot.length / 2, robot.get_position()[1] - robot.width / 2))
            robot_rect.angle = np.degrees(robot.theta)

            # Update safety margin circle position
            safety_circle.center = robot.get_position()

            # Update trajectory line
            x_traj, y_traj = robot.get_trajectory()
            trajectory_line.set_data(x_traj, y_traj)

            # Update global path visualization
            x_path, y_path  = robot.spline_path
            global_path_line.set_data(x_path, y_path)

            # Update waypoints
            waypoint_scatter.set_offsets(np.column_stack((x_path, y_path)))

            updated_elements.extend([robot_rect, safety_circle, trajectory_line, global_path_line, waypoint_scatter])

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

        # Create robots based on the YAML file (without avoidance system)
        for robot_config in config_data["robots"]:
            robot = Robot(robot_config, global_dims)
            self.robots.append(robot)
            self.plot.add_robot(robot)

        # Now, update each robot with the complete list of robots
        for robot in self.robots:
            robot.update_total_robots(self.robots)

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
