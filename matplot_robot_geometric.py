import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import yaml

dt = 0.075  # Time step
robot_radius = 1.0  # Assumed robot radius
safety_buffer = 4.0  # Safety margin around the robot

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
        self.controller = ControllerClass(self)  
        self.global_path = ([], [])  # Store the global path for visualization

    def update_total_robots(self, all_robots):
        """Assign the Collision Avoidance System after all robots are created."""
        self.avoidance_system = CollisionAvoidanceClass(all_robots)

    def update(self):
        """Move the robot forward using computed velocity and angular velocity."""
        if self.avoidance_system is None:
            return  

        # Compute the global path dynamically at each step
        self.global_path = self.avoidance_system.find_global_path(self)  # Store the path

        # Compute velocity and angular velocity using the latest path
        self.velocity, self.theta = self.controller.compute_control(self.global_path)

        print("[] global_path: {} velcity: {}".format(self.global_path, self.velocity))

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

    def collision_prediction(self, robot_pos, goal_pos, max_vel, obstacle_pos, obstacle_vel):
        """
        Predicts if a collision will occur between the robot and an obstacle based on motion dynamics.
        The robot's velocity is computed from its current position, goal position, and max velocity.
        """
        # Compute the robot's velocity vector towards the goal
        direction = np.array(goal_pos) - np.array(robot_pos)
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            robot_vel = np.array([0, 0])
        else:
            robot_vel = (direction / direction_norm) * max_vel

        rel_pos = np.array([obstacle_pos[0] - robot_pos[0], obstacle_pos[1] - robot_pos[1]])
        rel_vel = np.array([obstacle_vel[0] - robot_vel[0], obstacle_vel[1] - robot_vel[1]])
        print("         [] POS: obstacles: {} {} robot: {} {}".format(obstacle_pos[0], obstacle_pos[1], robot_pos[0], robot_pos[1]))
        print("         [] VEL: obstacles: {} {} robot: {} {}".format(obstacle_vel[0], obstacle_vel[1], robot_vel[0], robot_vel[1]))
        print("         [] rel_vel: {}".format(rel_vel))
        
        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq == 0:
            print("         [] Returning false")
            return 0, False
        
        TCPA = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        closest_point_robot = np.array(robot_pos) + robot_vel * TCPA
        closest_point_obstacle = np.array(obstacle_pos) + obstacle_vel * TCPA
        DCPA = np.linalg.norm(closest_point_robot - closest_point_obstacle)
        
        collision_distance = 3 * robot_radius + safety_buffer
        print("     [$] DCPA: {} TCPA: {} collision_distance: {}".format(DCPA, TCPA, collision_distance))
        
        if DCPA <= collision_distance and 0 < TCPA < 15:
            return DCPA, True
        else:
            return 0, False


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


    def straight_to_goal_safe_tangent_segments(self, robot, tangent1, tangent2, width):
        """
        Check if the straight-line path from the robot to its goal intersects a rectangle
        bounded by the tangents with a given width.
        """
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def segments_intersect(A, B, C, D):
            """Check if segment AB intersects with segment CD."""
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        def get_rectangle_corners(A, B, width):
            """Compute the four corners of the rectangle centered on segment AB with given width."""
            AB = np.array(B) - np.array(A)
            AB_unit = AB / np.linalg.norm(AB)
            perp = np.array([-AB_unit[1], AB_unit[0]]) * (width / 2)
            
            return [A + perp, B + perp, B - perp, A - perp]

        def rectangle_intersects_line(rect_corners, P1, P2):
            """Check if a line segment intersects any of the rectangle edges."""
            edges = [
                (rect_corners[0], rect_corners[1]),
                (rect_corners[1], rect_corners[2]),
                (rect_corners[2], rect_corners[3]),
                (rect_corners[3], rect_corners[0])
            ]
            return any(segments_intersect(P1, P2, edge[0], edge[1]) for edge in edges)

        # Define the robot's straight-line path to the goal
        robot_pos = robot.get_position()
        goal_pos = robot.goal_position

        # Compute the rectangle around the tangent segment
        rectangle_corners = get_rectangle_corners(tangent1, tangent2, width)

        # Check if the robot's straight path to the goal intersects the rectangle
        return rectangle_intersects_line(rectangle_corners, robot_pos, goal_pos)



    
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

            dcpa, collision = self.collision_prediction(
                current_pos, goal_pos, robot.max_vel, nearest_obstacle, nearest_velocity
            )
            print("     [] collision: {}".format(collision))

            if collision:
                tangent1, tangent2 = self.compute_tangents(current_pos, nearest_obstacle, robot_radius + safety_buffer)
                # check if the straight line to the goal does hit the segment bounded by tangent1 and tangent2

                if self.straight_to_goal_safe_tangent_segments(robot, tangent1, tangent2, 2):
                    print("     [$$$] Straight to goal doesn't intersect with tangent segments")
                    pass
                else:
                    # If it doesn not, just leave this step, else do the tangent selection and add as a waypoint
                    tangent_points.extend([tangent1, tangent2])
                    path_length1 = np.linalg.norm(np.array(current_pos) - np.array(tangent1)) + np.linalg.norm(np.array(tangent1) - np.array(goal_pos))
                    path_length2 = np.linalg.norm(np.array(current_pos) - np.array(tangent2)) + np.linalg.norm(np.array(tangent2) - np.array(goal_pos))
                    selected_tangent = tangent1 if path_length1 < path_length2 else tangent2
                    if np.sqrt((current_pos[0] - selected_tangent[0])**2 - (current_pos[1] - selected_tangent[1])**2) < 1:
                        pass
                    else:
                        full_path_x.append(selected_tangent[0]*1.1)
                        full_path_y.append(selected_tangent[1]*1.1)
                        current_pos = selected_tangent
            else:
                break

            obstacles.pop(nearest_idx)
            obstacle_velocities.pop(nearest_idx)

        full_path_x.append(goal_pos[0])
        full_path_y.append(goal_pos[1])
        print(" [] full_path: {} {}".format(full_path_x, full_path_y))
        return full_path_x, full_path_y


class ControllerClass:
    def __init__(self, robot):
        """Initialize the controller."""
        self.robot = robot

    def compute_control(self, global_path):
        """Compute the velocity and theta needed to move toward the first waypoint in the global path."""
        if not global_path[0]:  # If the global path is empty
            return 0, self.robot.theta  # Stop and maintain current orientation

        # Get the first waypoint from the global path
        target_x = global_path[0][1]
        target_y = global_path[1][1]
        current_x, current_y = self.robot.get_position()

        # Compute the desired theta to the first waypoint
        theta = np.arctan2(target_y - current_y, target_x - current_x)

        # Compute the angle difference between the current direction and target direction
        angle_diff = np.arctan2(np.sin(theta - self.robot.theta), np.cos(theta - self.robot.theta))

        # Compute distance to the first waypoint
        distance = np.linalg.norm([target_x - current_x, target_y - current_y])

        # Determine velocity based on the angle difference
        if abs(angle_diff) > np.pi / 2:  # If angle difference is greater than 90 degrees
            velocity = -self.robot.max_vel  # Move backward
            theta += np.pi  # Adjust theta to face the opposite direction
        else:
            velocity = self.robot.max_vel  # Move forward

        # Normalize theta to keep it within [-π, π]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        # Stop if very close to the waypoint
        if distance < 0.5:
            velocity = 0  

        return velocity, theta  # Return velocity and required theta




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
            x_path, y_path = robot.global_path
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
        print("[+] ======== Robot Simulation ======")
        for robot in self.robots:
            robot.update()
        return self.plot.update_plot()

    def run(self):
        """Run the Matplotlib animation."""
        plt.show()



# Run the simulation
if __name__ == "__main__":
    print("robot simulation")
    simulation = RobotSimulation()
    simulation.run()
