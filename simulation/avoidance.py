# simulation/avoidance.py

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

# Constants
robot_radius = 1.0
safety_buffer = 4.0


class CollisionAvoidance:
    def __init__(self,robot_id=None):
        self.logger = logging.getLogger(f"avoidance.{robot_id or 'generic'}")

    def collision_prediction(self, robot_pos, goal_pos, max_vel, obstacle_pos, obstacle_vel):
        obstacle_vel = np.array(obstacle_vel)
        direction = np.array(goal_pos) - np.array(robot_pos)
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            robot_vel = np.array([0, 0])
        else:
            robot_vel = (direction / direction_norm) * max_vel

        rel_pos = np.array(obstacle_pos) - np.array(robot_pos)
        rel_vel = obstacle_vel - robot_vel

        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq == 0:
            return 0, float("inf"), False

        TCPA = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        closest_point_robot = np.array(robot_pos) + robot_vel * TCPA
        closest_point_obstacle = np.array(obstacle_pos) + obstacle_vel * TCPA
        DCPA = np.linalg.norm(closest_point_robot - closest_point_obstacle)

        collision_distance = 5 * robot_radius + safety_buffer
        will_collide = DCPA <= collision_distance and 0 < TCPA < 15

        self.logger.info(
            f"[COLLISION] Checking: robot @ {robot_pos}, obstacle @ {obstacle_pos}"
        )
        self.logger.info(
            f"→ rel_vel={rel_vel}, TCPA={TCPA:.2f}, DCPA={DCPA:.2f}, result={'YES' if will_collide else 'NO'}"
        )

        return DCPA, TCPA, will_collide

    def filter_and_rank_obstacles_by_tcpa(self, robot_pos, goal_pos, max_vel, obstacles, velocities):
        ranked_threats = []

        for idx, (obs, vel) in enumerate(zip(obstacles, velocities)):
            dcpa, tcpa, collision = self.collision_prediction(robot_pos, goal_pos, max_vel, obs, vel)
            if collision:
                ranked_threats.append((idx, obs, vel, tcpa))

        ranked_threats.sort(key=lambda tup: tup[3])  # sort by TCPA
        return ranked_threats

    def compute_tangents(self, robot_pos, obstacle_pos, radius_with_buffer):
        x_p, y_p = robot_pos
        x_c, y_c = obstacle_pos
        Q = np.sqrt((x_c - x_p) ** 2 + (y_c - y_p) ** 2)
        L = radius_with_buffer
        if Q <= robot_radius:
            raise ValueError("Too close to compute tangents!")
        if Q < L:
            Q = L
        M = np.sqrt(Q**2 - L**2)
        theta = np.arcsin(L / Q)
        alpha = np.arctan2(y_c - y_p, x_c - x_p)
        x_s1 = x_p + M * np.cos(alpha - theta)
        y_s1 = y_p + M * np.sin(alpha - theta)
        x_s2 = x_p + M * np.cos(alpha + theta)
        y_s2 = y_p + M * np.sin(alpha + theta)
        return (x_s1, y_s1), (x_s2, y_s2)

    def straight_to_goal_safe_tangent_segments(self, robot, tangent1, tangent2, width):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def segments_intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        def get_rectangle_corners(A, B, width):
            AB = np.array(B) - np.array(A)
            AB_unit = AB / np.linalg.norm(AB)
            perp = np.array([-AB_unit[1], AB_unit[0]]) * (width / 2)
            return [A + perp, B + perp, B - perp, A - perp]

        def rectangle_intersects_line(rect_corners, P1, P2):
            edges = [
                (rect_corners[0], rect_corners[1]),
                (rect_corners[1], rect_corners[2]),
                (rect_corners[2], rect_corners[3]),
                (rect_corners[3], rect_corners[0])
            ]
            return any(segments_intersect(P1, P2, edge[0], edge[1]) for edge in edges)

        robot_pos = robot.position
        goal_pos = robot.goal
        rectangle_corners = get_rectangle_corners(tangent1, tangent2, width)
        return rectangle_intersects_line(rectangle_corners, robot_pos, goal_pos)

    def estimate_ghost_goals(self, other_robots):
        ghost_goals = {}

        for robot_id, state in other_robots.items():
            x_traj, y_traj = state["trajectory"]

            if len(x_traj) < 2:
                continue  # Not enough data to infer motion

            # Use last two points for direction
            p_prev = np.array([x_traj[-2], y_traj[-2]])
            p_curr = np.array([x_traj[-1], y_traj[-1]])
            direction = p_curr - p_prev
            direction_norm = np.linalg.norm(direction)

            if direction_norm < 1e-5:
                continue  # Robot is not moving significantly

            direction_unit = direction / direction_norm
            ghost_goal = p_curr + direction_unit * 10.0  # Projected 10 meters ahead
            self.logger.info(f"[GHOST] Robot {robot_id}: from {p_prev.tolist()} → {p_curr.tolist()} → projected ghost goal {ghost_goal}")


            ghost_goals[robot_id] = tuple(ghost_goal)

        return ghost_goals

    def compute_trajectory_linearity(self, trajectory):
        x_traj, y_traj = trajectory
        if len(x_traj) < 3:
            return 0.0  # Not enough data to judge

        # Stack points into 2D array
        points = np.column_stack((x_traj, y_traj))

        # Fit a line through the trajectory (simple linear regression)
        start = points[0]
        end = points[-1]
        total_length = np.linalg.norm(end - start)

        if total_length < 1e-5:
            return 0.0  # Robot hasn't moved

        # Compute average deviation from the line
        direction = (end - start) / total_length
        line_points = start + np.outer(np.dot(points - start, direction), direction)
        deviation = np.linalg.norm(points - line_points, axis=1)
        avg_deviation = np.mean(deviation)

        # Convert deviation into a confidence score
        # Lower deviation → closer to 1.0, higher deviation → closer to 0.0
        score = np.clip(1.0 - (avg_deviation / total_length), 0.0, 1.0)
        self.logger.info(f"[LINEARITY] score={score:.2f}, avg_deviation={avg_deviation:.2f}")

        return score

    def compute_ghost_path_distance(self, tangent, line_start, line_end):
        p = np.array(tangent)
        a = np.array(line_start)
        b = np.array(line_end)

        ab = b - a
        if np.linalg.norm(ab) < 1e-5:
            return float("inf")  # Avoid divide-by-zero: line too short

        # Compute perpendicular distance from point to line
        projected = a + np.dot(p - a, ab) / np.dot(ab, ab) * ab
        distance = np.linalg.norm(projected - p)
        return distance

    def compute_heading_alignment(self, from_pos, to_pos, ghost_start, ghost_goal):
        my_vec = np.array(to_pos) - np.array(from_pos)
        ghost_vec = np.array(ghost_goal) - np.array(ghost_start)

        if np.linalg.norm(my_vec) < 1e-5 or np.linalg.norm(ghost_vec) < 1e-5:
            return 0.0  # No movement to compare

        my_unit = my_vec / np.linalg.norm(my_vec)
        ghost_unit = ghost_vec / np.linalg.norm(ghost_vec)

        cos_theta = np.clip(np.dot(my_unit, ghost_unit), -1.0, 1.0)
        angle_penalty = 1 - cos_theta  # 0 if aligned, 2 if opposite
        return angle_penalty

    def combine_cost_components(self, normalized_path_length, ghost_distances, heading_penalties, linearity_weights, weights):
        ghost_penalty = 0.0
        angle_penalty = 0.0

        # Combine ghost and angle costs, weighted by each robot's linearity confidence
        for d, a, w in zip(ghost_distances, heading_penalties, linearity_weights):
            if d == 0:
                d = 1e-3  # avoid divide by zero

            ghost_penalty += w * (1.0 / d)
            angle_penalty += w * a

        total_cost = (
            weights["goal"] * normalized_path_length +
            weights["ghost_dist"] * ghost_penalty +
            weights["angle"] * angle_penalty
        )

        self.logger.info(
            f"[COST] Tangent: CURRENT_TANGENT | "
            f"PathLen: {normalized_path_length:.2f} | "
            f"GhostPenalty: {ghost_penalty:.2f} | "
            f"AnglePenalty: {angle_penalty:.2f} | "
            f"Total: {total_cost:.2f}"
        )

        return total_cost

    def select_best(self, candidates, costs):
        if not candidates or not costs or len(candidates) != len(costs):
            return None  # Defensive programming

        min_index = np.argmin(costs)
        self.logger.info(f"[DECISION] Selected tangent {candidates[min_index]} with cost {costs[min_index]:.2f}")

        return candidates[min_index]

    def select_best_via_point(self, robot, current_pos, goal_pos, tangent_points, other_robots):
        weights = {
            "goal": 1.0,
            "ghost_dist": 4.0,
            "angle": 1.5
        }

        costs = []
        ghost_goals = self.estimate_ghost_goals(other_robots)

        for tangent in tangent_points:
            # Path length
            path_len = np.linalg.norm(np.array(current_pos) - np.array(tangent)) + np.linalg.norm(np.array(tangent) - np.array(goal_pos))
            normalized_path_len = path_len / (np.linalg.norm(np.array(current_pos) - np.array(goal_pos)) + 1e-3)

            ghost_distances = []
            heading_penalties = []
            linearity_weights = []

            for robot_id, ghost_goal in ghost_goals.items():
                state = other_robots[robot_id]
                ghost_start = state["position"]
                trajectory = state["trajectory"]

                linearity = self.compute_trajectory_linearity(trajectory)
                linearity_weights.append(linearity)

                dist_to_ghost_path = self.compute_ghost_path_distance(tangent, ghost_start, ghost_goal)
                ghost_distances.append(dist_to_ghost_path)

                angle_penalty = self.compute_heading_alignment(current_pos, tangent, ghost_start, ghost_goal)
                heading_penalties.append(angle_penalty)
                
            if ghost_distances and min(ghost_distances) < 0.3:
                self.logger.info(
                    f"[FILTER] Skipping tangent {tangent} — too close to ghost path (min ghost_dist={min(ghost_distances):.2f})"
                )
                continue  # Skip evaluating this tangent

            self.logger.info(f"[TANGENT] Evaluating via point: {tangent}")
            self.logger.info(f"→ normalized_path_length={normalized_path_len:.2f}")
            self.logger.info(f"→ ghost_distances={ghost_distances}")
            self.logger.info(f"→ angle_penalties={heading_penalties}")
            self.logger.info(f"→ linearity_weights={linearity_weights}")


            cost = self.combine_cost_components(
                normalized_path_len,
                ghost_distances,
                heading_penalties,
                linearity_weights,
                weights
            )
            costs.append(cost)
            self.logger.info(f"→ total_cost={cost:.2f}")


        return self.select_best(tangent_points, costs)

    def compute_velocity_based_viapoints(self, robot_pos, goal_pos, obstacle_pos, obstacle_velocity, tangent_points, cone_slope=1.0):
        """
        Computes via points by intersecting avoiding lines (through tangent points) with the robot's motion cone.

        Parameters:
            robot_pos: np.array([x, y]) — Robot's current position
            goal_pos: np.array([x, y]) — Robot's goal position
            obstacle_pos: np.array([x, y]) — Obstacle's current position
            obstacle_velocity: np.array([vx, vy]) — Obstacle's velocity vector
            tangent_points: List of 2D points (from compute_tangents)
            cone_slope: float — r in the cone equation

        Returns:
            List of 3D via points (intersection points on avoiding lines within the cone)
        """
        vz = 1.0  # assume linear time progression (1 unit of z per time unit)
        obstacle_vel_3d = np.array([obstacle_velocity[0], obstacle_velocity[1], vz])
        via_points = []

        for tangent in tangent_points:
            # Lift the tangent point to 3D (z=0)
            tangent_3d = np.array([tangent[0], tangent[1], 0.0])

            # Call cone intersection method
            intersections = self.intersect_line_with_cone(
                line_origin=tangent_3d,
                line_direction=obstacle_vel_3d,
                robot_position=np.array(robot_pos),
                cone_slope=cone_slope
            )

            via_points.extend(intersections)

        return via_points

    def evaluate_viapoints(self, viapoints, goal_pos, robot_max_vel):
        """
        Select the via point that results in the minimum estimated time to reach the goal.

        Parameters:
            viapoints: List of 3D np.array([x, y, z]) — valid via points
            goal_pos: np.array([x, y]) — robot's goal position
            robot_max_vel: float — maximum velocity of the robot

        Returns:
            np.array([x, y]) — the best via point in 2D space
        """
        if not viapoints:
            return None

        best_time = float('inf')
        best_point = None

        goal = np.array(goal_pos)

        for vp in viapoints:
            vp_2d = vp[:2]
            z = vp[2]  # time to reach the via point
            dist_to_goal = np.linalg.norm(goal - vp_2d)

            time_to_goal = z + dist_to_goal / (robot_max_vel + 1e-6)  # avoid div by zero
            if time_to_goal < best_time:
                best_time = time_to_goal
                best_point = vp_2d

        return best_point


    def intersect_line_with_cone(self, line_origin, line_direction, robot_position, cone_slope):
        """
        Find the intersection(s) between a parametric avoiding line and the robot's motion cone.

        Parameters:
            line_origin: np.array([px, py, pz]) — starting point of the avoiding line (3D)
            line_direction: np.array([vx, vy, vz]) — direction vector of the avoiding line (3D)
            robot_position: np.array([x_p, y_p]) — current 2D position of the robot (assumed z = 0)
            cone_slope: float — r in the cone equation, defines cone "steepness"

        Returns:
            List of np.array([x, y, z]) — valid intersection points in 3D space
        """
        epsilon = 1e-8  # numerical tolerance

        px, py, pz = line_origin
        vx, vy, vz = line_direction
        x_p, y_p = robot_position

        # Coefficients of the quadratic Aλ² + Bλ + C = 0
        A = vx**2 + vy**2 - (cone_slope**2) * vz**2
        B = 2 * ((px - x_p) * vx + (py - y_p) * vy - (cone_slope**2) * pz * vz)
        C = (px - x_p)**2 + (py - y_p)**2 - (cone_slope**2) * (pz**2)

        intersections = []

        # Handle degenerate case: A ≈ 0 (line nearly tangent to cone or flat)
        if abs(A) < epsilon:
            self.logger.warning(f"[INTERSECT] Near-degenerate case (A≈0). Perturbing slope slightly.")
            A += epsilon  # Avoid full degeneracy and still solve the quadratic

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return intersections  # No real intersection

        sqrt_discriminant = np.sqrt(discriminant)

        for sign in [-1, 1]:
            lambda_val = (-B + sign * sqrt_discriminant) / (2 * A)
            if lambda_val > 0:
                point = line_origin + lambda_val * line_direction
                intersections.append(point)

        return intersections

    def filter_non_progressive_xpoints(self, path, goal):
        """
        Filters intermediate points where x-distance to the goal does not decrease.

        Parameters:
            path (List[Tuple[float, float]]): Complete path [start, ..., goal]
            goal (Tuple[float, float]): Robot's goal position

        Returns:
            List[Tuple[float, float]]: Filtered path with only improving x-distance to goal
        """
        if len(path) <= 3:
            return path  # No meaningful intermediates to filter

        goal_x = goal[0]
        filtered = [path[1]]  # First intermediate (skip start)

        last_x_dist = abs(path[1][0] - goal_x)

        for pt in path[2:-1]:  # Remaining intermediates (excluding goal)
            x_dist = abs(pt[0] - goal_x)
            if x_dist <= last_x_dist + 1e-3:  # Allow small tolerance
                filtered.append(pt)
                last_x_dist = x_dist
            else:
                self.logger.info(
                    f"[FILTER] Discarding {pt} — x-distance to goal increased ({x_dist:.2f} > {last_x_dist:.2f})"
                )

        return [path[0]] + filtered + [path[-1]]



    def plan(self, robot, other_robots):
        self.logger.info(f"[PLAN] Starting path planning at pos {robot.position} to goal {robot.goal}")
        robot_pos = robot.position
        goal_pos = robot.goal

        obstacles = [r["position"] for r in other_robots.values()]
        obstacle_velocities = [r["velocity"] for r in other_robots.values()]

        current_pos = robot_pos.copy()
        full_path_x = [current_pos[0]]
        full_path_y = [current_pos[1]]
        tangent_points = []

        # Step 1: Filter obstacles that will cause collision and sort by TCPA
        ranked_threats = self.filter_and_rank_obstacles_by_tcpa(
            robot_pos=current_pos,
            goal_pos=goal_pos,
            max_vel=robot.max_vel,
            obstacles=obstacles,
            velocities=obstacle_velocities
        )

        # Step 2: Process each ranked obstacle and evaluate avoidance
        for idx, nearest_obstacle, nearest_velocity, tcpa in ranked_threats:
            tangent1, tangent2 = self.compute_tangents(current_pos, nearest_obstacle, robot_radius + safety_buffer)
            tangent_points.extend([tangent1, tangent2])

            via_points = self.compute_velocity_based_viapoints(
                robot_pos=current_pos,
                goal_pos=goal_pos,
                obstacle_pos=nearest_obstacle,
                obstacle_velocity=nearest_velocity,
                tangent_points=[tangent1, tangent2],
                cone_slope=robot.max_vel
            )

            selected_viapoint = self.evaluate_viapoints(via_points, goal_pos, robot.max_vel)
            
            if selected_viapoint is not None and np.linalg.norm(np.array(current_pos) - np.array(selected_viapoint)) >= 1.0:
                full_path_x.append(selected_viapoint[0])
                full_path_y.append(selected_viapoint[1])
                current_pos = selected_viapoint
            else:
                self.logger.info("[AVOIDANCE] No valid via point selected for obstacle — checking next threat.")

        # Step 3: Append goal position
        full_path_x.append(goal_pos[0])
        full_path_y.append(goal_pos[1])
        full_path = list(zip(full_path_x, full_path_y))
        full_path = self.filter_non_progressive_xpoints(full_path, goal_pos)
        full_path_x = [pt[0] for pt in full_path]
        full_path_y = [pt[1] for pt in full_path]
        self.logger.info(f"[PATH] Robot {robot.robot_id}: planned path with {len(full_path)} points → {full_path}")


        return full_path_x, full_path_y
