# simulation/avoidance.py

import numpy as np

# Constants
robot_radius = 1.0
safety_buffer = 4.0

class CollisionAvoidance:
    def collision_prediction(self, robot_pos, goal_pos, max_vel, obstacle_pos, obstacle_vel):
        obstacle_vel = np.array(obstacle_vel)
        direction = np.array(goal_pos) - np.array(robot_pos)
        direction_norm = np.linalg.norm(direction)

        if direction_norm == 0:
            robot_vel = np.array([0, 0])
        else:
            robot_vel = (direction / direction_norm) * max_vel

        rel_pos = np.array([obstacle_pos[0] - robot_pos[0], obstacle_pos[1] - robot_pos[1]])
        rel_vel = np.array([obstacle_vel[0] - robot_vel[0], obstacle_vel[1] - robot_vel[1]])

        rel_speed_sq = np.dot(rel_vel, rel_vel)
        if rel_speed_sq == 0:
            return 0, False

        TCPA = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        closest_point_robot = np.array(robot_pos) + robot_vel * TCPA
        closest_point_obstacle = np.array(obstacle_pos) + obstacle_vel * TCPA
        DCPA = np.linalg.norm(closest_point_robot - closest_point_obstacle)

        collision_distance = 3 * robot_radius + safety_buffer

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

    def plan(self, robot, other_robots):
        robot_pos = robot.position
        goal_pos = robot.goal

        obstacles = [r["position"] for r in other_robots.values()]
        obstacle_velocities = [r["velocity"] for r in other_robots.values()]

        current_pos = robot_pos.copy()
        full_path_x = [current_pos[0]]
        full_path_y = [current_pos[1]]
        tangent_points = []

        while obstacles:
            nearest_idx = np.argmin([np.linalg.norm(np.array(obs) - np.array(current_pos)) for obs in obstacles])
            nearest_obstacle = obstacles[nearest_idx]
            nearest_velocity = obstacle_velocities[nearest_idx]

            dcpa, collision = self.collision_prediction(
                current_pos, goal_pos, robot.max_vel, nearest_obstacle, nearest_velocity
            )

            if collision:
                tangent1, tangent2 = self.compute_tangents(current_pos, nearest_obstacle, robot_radius + safety_buffer)
                if self.straight_to_goal_safe_tangent_segments(robot, tangent1, tangent2, 2):
                    pass
                else:
                    tangent_points.extend([tangent1, tangent2])
                    path_length1 = np.linalg.norm(np.array(current_pos) - np.array(tangent1)) + np.linalg.norm(np.array(tangent1) - np.array(goal_pos))
                    path_length2 = np.linalg.norm(np.array(current_pos) - np.array(tangent2)) + np.linalg.norm(np.array(tangent2) - np.array(goal_pos))
                    selected_tangent = tangent1 if path_length1 < path_length2 else tangent2
                    if np.sqrt((current_pos[0] - selected_tangent[0])**2 - (current_pos[1] - selected_tangent[1])**2) < 1:
                        pass
                    else:
                        full_path_x.append(selected_tangent[0] * 1.1)
                        full_path_y.append(selected_tangent[1] * 1.1)
                        current_pos = selected_tangent
            else:
                break

            obstacles.pop(nearest_idx)
            obstacle_velocities.pop(nearest_idx)

        full_path_x.append(goal_pos[0])
        full_path_y.append(goal_pos[1])
        return full_path_x, full_path_y
