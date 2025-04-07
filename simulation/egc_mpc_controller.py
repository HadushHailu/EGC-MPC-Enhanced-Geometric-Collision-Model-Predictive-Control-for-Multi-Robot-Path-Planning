# simulation/egc_mpc_controller.py

from simulation.robot import Robot

class EgcMpcController:
    def __init__(self, config, robot_states, stop_event):
        self.config = config
        self.robot_states = robot_states
        self.robots = []

        robot_dim = config["robot_dim"]
        robot_list = config["robots"]

        for idx, robot_config in enumerate(robot_list):
            robot_id = f"R{idx}"
            robot = Robot(robot_id, robot_config, robot_dim, robot_states, stop_event)
            self.robots.append(robot)

    def start_robots(self):
        for robot in self.robots:
            robot.start()
