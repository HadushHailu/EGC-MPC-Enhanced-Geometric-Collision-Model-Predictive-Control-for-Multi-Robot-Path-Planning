import yaml
from simulation.robot_states import RobotStates
from simulation.egc_mpc_controller import EgcMpcController
from simulation.egc_mpc_simulation import EgcMpcSimulation
import threading
import signal
import sys

stop_event = threading.Event()

def signal_handler(sig, frame):
    print("\nReceived interrupt signal, shutting down gracefully...")
    stop_event.set()
    sys.exit(0)

def load_config(file_path="robots_config.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("EGC-MPC Simulation Starting...")
    config = load_config()

    robot_states = RobotStates()

    controller = EgcMpcController(config, robot_states, stop_event)
    controller.start_robots()

    simulation = EgcMpcSimulation(robot_states, config, stop_event)
    sim_thread = threading.Thread()
    sim_thread.start()

    # Or just run in main thread
    simulation.run()

if __name__ == "__main__":
    main()
