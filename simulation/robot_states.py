import threading

class RobotStates:
    def __init__(self):
        self._states = {}
        self._lock = threading.Lock()

    def update(self, robot_id, state_dict):
        """Update the state of a specific robot."""
        with self._lock:
            self._states[robot_id] = state_dict

    def get(self, robot_id):
        """Get the state of a specific robot."""
        with self._lock:
            return self._states.get(robot_id, None)

    def get_all(self):
        """Get the states of all robots (for visualization)."""
        with self._lock:
            return dict(self._states)

    def get_others(self, robot_id):
        """Get the states of all robots except the given one (for avoidance)."""
        with self._lock:
            return {rid: state for rid, state in self._states.items() if rid != robot_id}
