import numpy as np
import casadi as ca
from simulation.robot_dynamics import unicycle_dynamics
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mpc_debug.log"),
        logging.StreamHandler()
    ]
)

class PointRobotMPC:
    def __init__(self, dt=0.1, N=20):
        self.dt = dt
        self.N = N
        self.v_max = 2.0
        self.omega_max = np.pi

        self.opti = ca.Opti()

        # Decision Variables
        self.X = self.opti.variable(3, N + 1)  # state: [x, y, theta]
        self.U = self.opti.variable(2, N)      # control: [v, omega]

        self.x, self.y, self.theta = self.X[0, :], self.X[1, :], self.X[2, :]
        self.v, self.omega = self.U[0, :], self.U[1, :]

        # Parameters
        self.X0 = self.opti.parameter(3)           # initial state
        self.X_ref = self.opti.parameter(3, N)     # reference trajectory

        self.setup_dynamics()
        self.setup_objective()
        self.setup_constraints()

        p_opts = {"print_time": False}
        s_opts = {"max_iter": 300, "print_level": 0}
        self.opti.solver("ipopt", p_opts, s_opts)

        self.logger = logging.getLogger("PointRobotMPC")

    def setup_dynamics(self):
        for k in range(self.N):
            next_state = unicycle_dynamics(self.X[:, k], self.U[:, k], self.dt)
            self.opti.subject_to(self.X[:, k + 1] == next_state)

    def setup_objective(self):
        cost = 0
        for k in range(self.N):
            pos_err = (self.x[k] - self.X_ref[0, k])**2 + (self.y[k] - self.X_ref[1, k])**2
            theta_err = (self.theta[k] - self.X_ref[2, k])**2
            control_effort = self.v[k]**2 + self.omega[k]**2
            cost += 10 * pos_err + 1 * theta_err + 0.1 * control_effort

        # Optional: Smooth control input
        for k in range(1, self.N):
            cost += 0.05 * ((self.v[k] - self.v[k-1])**2 + (self.omega[k] - self.omega[k-1])**2)

        self.opti.minimize(cost)

    def setup_constraints(self):
        self.opti.subject_to(self.X[:, 0] == self.X0)

        self.opti.subject_to(self.v <= self.v_max)
        self.opti.subject_to(self.v >= -self.v_max)
        self.opti.subject_to(self.omega <= self.omega_max)
        self.opti.subject_to(self.omega >= -self.omega_max)

    def solve(self, x, y, theta, path_x, path_y):
        self.logger.info("----- MPC Solve Start -----")
        self.logger.info(f"Start: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
        self.logger.info(f"Path len: {len(path_x)}")

        # Pad path if shorter than N
        path_x = np.pad(path_x, (0, self.N - len(path_x)), 'edge')[:self.N]
        path_y = np.pad(path_y, (0, self.N - len(path_y)), 'edge')[:self.N]
        ref_theta = np.arctan2(np.diff(np.append(path_y, path_y[-1])),
                               np.diff(np.append(path_x, path_x[-1])))

        self.opti.set_value(self.X0, [x, y, theta])
        self.opti.set_value(self.X_ref, np.vstack((path_x, path_y, ref_theta)))

        # Initial guess
        self.opti.set_initial(self.X, np.tile([x, y, theta], (self.N + 1, 1)).T)
        self.opti.set_initial(self.U, np.zeros((2, self.N)))

        try:
            sol = self.opti.solve()
            v = sol.value(self.v[0])
            omega = sol.value(self.omega[0])
            self.logger.info(f"v = {v:.2f}, omega = {omega:.2f}")
            return v, omega
        except RuntimeError:
            self.logger.error("MPC solve failed!")
            return 0.0, 0.0
