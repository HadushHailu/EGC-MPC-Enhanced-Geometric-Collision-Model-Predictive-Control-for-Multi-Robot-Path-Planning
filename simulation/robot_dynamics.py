import casadi as ca

def unicycle_dynamics(x, u, dt):
    """
    Unicycle model for point robot.
    
    Args:
        x: casadi MX[3] (x, y, theta)
        u: casadi MX[2] (v, omega)
        dt: float

    Returns:
        casadi MX[3]: next state [x, y, theta]
    """
    theta = x[2]
    x_next = x[0] + u[0] * ca.cos(theta) * dt
    y_next = x[1] + u[0] * ca.sin(theta) * dt
    theta_next = theta + u[1] * dt
    return ca.vertcat(x_next, y_next, theta_next)