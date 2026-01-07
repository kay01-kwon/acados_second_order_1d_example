import numpy as np

class PID_Position_Control:
    """
    PID Position Controller for quadrotor/hexacopter

    Cascaded control structure:
    1. Position error -> desired acceleration
    2. Desired acceleration -> desired thrust and desired attitude (roll, pitch)

    The output is:
    - f_des: desired total thrust (scalar)
    - q_des: desired quaternion attitude
    """

    def __init__(self, DynamicParams, GainParams):
        """
        Initialize PID position controller

        Args:
            DynamicParams: dict with 'm' (mass) and 'g' (gravity)
            GainParams: dict with 'Kp_pos', 'Kd_pos', 'Ki_pos'
        """
        self.m = DynamicParams['m']
        self.g = 9.81

        # Position control gains (3x3 diagonal matrices)
        self.Kp_pos = np.diag(GainParams['Kp_pos'])
        self.Kd_pos = np.diag(GainParams['Kd_pos'])
        self.Ki_pos = np.diag(GainParams['Ki_pos'])

        # Integral error accumulator
        self.pos_error_integral = np.zeros(3)

        # Integral limits (anti-windup)
        self.integral_limit = 5.0  # meters * seconds

    def reset_integral(self):
        """Reset integral error to zero"""
        self.pos_error_integral = np.zeros(3)

    def compute_control(self, s, p_des, v_des, dt, yaw_des=0.0):
        """
        Compute desired thrust and desired rotation matrix (geometric control)

        Args:
            s: current state vector [p(3), v(3), q(4), w(3), ...]
            p_des: desired position [x, y, z] in world frame
            v_des: desired velocity [vx, vy, vz] in world frame
            dt: time step for integral
            yaw_des: desired yaw angle (rad)

        Returns:
            f_des: desired total thrust (scalar, in Newtons)
            R_des: desired rotation matrix (3x3) from body to world
        """
        # Extract current position and velocity
        p = s[0:3]
        v = s[3:6]

        # Position and velocity errors
        pos_error = p_des - p
        vel_error = v_des - v

        # Update integral with anti-windup
        self.pos_error_integral += pos_error * dt
        self.pos_error_integral = np.clip(
            self.pos_error_integral,
            -self.integral_limit,
            self.integral_limit
        )

        # Desired acceleration (PID control law)
        accel_des = (self.Kp_pos @ pos_error +
                     self.Kd_pos @ vel_error +
                     self.Ki_pos @ self.pos_error_integral)

        # Add gravity compensation to get desired force
        # F_des = m * (a_des + g*e_z)
        gravity_compensation = np.array([0.0, 0.0, self.g])
        force_des = self.m * (accel_des + gravity_compensation)

        # Desired thrust magnitude
        f_des = np.linalg.norm(force_des)

        # Saturation: limit thrust to reasonable range (0.1g to 2.5g)
        f_min = 0.1 * self.m * self.g
        f_max = 2.5 * self.m * self.g
        f_des = np.clip(f_des, f_min, f_max)

        # Desired body z-axis direction (thrust direction)
        if f_des > 1e-6:
            z_body_des = force_des / np.linalg.norm(force_des)
        else:
            z_body_des = np.array([0.0, 0.0, 1.0])
            f_des = self.m * self.g  # Hover thrust

        # Compute desired rotation matrix from z-axis and yaw
        R_des = self._compute_desired_rotation_matrix(z_body_des, yaw_des)

        return f_des, R_des

    def _compute_desired_rotation_matrix(self, z_body_des, yaw_des):
        """
        Compute desired rotation matrix from desired body z-axis and yaw angle
        Direct construction for geometric control (more stable than quaternions)

        Args:
            z_body_des: desired body z-axis direction (unit vector in world frame)
            yaw_des: desired yaw angle (rad)

        Returns:
            R_des: desired rotation matrix (3x3) from body to world frame
        """
        # Ensure z_body_des is normalized
        z_b = z_body_des / np.linalg.norm(z_body_des)

        # Vector in horizontal plane aligned with desired yaw
        c_yaw = np.cos(yaw_des)
        s_yaw = np.sin(yaw_des)
        x_c = np.array([c_yaw, s_yaw, 0.0])

        # Compute y-axis: y_b = z_b × x_c (perpendicular to both)
        y_b = np.cross(z_b, x_c)
        y_b_norm = np.linalg.norm(y_b)

        if y_b_norm < 1e-6:
            # Singularity: z_body is vertical (aligned with world z)
            # In this case, just use yaw directly
            x_b = x_c
            y_b = np.cross(z_b, x_b)
            y_b = y_b / np.linalg.norm(y_b)
        else:
            y_b = y_b / y_b_norm

        # Compute x-axis: x_b = y_b × z_b (complete right-handed frame)
        x_b = np.cross(y_b, z_b)

        # Construct rotation matrix R = [x_b | y_b | z_b]
        R_des = np.column_stack([x_b, y_b, z_b])

        return R_des
