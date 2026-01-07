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
        Compute desired thrust and attitude quaternion

        Args:
            s: current state vector [p(3), v(3), q(4), w(3), ...]
            p_des: desired position [x, y, z] in world frame
            v_des: desired velocity [vx, vy, vz] in world frame
            dt: time step for integral
            yaw_des: desired yaw angle (rad)

        Returns:
            f_des: desired total thrust (scalar, in Newtons)
            q_des: desired quaternion [qw, qx, qy, qz]
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

        # Desired body z-axis direction (thrust direction)
        if f_des > 1e-6:
            z_body_des = force_des / f_des
        else:
            z_body_des = np.array([0.0, 0.0, 1.0])
            f_des = self.m * self.g  # Hover thrust

        # Desired quaternion from desired z-axis and yaw
        q_des = self._compute_desired_quaternion(z_body_des, yaw_des)

        return f_des, q_des

    def _compute_desired_quaternion(self, z_body_des, yaw_des):
        """
        Compute desired quaternion from desired body z-axis and yaw angle

        Args:
            z_body_des: desired body z-axis direction (unit vector in world frame)
            yaw_des: desired yaw angle (rad)

        Returns:
            q_des: desired quaternion [qw, qx, qy, qz]
        """
        # Ensure z_body_des is normalized
        z_body_des = z_body_des / np.linalg.norm(z_body_des)

        # Desired x-axis in world frame (from yaw)
        x_world = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

        # Compute desired y-axis: y_body = z_body x x_world_projection
        y_body_des = np.cross(z_body_des, x_world)
        y_body_norm = np.linalg.norm(y_body_des)

        if y_body_norm > 1e-6:
            y_body_des = y_body_des / y_body_norm
        else:
            # Edge case: z_body aligned with x_world
            y_body_des = np.array([0.0, 1.0, 0.0])

        # Compute desired x-axis: x_body = y_body x z_body
        x_body_des = np.cross(y_body_des, z_body_des)

        # Construct rotation matrix
        R_des = np.column_stack([x_body_des, y_body_des, z_body_des])

        # Convert rotation matrix to quaternion
        q_des = self._rotation_matrix_to_quaternion(R_des)

        return q_des

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion

        Args:
            R: 3x3 rotation matrix

        Returns:
            q: quaternion [qw, qx, qy, qz]
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        # Normalize quaternion
        q = np.array([qw, qx, qy, qz])
        q = q / np.linalg.norm(q)

        return q
