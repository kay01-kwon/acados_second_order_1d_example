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

        # Desired quaternion from desired z-axis and yaw
        q_des = self._compute_desired_quaternion(z_body_des, yaw_des)

        return f_des, q_des

    def _compute_desired_quaternion(self, z_body_des, yaw_des):
        """
        Compute desired quaternion from desired body z-axis and yaw angle
        Using angle-axis representation for numerical stability

        Args:
            z_body_des: desired body z-axis direction (unit vector in world frame)
            yaw_des: desired yaw angle (rad)

        Returns:
            q_des: desired quaternion [qw, qx, qy, qz]
        """
        # Ensure z_body_des is normalized
        z_body_des = z_body_des / np.linalg.norm(z_body_des)

        # Build rotation matrix using ZYX convention
        # First, construct intermediate frame aligned with yaw
        c_yaw = np.cos(yaw_des)
        s_yaw = np.sin(yaw_des)

        # Vector in horizontal plane aligned with yaw
        c_vec = np.array([c_yaw, s_yaw, 0.0])

        # Desired y-axis: perpendicular to both z_body_des and c_vec
        y_body_des = np.cross(z_body_des, c_vec)
        y_norm = np.linalg.norm(y_body_des)

        if y_norm < 1e-6:
            # Singularity: z_body_des is vertical
            # Use yaw to define x-axis directly
            x_body_des = c_vec
            y_body_des = np.cross(z_body_des, x_body_des)
            y_body_des = y_body_des / np.linalg.norm(y_body_des)
        else:
            y_body_des = y_body_des / y_norm

        # Desired x-axis: orthogonal to y and z
        x_body_des = np.cross(y_body_des, z_body_des)
        x_body_des = x_body_des / np.linalg.norm(x_body_des)

        # Construct rotation matrix [x_body | y_body | z_body]
        R_des = np.column_stack([x_body_des, y_body_des, z_body_des])

        # Convert to quaternion using angle-axis
        q_des = self._rotation_matrix_to_quaternion_angleaxis(R_des)

        return q_des

    def _rotation_matrix_to_quaternion_angleaxis(self, R):
        """
        Convert rotation matrix to quaternion using angle-axis representation
        More numerically stable than direct conversion

        Uses: trace(R) for angle, (R - R^T) for axis

        Args:
            R: 3x3 rotation matrix

        Returns:
            q: quaternion [qw, qx, qy, qz]
        """
        # Compute rotation angle from trace
        # trace(R) = 1 + 2*cos(theta)
        trace_R = np.trace(R)
        cos_theta = (trace_R - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical safety
        theta = np.arccos(cos_theta)

        # Handle small angle case (near identity)
        if theta < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        # Compute rotation axis from skew-symmetric part
        # R - R^T = 2*sin(theta)*[k]_x where k is the unit axis
        sin_theta = np.sin(theta)

        if abs(sin_theta) > 1e-6:
            # Extract axis from skew-symmetric matrix
            k_x = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
            k_y = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
            k_z = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)
            k = np.array([k_x, k_y, k_z])

            # Normalize axis (should already be unit, but ensure numerical stability)
            k = k / np.linalg.norm(k)

            # Convert angle-axis to quaternion
            # q = [cos(theta/2), sin(theta/2)*k]
            half_theta = theta / 2.0
            qw = np.cos(half_theta)
            sin_half = np.sin(half_theta)
            qx = sin_half * k[0]
            qy = sin_half * k[1]
            qz = sin_half * k[2]

            q = np.array([qw, qx, qy, qz])
        else:
            # theta ≈ π, use alternative method
            # Find the column of R with largest diagonal element
            diag = np.diag(R)
            k_idx = np.argmax(diag)

            if k_idx == 0:
                k = np.array([R[0, 0] + 1, R[1, 0], R[2, 0]])
            elif k_idx == 1:
                k = np.array([R[0, 1], R[1, 1] + 1, R[2, 1]])
            else:
                k = np.array([R[0, 2], R[1, 2], R[2, 2] + 1])

            k = k / np.linalg.norm(k)

            # For theta ≈ π: q ≈ [0, k]
            half_theta = theta / 2.0
            qw = np.cos(half_theta)
            sin_half = np.sin(half_theta)
            q = np.array([qw, sin_half * k[0], sin_half * k[1], sin_half * k[2]])

        # Normalize quaternion
        q = q / np.linalg.norm(q)

        return q
