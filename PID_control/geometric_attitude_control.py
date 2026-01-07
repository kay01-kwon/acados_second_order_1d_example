import numpy as np

class GeometricAttitudeControl:
    """
    Geometric attitude control on SO(3)
    Based on Lee et al. "Geometric tracking control of a quadrotor UAV on SE(3)"

    Works directly with rotation matrices, avoiding singularities of Euler angles
    and numerical issues with quaternions.
    """

    def __init__(self, DynamicParams, GainParams):
        """
        Initialize geometric attitude controller

        Args:
            DynamicParams: dict with 'MoiArray' (moment of inertia)
            GainParams: dict with 'Kp' and 'Kd' (attitude and rate gains)
        """
        J_array = DynamicParams['MoiArray']
        self.J = np.diag(J_array)

        # Attitude error gains (3x3 diagonal)
        self.Kp = np.diag(GainParams['Kp'])
        self.Kd = np.diag(GainParams['Kd'])

    def compute_control(self, s, R_des, w_des=np.zeros(3), tau_ext=np.zeros(3)):
        """
        Compute control moment using geometric control law

        Args:
            s: current state vector [p(3), v(3), q(4), w(3), w_rot(6), alpha_rot(6)]
            R_des: desired rotation matrix (3x3, body to world)
            w_des: desired angular velocity (default: zeros)
            tau_ext: external disturbance torque (default: zeros)

        Returns:
            M: control moment vector [Mx, My, Mz] in body frame
        """
        # Extract current quaternion and angular velocity
        q = s[6:10]
        w = s[10:13]

        # Convert current quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(q)

        # Compute rotation error using vee map
        # e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)
        e_R = self._compute_rotation_error(R, R_des)

        # Angular velocity error
        e_w = w - w_des

        # Geometric control law (from Lee et al. 2010)
        # M = -Kp*e_R - Kd*e_w + w × J*w - tau_ext
        J_w = self.J @ w
        M = -self.Kp @ e_R - self.Kd @ e_w + np.cross(w, J_w) - tau_ext

        return M

    def _quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix (body to world)

        Args:
            q: quaternion [qw, qx, qy, qz]

        Returns:
            R: 3x3 rotation matrix
        """
        qw, qx, qy, qz = q

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])

        return R

    def _compute_rotation_error(self, R, R_des):
        """
        Compute rotation error on SO(3) using the vee map

        e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)

        This gives the error vector that points along the axis of rotation
        needed to align R with R_des, scaled by the sine of the rotation angle.

        Args:
            R: current rotation matrix (3x3)
            R_des: desired rotation matrix (3x3)

        Returns:
            e_R: rotation error vector (3,)
        """
        # Compute skew-symmetric error matrix
        R_err = R_des.T @ R - R.T @ R_des

        # Extract vector using vee map (inverse of hat/skew operator)
        # For skew-symmetric matrix [0 -a3 a2; a3 0 -a1; -a2 a1 0]
        # vee gives [a1, a2, a3]
        e_R = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])

        return e_R

    def _skew_symmetric(self, v):
        """
        Create skew-symmetric matrix from vector (hat map)

        Args:
            v: vector [v1, v2, v3]

        Returns:
            S: 3x3 skew-symmetric matrix such that S @ x = v × x
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
