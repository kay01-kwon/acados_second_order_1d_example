"""

Hexacopter X + Rotor Dynamics Simulation with Acados
- Full 6DOF dynamics (position + attitude)
- Quaternion-based attitude dynamics
- six rotor 2nd order dynamics with constraints
- COM offset included
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from sympy.stats import moment


@dataclass
class S550_State:
    """
    Complete state of S550
    """
    p: np.ndarray           # position [x, y, z]
    v: np.ndarray           # linear velocity [vx, vy, vz]
    q: np.ndarray           # quaternion [w, x, y, z]
    w: np.ndarray           # angular velocity [x, y, z]
    w_rot: np.ndarray       # rotor speeds [6] (RPM/s)
    alpha_rot: np.ndarray   # rotor acceleration [6] (RPM/s^2)

class S550_Attitude_model:
    """
    S550_Full_6DOF_model

    DynamicParams: m, MoiArray, com_offset
    DroneParams: arm_length, motor_const, moment_const
    RotorParams: p, w_rotor_min, w_rotor_max, alpha_rotor_max, jerk_rotor_max

    State: [p(3), v(3), q(4), omega(3), rotor_omega(6), rotor_omega_dot(6)] = 25 states
    Input: rotor_cmd(6)
    """
    def __init__(self, DynamicParams, DroneParams, RotorParams):
        # Air frame
        self.m = DynamicParams['m']
        self.g = 9.81
        Jxx = DynamicParams['MoiArray'][0]
        Jyy = DynamicParams['MoiArray'][1]
        Jzz = DynamicParams['MoiArray'][2]
        self.J = np.diag([Jxx, Jyy, Jzz])
        self.com_offset = DynamicParams['com_offset']

        # Geometry
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']

        # Rotor config
        self.rotor_angles = np.deg2rad([30, 90, 150,
                                        210, 270, 330])
        self.rotor_dirs = np.array([1, -1, 1,
                                    -1, 1, -1])

        # Rotor param
        self.p1 = RotorParams['p'][0]   # Friction
        self.p2 = RotorParams['p'][1]   # Drag
        self.p3 = RotorParams['p'][2]   # Stiffness

        # Rotor constraints
        self.w_rotor_min = RotorParams['w_rotor_min']   # RPM
        self.w_rotor_max = RotorParams['w_rotor_max']   # RPM
        self.alpha_rotor_max = RotorParams['alpha_rotor_max']   # RPM/s
        self.jerk_rotor_max = RotorParams['jerk_rotor_max']     # RPM/s^2

        self.rotor_pos = np.zeros((6,3))
        self._compute_rotor_pos()

    def _compute_rotor_pos(self):
        """
        Precompute rotor positions
        :return: Rotor positions in COM frame
        """
        for i in range(6):
            theta = self.rotor_angles[i]
            self.rotor_pos[i] = np.array([
                self.l * np.cos(theta) - self.com_offset[0],
                self.l * np.sin(theta) - self.com_offset[1],
                -self.com_offset[2]
            ])

    def pack_state(self, p, v, q, w, w_rot, alpha_rot):
        """ Pack state into vector"""
        return np.concatenate([p, v, q, w, w_rot, alpha_rot])

    def unpack_state(self, s):
        """ Unpack state from vector"""
        p = s[0:3]
        v = s[3:6]
        q = s[6:10]
        w = s[10:13]
        w_rot = s[13:19]
        alpha_rot = s[19:25]
        return p, v, q, w, w_rot, alpha_rot

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to rotation matrix (body to world)
        :param q: quaternion [qw, qx, qy, qz]
        :return: 3x3 rotation matrix
        """
        qw, qx, qy, qz = q

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])

        return R

    def compute_thrust_moment(self, w_rot):
        """Compute thrust and moment"""
        thrust = 0.0
        moment = np.zeros((3,))

        for i in range(6):
            T_i = self.C_T*w_rot[i]**2
            thrust += T_i

            thrust_vec = np.array([0.0, 0.0, T_i])
            moment += np.cross(self.rotor_pos[i],
                                thrust_vec)
            moment[2] += -self.k_m * self.rotor_dirs[i] * T_i
        return thrust, moment

    def dynamics(self, t, s, u):
        """
        Dynamics function for rk4
        :param t: time (Not use, but required for rk4 interface)
        :param s: state vector [25]
        :param u: input (rotor command) [6]
        :return: dsdt state derivative [25]
        """

        p, v, q, w, w_rot, alpha_rot = self.unpack_state(s)

        thrust, moment = self.compute_thrust_moment(w_rot)

        # Position dynamics: dpdt = v
        dpdt = v

        # Translational dynamics: dvdt = R * thrust_body / m - g_world
        R = self.quaternion_to_rotation_matrix(q)
        thrust_body = np.array([0.0, 0.0, thrust])  # Thrust in body frame (z-axis)
        thrust_world = R @ thrust_body  # Transform to world frame
        gravity_world = np.array([0.0, 0.0, -self.m * self.g])  # Gravity in world frame

        dvdt = (thrust_world + gravity_world) / self.m

        # Attitude dynamics: dqdt = 0.5 * q_l * w_quat
        qw, qx, qy, qz = q
        wx, wy, wz = w
        w_quat = np.array([0.0, wx, wy, wz])

        q_l = np.array([
            [qw, -qx, -qy, -qz],
            [qx, qw, -qz, qy],
            [qy, qz, qw, -qx],
            [qz, -qy, qx, qw],
        ])

        dqdt = 0.5 * q_l @ w_quat

        # Rotational dynamics: dwdt = J^-1 * (moment - w x Jw)
        J_w = self.J @ w
        dwdt = np.linalg.solve(self.J, moment - np.cross(w, J_w))

        # Rotor dynamics
        w_dot_rot = np.zeros(6)
        jerk_rot = np.zeros(6)

        for i in range(6):
            damping = self.p1 + self.p2 * w_rot[i]
            j_i_temp = (-damping * alpha_rot[i]
                      + self.p3 * (u[i] - w_rot[i]))

            # Jerk clamping
            j_i_clamp = np.clip(j_i_temp,
                              -self.jerk_rotor_max,
                              self.jerk_rotor_max)

            if alpha_rot[i] >= self.alpha_rotor_max and j_i_clamp > 0:
                w_dot_rot[i] = self.alpha_rotor_max
                j_i_clamp = 0
            elif alpha_rot[i] <= -self.alpha_rotor_max and j_i_clamp < 0:
                w_dot_rot[i] = -self.alpha_rotor_max
                j_i_clamp = 0
            else:
                w_dot_rot[i] = alpha_rot[i]

            jerk_rot[i] = j_i_clamp

        return self.pack_state(dpdt, dvdt, dqdt, dwdt, w_dot_rot, jerk_rot)


