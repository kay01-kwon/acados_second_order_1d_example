import numpy as np

class ControlAllocator:
    def __init__(self, DroneParams, RotorParams):
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']
        w_max = RotorParams['w_rotor_max']
        w_min = RotorParams['w_rotor_min']
        self.T_max = self.C_T * w_max**2
        self.T_min = self.C_T * w_min**2

        # Rotor config (hexacopter X configuration)
        self.rotor_angles = np.deg2rad([30, 90, 150,
                                        210, 270, 330])

        self.rotor_dirs = np.array([1, -1, 1,
                                    -1, 1, -1])

        # Build allocation matrix Kf that maps thrusts to [f, Mx, My, Mz]
        # Based on the dynamics in compute_thrust_moment:
        # Mx = sum(y_i * T_i)
        # My = sum(-x_i * T_i)
        # Mz = sum(-k_m * dir_i * T_i)
        self.Kf = np.zeros((4, 6))

        for i in range(6):
            x_i = self.l * np.cos(self.rotor_angles[i])
            y_i = self.l * np.sin(self.rotor_angles[i])

            # Row 0: Total thrust
            self.Kf[0, i] = 1.0

            # Row 1: Moment around X-axis (roll)
            self.Kf[1, i] = y_i

            # Row 2: Moment around Y-axis (pitch)
            self.Kf[2, i] = -x_i

            # Row 3: Moment around Z-axis (yaw)
            self.Kf[3, i] = -self.rotor_dirs[i] * self.k_m

        # Compute pseudo-inverse for control allocation
        self.K_inv = np.linalg.pinv(self.Kf)

    def compute_des_rpm(self, f, M):
        """
        Compute desired rotor speeds from desired thrust and moment

        Args:
            f: desired total thrust (scalar)
            M: desired moment vector [Mx, My, Mz]

        Returns:
            rotor_speeds: desired rotor speeds [RPM] for 6 rotors
        """
        # Stack into control vector [f, Mx, My, Mz]
        u = np.hstack([f, M])

        # Allocate to individual rotor thrusts
        rotors_thrust = self.K_inv @ u

        # Clamp thrusts to feasible range
        rotors_thrust = self._clamp(rotors_thrust)

        # Convert thrust to rotor speed: T = C_T * w^2 => w = sqrt(T / C_T)
        # Ensure non-negative values before sqrt
        rotors_thrust = np.maximum(rotors_thrust, 0.0)
        rotors_speed = np.sqrt(rotors_thrust / self.C_T)

        return rotors_speed

    def _clamp(self, rotors_thrust):
        """Clamp rotor thrusts to physically feasible range"""
        return np.clip(rotors_thrust, self.T_min, self.T_max)