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

        # Rotor config
        self.rotor_angles = np.deg2rad([30, 90, 150,
                                        210, 270, 330])

        self.rotor_dirs = np.array([1, -1, 1,
                                    -1, 1, -1])

        lx = np.zeros((6,))
        ly = np.zeros((6,))

        self.Kf = np.zeros((4,6))

        for i in range(6):
            self.Kf[0,i] = 1.0
            lx[i] = self.l*np.cos(self.rotor_angles[i])
            self.Kf[2,i] = -lx[i]
            ly[i] = self.l*np.sin(self.rotor_angles[i])
            self.Kf[1,i] = ly[i]
            self.Kf[3,i] = -self.rotor_dirs[i]*self.k_m

        self.K_inv = np.linalg.pinv(self.Kf)

    def compute_des_rpm(self, f, M):
        u = np.hstack([f,M])
        rotors_thrust = self.K_inv @ u
        rotors_thrust = self._clamp(rotors_thrust)
        rotors_speed = np.sqrt(rotors_thrust/self.C_T)
        return rotors_speed

    def _clamp(self, rotors_thrust):
        for i in range(len(rotors_thrust)):
            if self.T_max < rotors_thrust[i]:
                rotors_thrust[i] = self.T_max
            if self.T_min > rotors_thrust[i]:
                rotors_thrust[i] = self.T_min
        return rotors_thrust