import numpy as np

class ControlAllocator:
    def __init__(self, DroneParams):
        self.l = DroneParams['arm_length']
        self.C_T = DroneParams['motor_const']
        self.k_m = DroneParams['moment_const']

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

        print(self.Kf)