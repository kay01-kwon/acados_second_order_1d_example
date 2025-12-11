import numpy as np

class Plant():
    def __init__(self, DynParam, RotorParam):
        self.m = DynParam['m']
        self.C_T = RotorParam['C_T']

        self.p1 = RotorParam['p'][0]    # Friction
        self.p2 = RotorParam['p'][1]    # Drag
        self.p3 = RotorParam['p'][2]    # Stiffness

    def ode_func(self, t, x, u):
        z = x[0]
        vz = x[1]
        w_rot = x[2]
        alpha_rot = x[3]

        dz = vz
        dvz = 6.0*self.C_T*(w_rot**2)/self.m - 9.81
        dw_rot = alpha_rot
        dalpha_rot = (-(self.p1 + self.p2*w_rot)*alpha_rot
                  + self.p3*(u-w_rot))

        if z <= 0.0 and 6.0*self.C_T*(w_rot**2)/self.m <= 9.81:
            dz = 0.0
            dvz = 0.0

        dxdt = np.hstack((dz, dvz, dw_rot, dalpha_rot))
        return dxdt