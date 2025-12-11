import numpy as np

class ExampleDoubleIntegrator:
    def __init__(self, DynParam):
        self.m = DynParam['m']

    def ode_func(self, t, x, f):
        A = np.array([[0.0, 1.0],[0.0, 0.0]])
        B = np.array([0.0, 1.0])

        u = f/self.m
        dxdt = A@x + B*u
        return dxdt
