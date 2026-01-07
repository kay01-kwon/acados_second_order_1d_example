import numpy as np

from ode_solver import custom_rk4
from ocp.example_ocp import ExampleOcpSolver
from model.example_sim_model import Plant

from matplotlib import pyplot as plt

def ParamDecl():

    # Dynamic parameter
    m = 3.

    # Rotor Parameter
    C_T = 1.465e-7
    p1 = 12.17
    p2 = 0.0012
    p3 = 137.3
    p = np.array([p1, p2, p3])

    # Ocp Parameter
    t_horizon = 0.20
    n_nodes = 20

    Qmat = np.diag([10.0, 5.0, 1e-9, 1e-12])
    Rmat = 1e-10

    u_min = 2000.0
    u_max = 8200.0

    alpha_max = 15000.0

    DynParam = {'m':m}

    RotorParam = {'C_T':C_T,
                'p':p}

    OcpParam = {'t_horizon':t_horizon,
                'n_nodes':n_nodes,
                'Qmat':Qmat,
                'Rmat':Rmat,
                'u_min':u_min,
                'u_max':u_max,
                'alpha_max':alpha_max}
    return DynParam, RotorParam, OcpParam



if __name__ == '__main__':

    DynParam, RotorParam, OcpParam = ParamDecl()
    ocp = ExampleOcpSolver(DynParam=DynParam, RotorParam=RotorParam, OcpParam=OcpParam)
    sim = Plant(DynParam=DynParam, RotorParam=RotorParam)

    x = np.array([0.0, 0.0, OcpParam['u_min'], 0.0])

    t_final = 100.0
    dt = 0.01
    time_sim = np.arange(0.0, t_final, dt)

    x_array = np.zeros((x.shape[0], len(time_sim)))
    x_array[:, 0] = x

    ref = np.array([1.0, 0.0, 0.0, 0.0])

    u = OcpParam['u_min']

    u_array = np.zeros((len(time_sim),))
    u_array[0] = u

    for i in range(len(time_sim)-1):

        t_ode = [time_sim[i], time_sim[i+1]]

        u, status = ocp.ocp_solve(x, ref, u)
        u_array[i+1] = u[0]
        ref[2] = u[0]
        # print(ref)

        x_next = custom_rk4.do_step(sim.ode_func,
                                    x, u, tspan=t_ode)

        x = x_next
        x_array[:,i+1] = x_next

    plt.figure()
    plt.plot(time_sim, x_array[0,:])

    plt.figure()
    plt.plot(time_sim, x_array[1,:])

    plt.figure()
    plt.plot(time_sim, x_array[2,:])

    plt.figure()
    plt.plot(time_sim, x_array[3,:])

    plt.figure()
    plt.plot(time_sim, u_array)
    plt.show()

    print(u_array[-1])