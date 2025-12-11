import numpy as np
from matplotlib import pyplot as plt

from ode_solver import custom_rk4
from model.example_double_integrator import ExampleDoubleIntegrator

def main():

    m = 1
    DynParam = {'m': m}

    plant = ExampleDoubleIntegrator(DynParam)

    x = np.array([0.0, 0.0])

    f = 1

    t_final = 10.0
    dt = 0.01
    time_sim = np.arange(0, t_final, dt)

    x_array = np.zeros((x.shape[0], len(time_sim)))

    for i in range(len(time_sim)-1):

        t_ode = [time_sim[i], time_sim[i+1]]

        x_next = custom_rk4.do_step(plant.ode_func,
                                    x, f, tspan=t_ode)

        x = x_next
        x_array[:, i+1] = x

    plt.figure()
    plt.plot(time_sim, x_array[0,:])
    plt.show()

    plt.figure()
    plt.plot(time_sim, x_array[1,:])
    plt.show()


if __name__ == '__main__':
    main()