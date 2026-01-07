import numpy as np
from model.S550_sim_model import S550_Attitude_model
from ode_solver import custom_rk4
from PID_control.pid_attitude import PID_control

def main():

    dynamic_params = {'m': 2.9,
                      'MoiArray': np.array([0.06, 0.06, 0.09]),
                      'com_offset': np.array([0., 0., 0.])}

    drone_params = {'arm_length': 0.265,
                    'motor_const': 1.4657e-7,
                    'moment_const': 0.01569}

    rotor_params = {'p': np.array([25.16687, 0.003933, 515.605]),
                    'w_rotor_min': 1800,
                    'w_rotor_max': 7300,
                    'alpha_rotor_max': 15e3,
                    'jerk_rotor_max': 250e3,}

    gain_params = {'Kp': np.array([10., 10., 10.]),
                   'Kd': np.array([5., 5., 5.]),
                   'Ki': np.array([0.01, 0.01, 0.01])}


    sim_model = S550_Attitude_model(DynamicParams=dynamic_params,
                              DroneParams=drone_params,
                              RotorParams=rotor_params)

    pid_control = PID_control(DynamicParams=dynamic_params,
                              GainParams=gain_params)
    tf = 10.0
    dt = 0.01
    t = np.arange(0, tf, dt)

    w_rotor_idle = 2000.0

    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    w_init = np.array([0.0, 0.0, 0.0])
    w_rotor_init = np.array([w_rotor_idle]*6)
    alpha_rotor_init = np.array([0.0]*6)
    s = np.concatenate((q_init, w_init, w_rotor_init, alpha_rotor_init))

    cmd = np.array([w_rotor_idle]*6)

    for i in range(len(t)-1):

        t_ode = [t[i], t[i+1]]

        s = custom_rk4.do_step(sim_model.dynamics,
                               s, cmd, tspan=t_ode)

if __name__ == '__main__':
    main()