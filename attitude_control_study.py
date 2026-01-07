import numpy as np
import matplotlib.pyplot as plt
from model.S550_sim_model import S550_Attitude_model
from ode_solver import custom_rk4
from PID_control.pid_attitude import PID_control
from PID_control.control_allocator import ControlAllocator

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    :param q: quaternion [qw, qx, qy, qz]
    :return: Euler angles [roll, pitch, yaw] in radians
    """
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

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

    control_allocator = ControlAllocator(DroneParams=drone_params,
                                         RotorParams=rotor_params)

    tf = 10.0
    dt = 0.01
    t = np.arange(0, tf, dt)

    w_rotor_idle = 2000.0

    # Initial conditions
    p_init = np.array([0.0, 0.0, 0.0])  # Initial position [x, y, z]
    v_init = np.array([0.0, 0.0, 0.0])  # Initial velocity [vx, vy, vz]
    q_init = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion (identity)
    w_init = np.array([0.0, 0.0, 0.0])  # Initial angular velocity
    w_rotor_init = np.array([w_rotor_idle]*6)  # Initial rotor speeds
    alpha_rotor_init = np.array([0.0]*6)  # Initial rotor accelerations
    s = np.concatenate((p_init, v_init, q_init, w_init, w_rotor_init, alpha_rotor_init))

    ref = np.array([1.0, 0.0, 0.0, 0.0])
    cmd = np.array([w_rotor_idle]*6)

    # Data storage
    pos_hist = []
    vel_hist = []
    roll_hist = []
    pitch_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []

    for i in range(len(t)-1):

        # Store current state data
        p, v, q, w, w_rot, alpha_rot = sim_model.unpack_state(s)
        roll, pitch, yaw = quaternion_to_euler(q)

        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        roll_hist.append(np.rad2deg(roll))
        pitch_hist.append(np.rad2deg(pitch))
        w_rotor_hist.append(w_rot.copy())
        alpha_rotor_hist.append(alpha_rot.copy())

        t_ode = [t[i], t[i+1]]

        M = pid_control.set(s,ref)
        cmd = control_allocator.compute_des_rpm(f,M)

        s = custom_rk4.do_step(sim_model.dynamics,
                               s, cmd, tspan=t_ode)

    # Convert to numpy arrays
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    roll_hist = np.array(roll_hist)
    pitch_hist = np.array(pitch_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    alpha_rotor_hist = np.array(alpha_rotor_hist)
    t_plot = t[:-1]

    # Create plots
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    # Plot Position
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'r-', label='x', linewidth=2)
    axs[0, 0].plot(t_plot, pos_hist[:, 1], 'g-', label='y', linewidth=2)
    axs[0, 0].plot(t_plot, pos_hist[:, 2], 'b-', label='z', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Position [m]')
    axs[0, 0].set_title('Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Velocity
    axs[0, 1].plot(t_plot, vel_hist[:, 0], 'r-', label='vx', linewidth=2)
    axs[0, 1].plot(t_plot, vel_hist[:, 1], 'g-', label='vy', linewidth=2)
    axs[0, 1].plot(t_plot, vel_hist[:, 2], 'b-', label='vz', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Velocity [m/s]')
    axs[0, 1].set_title('Velocity')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Roll
    axs[1, 0].plot(t_plot, roll_hist, 'b-', linewidth=2)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Roll [deg]')
    axs[1, 0].set_title('Roll Angle')
    axs[1, 0].grid(True)

    # Plot Pitch
    axs[1, 1].plot(t_plot, pitch_hist, 'r-', linewidth=2)
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Pitch [deg]')
    axs[1, 1].set_title('Pitch Angle')
    axs[1, 1].grid(True)

    # Plot Rotor Speeds
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 0], label='Rotor 1')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 1], label='Rotor 2')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 2], label='Rotor 3')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 3], label='Rotor 4')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 4], label='Rotor 5')
    axs[2, 0].plot(t_plot, w_rotor_hist[:, 5], label='Rotor 6')
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].set_ylabel('Rotor Speed [RPM]')
    axs[2, 0].set_title('Rotor Speeds')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Plot Rotor Accelerations
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 0], label='Rotor 1')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 1], label='Rotor 2')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 2], label='Rotor 3')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 3], label='Rotor 4')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 4], label='Rotor 5')
    axs[2, 1].plot(t_plot, alpha_rotor_hist[:, 5], label='Rotor 6')
    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].set_ylabel('Rotor Acceleration [RPM/s]')
    axs[2, 1].set_title('Rotor Accelerations')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()