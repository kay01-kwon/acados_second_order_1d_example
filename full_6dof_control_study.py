import numpy as np
import matplotlib.pyplot as plt
from model.S550_sim_model import S550_Attitude_model
from ode_solver import custom_rk4
from PID_control.pid_position import PID_Position_Control
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

    # ========== Parameters ==========
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

    # Position control gains
    position_gain_params = {'Kp_pos': np.array([5.0, 5.0, 10.0]),
                           'Kd_pos': np.array([4.0, 4.0, 6.0]),
                           'Ki_pos': np.array([0.1, 0.1, 0.2])}

    # Attitude control gains
    attitude_gain_params = {'Kp': np.array([10., 10., 10.]),
                           'Kd': np.array([5., 5., 5.]),
                           'Ki': np.array([0.01, 0.01, 0.01])}

    # ========== Initialize Model and Controllers ==========
    sim_model = S550_Attitude_model(DynamicParams=dynamic_params,
                              DroneParams=drone_params,
                              RotorParams=rotor_params)

    pid_pos_control = PID_Position_Control(DynamicParams=dynamic_params,
                                           GainParams=position_gain_params)

    pid_att_control = PID_control(DynamicParams=dynamic_params,
                                  GainParams=attitude_gain_params)

    control_allocator = ControlAllocator(DroneParams=drone_params,
                                         RotorParams=rotor_params)

    # ========== Simulation Parameters ==========
    tf = 15.0
    dt = 0.01
    t = np.arange(0, tf, dt)

    w_rotor_idle = 2000.0

    # ========== Initial Conditions ==========
    p_init = np.array([0.0, 0.0, 0.0])  # Initial position [x, y, z]
    v_init = np.array([0.0, 0.0, 0.0])  # Initial velocity [vx, vy, vz]
    q_init = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion (identity)
    w_init = np.array([0.0, 0.0, 0.0])  # Initial angular velocity
    w_rotor_init = np.array([w_rotor_idle]*6)  # Initial rotor speeds
    alpha_rotor_init = np.array([0.0]*6)  # Initial rotor accelerations
    s = np.concatenate((p_init, v_init, q_init, w_init, w_rotor_init, alpha_rotor_init))

    # ========== Trajectory Definition ==========
    def get_desired_trajectory(t_current):
        """
        Define desired trajectory as a function of time
        Returns: p_des, v_des, yaw_des
        """
        if t_current < 3.0:
            # Hover at origin
            p_des = np.array([0.0, 0.0, 1.0])
            v_des = np.array([0.0, 0.0, 0.0])
            yaw_des = 0.0
        elif t_current < 8.0:
            # Move to [2, 2, 1.5]
            p_des = np.array([2.0, 2.0, 1.5])
            v_des = np.array([0.0, 0.0, 0.0])
            yaw_des = 0.0
        elif t_current < 13.0:
            # Move to [2, -2, 2]
            p_des = np.array([2.0, -2.0, 2.0])
            v_des = np.array([0.0, 0.0, 0.0])
            yaw_des = np.pi/4  # 45 degrees
        else:
            # Return to origin
            p_des = np.array([0.0, 0.0, 1.0])
            v_des = np.array([0.0, 0.0, 0.0])
            yaw_des = 0.0

        return p_des, v_des, yaw_des

    # ========== Data Storage ==========
    pos_hist = []
    vel_hist = []
    pos_des_hist = []
    roll_hist = []
    pitch_hist = []
    yaw_hist = []
    w_rotor_hist = []
    alpha_rotor_hist = []

    # ========== Main Simulation Loop ==========
    for i in range(len(t)-1):

        # Get current state
        p, v, q, w, w_rot, alpha_rot = sim_model.unpack_state(s)
        roll, pitch, yaw = quaternion_to_euler(q)

        # Store data
        pos_hist.append(p.copy())
        vel_hist.append(v.copy())
        roll_hist.append(np.rad2deg(roll))
        pitch_hist.append(np.rad2deg(pitch))
        yaw_hist.append(np.rad2deg(yaw))
        w_rotor_hist.append(w_rot.copy())
        alpha_rotor_hist.append(alpha_rot.copy())

        # Get desired trajectory
        p_des, v_des, yaw_des = get_desired_trajectory(t[i])
        pos_des_hist.append(p_des.copy())

        # ===== Cascaded Control =====
        # 1. Position controller: compute desired thrust and attitude
        f_des, q_des = pid_pos_control.compute_control(s, p_des, v_des, dt, yaw_des)

        # 2. Attitude controller: compute desired moment
        # Create reference state for attitude controller
        ref = q_des
        M_des = pid_att_control.set(s, ref)

        # 3. Control allocation: compute rotor speeds
        cmd = control_allocator.compute_des_rpm(f_des, M_des)

        # ===== Step dynamics forward =====
        t_ode = [t[i], t[i+1]]
        s = custom_rk4.do_step(sim_model.dynamics, s, cmd, tspan=t_ode)

    # ========== Post-processing ==========
    pos_hist = np.array(pos_hist)
    vel_hist = np.array(vel_hist)
    pos_des_hist = np.array(pos_des_hist)
    roll_hist = np.array(roll_hist)
    pitch_hist = np.array(pitch_hist)
    yaw_hist = np.array(yaw_hist)
    w_rotor_hist = np.array(w_rotor_hist)
    alpha_rotor_hist = np.array(alpha_rotor_hist)
    t_plot = t[:-1]

    # ========== Plotting ==========
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # Plot Position X
    axs[0, 0].plot(t_plot, pos_hist[:, 0], 'b-', label='Actual', linewidth=2)
    axs[0, 0].plot(t_plot, pos_des_hist[:, 0], 'r--', label='Desired', linewidth=2)
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('X Position [m]')
    axs[0, 0].set_title('X Position Tracking')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Position Y
    axs[0, 1].plot(t_plot, pos_hist[:, 1], 'b-', label='Actual', linewidth=2)
    axs[0, 1].plot(t_plot, pos_des_hist[:, 1], 'r--', label='Desired', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Y Position [m]')
    axs[0, 1].set_title('Y Position Tracking')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Position Z
    axs[0, 2].plot(t_plot, pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    axs[0, 2].plot(t_plot, pos_des_hist[:, 2], 'r--', label='Desired', linewidth=2)
    axs[0, 2].set_xlabel('Time [s]')
    axs[0, 2].set_ylabel('Z Position [m]')
    axs[0, 2].set_title('Z Position Tracking')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot Velocity
    axs[1, 0].plot(t_plot, vel_hist[:, 0], 'r-', label='vx', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 1], 'g-', label='vy', linewidth=2)
    axs[1, 0].plot(t_plot, vel_hist[:, 2], 'b-', label='vz', linewidth=2)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Velocity [m/s]')
    axs[1, 0].set_title('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Attitude Angles
    axs[1, 1].plot(t_plot, roll_hist, 'r-', label='Roll', linewidth=2)
    axs[1, 1].plot(t_plot, pitch_hist, 'g-', label='Pitch', linewidth=2)
    axs[1, 1].plot(t_plot, yaw_hist, 'b-', label='Yaw', linewidth=2)
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Angle [deg]')
    axs[1, 1].set_title('Attitude Angles')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 3D Trajectory
    ax_3d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], 'b-', label='Actual', linewidth=2)
    ax_3d.plot(pos_des_hist[:, 0], pos_des_hist[:, 1], pos_des_hist[:, 2], 'r--', label='Desired', linewidth=2)
    ax_3d.scatter(pos_hist[0, 0], pos_hist[0, 1], pos_hist[0, 2], c='g', marker='o', s=100, label='Start')
    ax_3d.scatter(pos_hist[-1, 0], pos_hist[-1, 1], pos_hist[-1, 2], c='r', marker='x', s=100, label='End')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.set_title('3D Trajectory')
    ax_3d.legend()
    ax_3d.grid(True)

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

    # Position Error
    pos_error = np.linalg.norm(pos_hist - pos_des_hist, axis=1)
    axs[2, 2].plot(t_plot, pos_error, 'b-', linewidth=2)
    axs[2, 2].set_xlabel('Time [s]')
    axs[2, 2].set_ylabel('Position Error [m]')
    axs[2, 2].set_title('Position Tracking Error')
    axs[2, 2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print final statistics
    print("\n========== Simulation Results ==========")
    print(f"Final position: [{pos_hist[-1, 0]:.3f}, {pos_hist[-1, 1]:.3f}, {pos_hist[-1, 2]:.3f}] m")
    print(f"Desired position: [{pos_des_hist[-1, 0]:.3f}, {pos_des_hist[-1, 1]:.3f}, {pos_des_hist[-1, 2]:.3f}] m")
    print(f"Final position error: {pos_error[-1]:.4f} m")
    print(f"Mean position error: {np.mean(pos_error):.4f} m")
    print(f"Max position error: {np.max(pos_error):.4f} m")
    print("========================================\n")

if __name__ == '__main__':
    main()
