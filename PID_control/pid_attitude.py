import numpy as np

class PID_control:
    def __init__(self, DynamicParams, GainParams):

        J_array = DynamicParams['MoiArray']
        self.J = np.diag(J_array)
        self.Kp = np.diag(GainParams['Kp'])
        self.Ki = np.diag(GainParams['Ki'])
        self.Kd = np.diag(GainParams['Kd'])

    def set(self, s, ref, tau = np.zeros((3,))):
        q = s[0:4]  # body -> world
        w = s[4:7]

        q_ref = ref[0:4]    # ref -> world
        q_ref_conj = self._conjugate(q_ref)

        q_tilde = self._otimes(q, q_ref_conj)

        qw_tilde = q_tilde[0]
        q_vec = q_tilde[1:]

        q_vec_signum = self._signum(qw_tilde) * q_vec

        accel_ang = -self.Kp @ q_vec_signum - self.Kd @ w

        M_control = self.J @ accel_ang - tau

        return M_control


    def _conjugate(self, q):
        qw, qx, qy, qz = q
        q_conj = np.array([qw, -qx, -qy, -qz])
        return q_conj

    def _otimes(self, q1, q2):
        qw, qx, qy, qz = q1

        q1_mat = np.array([
            [qw, -qx, -qy, -qz],
            [qx, qw, -qz, qy],
            [qy, qy, qw, -qx],
            [qz, -qy, qz, qw]
        ])

        return q1_mat @ q2

    def _signum(self, qw):
        return 1 if qw >= 0 else -1