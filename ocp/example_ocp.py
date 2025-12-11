import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcpSolver, AcadosOcp
from model.example_model import ExampleModel
import casadi as cs

class ExampleOcpSolver():
    def __init__(self,DynParam, RotorParam, OcpParam):

        self.ocp = AcadosOcp()

        # Instantiate model object
        model_obj = ExampleModel(DynParam=DynParam, RotorParam=RotorParam)
        model = model_obj.export_acados_model()

        # Insert acados model into ocp
        self.ocp.model = model

        t_horizon = OcpParam['t_horizon']
        n_nodes = OcpParam['n_nodes']

        Qmat = OcpParam['Qmat']
        Rmat = OcpParam['Rmat']

        self.u_min = OcpParam['u_min']
        self.u_max = OcpParam['u_max']

        self.alpha_max = OcpParam['alpha_max']
        self.alpha_min = -self.alpha_max

        self.ocp.dims.N = n_nodes

        nx = model.x.rows()
        nu = model.u.rows()
        self.nu = nu
        ny = nx + nu

        # 1. Cost setup
        # 1.1 Type of cost function
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # 1.2 Vx setup
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        self.ocp.cost.Vx_e = np.eye(nx)

        # 1.3 Vu setup
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[-nu:,-nu:] = np.eye(nu)

        # 1.4 Weight
        self.ocp.cost.W = block_diag(Qmat, Rmat)
        self.ocp.cost.W_e = Qmat

        # 1.5 Reference setup
        x0 = np.array([0.0, 0.0, self.u_min, 0.0])
        self.ocp.cost.yref = np.concatenate((x0, np.zeros(nu,)))
        self.ocp.cost.yref_e = x0

        # 2. Constraints setup
        self.ocp.constraints.x0 = x0
        self.ocp.constraints.lbx = np.array([self.u_min, self.alpha_min])
        self.ocp.constraints.ubx = np.array([self.u_max, self.alpha_max])
        self.ocp.constraints.idxbx = np.array([2, 3])

        self.ocp.constraints.lbu = np.array([self.u_min])
        self.ocp.constraints.ubu = np.array([self.u_max])
        self.ocp.constraints.idxbu = np.array([0])

        # 3. Ocp solver
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.tf = t_horizon

        self.acados_ocp_solver = AcadosOcpSolver(self.ocp)

    def ocp_solve(self, state, ref, u_prev=None):

        if u_prev is None:
            u_prev = self.u_min

        u_prev_ = u_prev*np.ones((self.nu,))

        y_ref = np.concatenate((ref, u_prev_))

        y_ref_N = ref

        self.acados_ocp_solver.set(0, 'lbx', state)
        self.acados_ocp_solver.set(0, 'ubx', state)

        for stage in range(self.ocp.dims.N):
            self.acados_ocp_solver.set(stage, 'y_ref', y_ref)

        self.acados_ocp_solver.set(self.ocp.dims.N, 'y_ref', y_ref_N)

        status = self.acados_ocp_solver.solve()
        u_opt = self.acados_ocp_solver.get(0,'u')

        return u_opt, status