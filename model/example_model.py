from acados_template import AcadosModel
import casadi as cs

class ExampleModel():

    def __init__(self, DynParam, RotorParam):
        '''

        :param DynParam: 'm' - mass
        :param RotorParam: 'C_T' - Thrust coefficient (N/rpm^2)
                           'p[0]' - Friction
                           'p[1]' - Drag
                           'p[2]' - Stiffness
        '''

        self.m = DynParam['m']
        self.C_T = RotorParam['C_T']
        self.p1 = RotorParam['p'][0]    # Friction
        self.p2 = RotorParam['p'][1]    # Drag
        self.p3 = RotorParam['p'][2]    # Stiffness

        self.model_name = 'example_model'
        self.model = AcadosModel()

        # State declaration
        self.z = cs.MX.sym('z',1)
        self.vz = cs.MX.sym('vz',1)
        self.w_rot = cs.MX.sym('w_rot',1)
        self.alpha_rot = cs.MX.sym('alpha_rot',1)
        self.x = cs.vertcat(self.z,self.vz,
                            self.w_rot,self.alpha_rot)

        # Command Declaration
        self.u = cs.MX.sym('u',1)

        # State Dot Declaration
        self.dz = cs.MX.sym('dzdt',1)
        self.dvz = cs.MX.sym('dvzdt',1)
        self.dw_rot = cs.MX.sym('dw_rot',1)
        self.dalpha_rot = cs.MX.sym('dalpha_rot',1)
        self.xdot = cs.vertcat(self.dz,self.dvz,
                               self.dw_rot,self.dalpha_rot)

    def export_acados_model(self)->AcadosModel:

        self.f_expl = cs.vertcat(self._z_dynamics(), self._vz_dynamics(),
                                    self._w_rot_dynamics(), self._alpha_rot_dynamics())
        self.f_impl = self.xdot - self.f_expl

        self.model.f_expl_expr = self.f_expl
        self.model.f_impl_expr = self.f_impl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.name = self.model_name
        return self.model

    def _z_dynamics(self):
        return self.vz

    def _vz_dynamics(self):

        g = -9.81
        dvzdt = 6.0*self.C_T*(self.w_rot**2)/self.m + g
        return dvzdt

    def _w_rot_dynamics(self):
        return self.alpha_rot

    def _alpha_rot_dynamics(self):
        dalphadt = (-(self.p1 + self.p2*self.w_rot)*self.alpha_rot
                    + self.p3*(self.u - self.w_rot))
        return dalphadt