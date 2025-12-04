import numpy as np
from scipy.linalg import expm
import pvlib


class BuildingModel():
    '''
    This is a model of a building envelope for a simple building, represented as RC model.

      /\
     /  \
    /____\
    | C1 | UA1
    |____| UA1_2
    | C2 | UA2
    |____|
    '''
    def __init__(self, name, delta_t=60, UA1=50, UA2=40, UA1_2=400, C1=50000000, C2=30000000,
            A_facade=2, T_o_0=20, T_u_0=21) -> None:
        # Parameters
        self.delta_t = delta_t

        self.UA1   = UA1
        self.UA2   = UA2
        self.UA1_2 = UA1_2
        self.C1    = C1
        self.C2    = C2
        self.A_facade = A_facade

        # Internal Parameters: (A and B matrices and discretization)

        # x    =           [        T_room1,         T_room2].T
        self._A = np.array([[(-UA1_2-UA1)/C1,        UA1_2/C1],
                           [     (UA1_2)/C2, (-UA1_2-UA2)/C2]])

        # u =               [T_amb, dot_Q_heat, dot_Q_sol]
        self._B = np.array([[UA1/C1,      1/C1,      1/C1],
                           [UA2/C2,         0,         0]])

        self._discretize()

        self.prev_states = np.array([T_o_0, T_u_0])

        #             [    0  ,       1     ,             2    ,     3    ,   4  ,   5  ,   6  ]
        self.inputs = ['T_amb', 'dot_Q_heat', 'apparent_zenith', 'azimuth', 'dni', 'dhi', 'ghi']
        self.outputs = ['T_room1', 'T_room2', 'dot_Q_sol']

        self.name = name

    def _discretize(self):
        '''Discretize System

        https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models

        exponent =  [[A, B],
                     [0, 0]] * delta_t'''
        exponent = np.vstack((np.hstack((self._A, self._B)), np.zeros((self._B.shape[1],
            self._A.shape[1]+self._B.shape[1]))))*self.delta_t

        res = expm(exponent)

        # res = [[Ad, Bd]
        #        [ 0, I]]
        self._Ad = res[:self._A.shape[0], :self._A.shape[1]]
        self._Bd = res[:self._B.shape[0], self._A.shape[1]:]

    def step(self, time, T_amb, dot_Q_heat, apparent_zenith, azimuth, dni, dhi, ghi):
        dot_Q_sol = pvlib.irradiance.get_total_irradiance(
        surface_tilt=90,
        surface_azimuth=180,
        solar_zenith  = apparent_zenith,
        solar_azimuth = azimuth,
        dni =           dni,
        dhi =           dhi,
        ghi =           ghi,
        )['poa_global']*self.A_facade

        u = np.array([T_amb, dot_Q_heat, dot_Q_sol])
        # Check if orientation of input and states is correct! should maybe be transposed???
        x = self._Ad@self.prev_states.T + self._Bd@u.T

        self.prev_states = x
        return {'T_room1':x[0], 'T_room2':x[1], 'dot_Q_sol':dot_Q_sol}
