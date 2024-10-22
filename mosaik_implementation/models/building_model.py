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
    def __init__(self, init_T_room1=20, init_T_room2=21, Δ_T=60, UA1=50, UA2=40, UA1_2=60, C1=50000000, C2=30000000, A_facade=2) -> None:
        # Parameters
        self.Δ_T = Δ_T

        self.UA1   = UA1   
        self.UA2   = UA2   
        self.UA1_2 = UA1_2
        self.C1    = C1   
        self.C2    = C2   
        self.A_facade = A_facade

        # Attributes 
        self.T_room1 = init_T_room1
        self.T_room2 = init_T_room2

        self.T_amb = 0
        self.dot_Q_sol = 0
        self.dot_Q_heat = 0

        # Calculation of A and B matrices and discretization of model

        # x    =           [        T_room1,         T_room2].T
        self.A = np.array([[(-UA1_2-UA1)/C1,        UA1_2/C1],
                           [     (UA1_2)/C2, (-UA1_2-UA2)/C2]])
        
        # u =               [T_amb, dot_Q_sol, dot_Q_heat]
        self.B = np.array([[UA1/C1,      1/C1,      1/C1],
                           [UA2/C2,         0,         0]])

        self.discretize()


    def discretize(self):
        # Discretize System
        # https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models

        # exponent =  [[A, B],
        #              [0, 0]] * Δ_T
        exponent = np.vstack((np.hstack((self.A, self.B)), np.zeros((self.B.shape[1], self.A.shape[1]+self.B.shape[1]))))*self.Δ_T

        res = expm(exponent)

        # res = [[Ad, Bd]
        #        [ 0, I]]
        self.Ad = res[:self.A.shape[0], :self.A.shape[1]]
        self.Bd = res[:self.B.shape[0], self.A.shape[1]:]

    
    def step(self):
        self.dot_Q_sol = self.calc_irradiation()*self.A_facade

        u = np.array([[self.T_amb], [self.dot_Q_sol], [self.dot_Q_heat]])
        x_prev = np.array([[self.T_room1], [self.T_room2]])
        x = self.Ad@x_prev + self.Bd@u
        self.T_room1, self.T_room2 = x.ravel()


    def calc_irradiation(self):
        return pvlib.irradiance.get_total_irradiance(
        surface_tilt=90,
        surface_azimuth=180,
        solar_zenith=self.solar_pos['apparent_zenith'],
        solar_azimuth=self.solar_pos['azimuth'],
        dni=self.dni,
        dhi=self.dhi,
        ghi=self.ghi
        # dni_extra=self.solar_pos['E'],
        # altitude=meta['location']['elevation']
        )['poa_global']


            
