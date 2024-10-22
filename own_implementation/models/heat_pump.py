import numpy as np

class HeatPump():
    def __init__(self, name, delta_t=60, eta=0.5, P_el_nom=2000) -> None:
        # Parameter
        self.delta_t = delta_t

        self.eta      = eta
        self.P_el_nom = P_el_nom # nominal Power

        # internal Parameter
        self._cop_fun = lambda Tsource, Tsink, eta: (Tsink+273.15)/((Tsink+273.15) - (Tsource+273.15)) * eta
        
        self.inputs  = ['on', 'T_source', 'T_sink']
        self.states  = []
        self.outputs = ['cop', 'P_el', 'dot_Q_hp']

        self.name = name

    
    def init(self):
        return np.array([])
    

    def step(self, time, prev_states, inputs):

        P_el = self.P_el_nom*inputs[0]
        cop = self._cop_fun(inputs[1], inputs[2], self.eta)
        dot_Q_hp = P_el * cop

        #     states, outputs
        return np.array([]), np.array([cop, P_el, dot_Q_hp])
        
