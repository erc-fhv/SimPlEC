import numpy as np

class HystController():
    def __init__(self, name, delta_t=60, hyst=4, T_set=21) -> None:
        # Parameter
        self.delta_t = delta_t

        self.T_set = T_set 
        self.hyst = hyst

        self.inputs  = ['T_is']
        self.states  = ['on']
        self.outputs = ['on']

        self.name = name

    
    def init(self):
        return np.array([0])
           

    def step(self, time, prev_states, inputs):
        T_is = inputs[0]
        if T_is - self.T_set > self.hyst/2:
            on = 0
        elif T_is - self.T_set < self.hyst/2:
            on = 1
        else:
            on = prev_states[0]
        
        # on = 0
        #     states, outputs
        return np.array([on]), np.array([on])
        
