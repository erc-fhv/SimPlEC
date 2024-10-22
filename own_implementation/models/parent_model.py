import numpy as np

class ParentModel():
    def __init__(self):
        self.name = 'Default'


    def step(self, time, prev_states, inputs) -> np.array, np.array:
        pass