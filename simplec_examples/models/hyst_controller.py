class HystController():
    def __init__(self, name, delta_t=60, hyst=4, T_set=21, state_0='off') -> None:
        # Parameter
        self.delta_t = delta_t

        self.T_set = T_set 
        self.hyst = hyst

        self.state = state_0

        self.inputs  = ['T_is']
        self.outputs = ['state']

        self.name = name

    def step(self, time, T_is):
        if T_is > self.T_set + self.hyst/2: # upper limit; switch ooff
            self.state = 'off'
        elif T_is < self.T_set - self.hyst/2:  # lower limit, switch on
            self.state = 'on'
        else:
            pass # else leave state as is
        return {'state': self.state}
        
