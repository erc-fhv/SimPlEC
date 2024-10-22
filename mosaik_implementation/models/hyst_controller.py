class HystController():
    def __init__(self, hyst=4, T_set=21, T_is_0=21) -> None:
        self.T_set = T_set 
        self.hyst = hyst
        self.T_is = T_is_0
        self.on = 0
        

    def step(self):
        if self.T_is - self.T_set > self.hyst/2:
            self.on = 0
        elif self.T_is - self.T_set < self.hyst/2:
            self.on = 1

        # self.on = 0