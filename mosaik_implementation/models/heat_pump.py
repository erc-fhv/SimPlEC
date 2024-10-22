class HeatPump():
    def __init__(self, T_source_init=5, T_sink_init=21, eta=0.5, P_el_nom=2000) -> None:
        self.eta      = eta
        self.P_el_nom   = P_el_nom # nominal Power

        self.cop_fun = lambda Tsource, Tsink, eta: (Tsink+273.15)/((Tsink+273.15) - (Tsource+273.15)) * eta
        self.cop     = 0

        self.T_source = T_source_init
        self.T_sink   = T_sink_init
        self.dot_Q_hp = 0
        self.on       = 1


    def step(self):
        self.P_el = self.P_el_nom*self.on
        self.cop = self.cop_fun(self.T_source, self.T_sink, self.eta)
        self.dot_Q_hp = self.P_el * self.cop

