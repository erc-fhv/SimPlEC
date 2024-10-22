import pandas as pd
from simulation import Simulation

from models.building_model import BuildingModel
from models.heat_pump import HeatPump
from models.hyst_controller import HystController
from models.weather_model import LocalWeather


sim = Simulation()

l = 5
buildings_list   = [BuildingModel(f'building_{i}') for i in range(l)]
heatpumps_list   = [HeatPump(f'heatpump_{i}') for i in range(l)]
controllers_list = [HystController(f'controller_{i}') for i in range(l)]
weathers_list    = [LocalWeather(f'weather_{i}') for i in range(l)]

for building, heatpump, controller, weather in zip(buildings_list, heatpumps_list, controllers_list, weathers_list):
    sim.add_model(building)
    sim.add_model(heatpump)
    sim.add_model(controller)
    sim.add_model(weather)

sim.init()

for building, heatpump, controller, weather in zip(buildings_list, heatpumps_list, controllers_list, weathers_list):
    sim.connect(weather, building, ('T_air', 'T_amb'), 'apparent_zenith', 'azimuth', 'dni', 'dhi', 'ghi')
    sim.connect(weather, heatpump, ('T_air', 'T_source'))
    sim.connect(heatpump, building, ('dot_Q_hp', 'dot_Q_heat'))
    sim.connect(controller, heatpump, ('on', 'on'))

    sim.connect(building, heatpump, ('T_room1', 'T_sink'), time_shifted=True, init_values={'T_room1': 21})
    sim.connect(building, controller, ('T_room1', 'T_is'), time_shifted=True, init_values={'T_room1': 21})

sim.prepare()

times = pd.date_range('2021-01-01 00:00:00', '2021-12-31 23:59:00', freq='1min', tz='UTC+01:00')
# times = pd.date_range('2021-01-01 00:00:00', '2021-01-02 00:00:00', freq='1min', tz='UTC+01:00')

sim.run(times)