# demo_1.py
import mosaik
import mosaik.util


# Sim config. and other parameters
SIM_CONFIG = {
    'BuildingSim': {
        'python': 'simulator.building_simulator:BuildingSim',
    },
    'Collector': {
        'python': 'simulator.collector:Collector',
        # 'cmd': '%(python)s collector.py %(addr)s',
    },
    'Clock': {
        'python': 'simulator.clock:Clock',
    },
    'Weather': {
        'python': 'simulator.weather_simulator:WeatherSim',
    },
    'HeatPump': {
        'python': 'simulator.heatpump_simulator:HeatPumpSim',
    },
    'HystController': {
        'python': 'simulator.hystcontroller_simulator:HystController',
    },
}
END = 60*60*24*365

# Create World
world = mosaik.World(SIM_CONFIG, time_resolution=60, debug=True)

# Start simulators
building_sim = world.start('BuildingSim', eid_prefix='Model_')
collector = world.start('Collector')
clock_creator = world.start('Clock')
weather_creator = world.start('Weather')
heatpump_creator = world.start('HeatPump')
hystctr_creator = world.start('HystController')

# Instantiate models
monitor = collector.Monitor()
clock = clock_creator.Clock(startdatetime='2021-01-01 00:00:00+01:00')

num = 20

l_weather = weather_creator.LocalWeather.create(num=num, latitude  = 47.4868936, longitude = 9.7406211)
l_building = building_sim.BuildingModel.create(num=num)
l_heatpump = heatpump_creator.HeatPumpModel.create(num=num)
l_hystctr = hystctr_creator.HystController.create(num=num)

# Connect entities
for weather, building, heatpump, hystctr in zip(l_weather, l_building, l_heatpump, l_hystctr): 
    world.connect(clock, weather, 'datetime')
    world.connect(weather, building, ('T_air', 'T_amb'), ('solar_pos', 'solar_pos'), 'dni', 'dhi', 'ghi')
    world.connect(weather, heatpump, ('T_air', 'T_source'))

    world.connect(heatpump, building, ('dot_Q_hp', 'dot_Q_heat'))
    world.connect(building, heatpump, ('T_room1', 'T_sink'), time_shifted=True, initial_data={'T_room1': 21})
    world.connect(building, hystctr, ('T_room1', 'T_is'), time_shifted=True, initial_data={'T_room1': 21})
    world.connect(hystctr, heatpump, ('on', 'on'))


world.connect(clock, monitor, 'datetime')

mosaik.util.connect_many_to_one(world, l_building, monitor, 'T_room1', 'T_room2', 'T_amb', 'dot_Q_sol', 'dot_Q_heat')
mosaik.util.connect_many_to_one(world, l_weather, monitor, 'T_air')
mosaik.util.connect_many_to_one(world, l_heatpump, monitor, 'cop')

# Run simulation
world.run(until=END)