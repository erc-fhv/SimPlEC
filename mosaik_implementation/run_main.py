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
END = 60*60*24*2  # 365/5

# Create World
world = mosaik.World(SIM_CONFIG, time_resolution=60, debug=False)

# Start simulators
building_sim = world.start('BuildingSim', eid_prefix='Model_')
collector = world.start('Collector')
clock_creator = world.start('Clock')
weather_creator = world.start('Weather')
heatpump_creator = world.start('HeatPump')
hystctr_creator = world.start('HystController')

# Instantiate models
building = building_sim.BuildingModel()
monitor = collector.Monitor()
clock = clock_creator.Clock(startdatetime='2021-01-01 00:00:00+01:00')
weather = weather_creator.LocalWeather()
heatpump = heatpump_creator.HeatPumpModel()
hystctr = hystctr_creator.HystController()

# Connect entities
world.connect(clock, weather, 'datetime')

world.connect(weather, building, ('T_air', 'T_amb'), ('solar_pos', 'solar_pos'), 'dni', 'dhi', 'ghi')
world.connect(weather, heatpump, ('T_air', 'T_source'))
world.connect(heatpump, building, ('dot_Q_hp', 'dot_Q_heat'))
world.connect(building, heatpump, ('T_room1', 'T_sink'), time_shifted=True, initial_data={'T_room1': 21})
world.connect(building, hystctr, ('T_room1', 'T_is'), time_shifted=True, initial_data={'T_room1': 21})
world.connect(hystctr, heatpump, ('on', 'on'))


world.connect(clock, monitor, 'datetime')
world.connect(building, monitor, 'T_room1', 'T_room2', 'T_amb', 'dot_Q_sol', 'dot_Q_heat')
world.connect(weather, monitor, 'T_air')
world.connect(heatpump, monitor, 'cop')


# # Create more entities
# more_models = building_sim.BuildingModel.create(2, init_val=3)
# mosaik.util.connect_many_to_one(world, more_models, monitor, 'val', 'delta')

# Run simulation
world.run(until=END)