import pandas as pd
from src.simplec import Simulation

from simplec_examples.models.building_model import BuildingModel
from simplec_examples.models.heat_pump import HeatPump, HeatPumpEventBased
from simplec_examples.models.hyst_controller import HystController
from simplec_examples.models.weather_model import LocalWeather

def test_compare_time_disc_vs_event_disc():
    # event based simulation 
    sim_event = Simulation()

    building   = BuildingModel('building')
    heatpump   = HeatPumpEventBased('heatpump')
    controller = HystController('controller', hyst=1.2)
    weather    = LocalWeather('weather')

    sim_event.add_model(building,   watch_values=['T_room1', 'T_room2', 'T_amb', 'apparent_zenith', 'dhi', 'ghi']  )
    sim_event.add_model(heatpump,   watch_values=['P_el', 'dot_Q_hp'], watch_heavy=['state'])
    sim_event.add_model(controller, watch_values=['T_is'], watch_heavy=['state'])
    sim_event.add_model(weather,    watch_values=['apparent_zenith', 'dhi', 'ghi'])

    sim_event.connect(weather, building, ('T_air', 'T_amb'), 'apparent_zenith', 'azimuth', 'dni', 'dhi', 'ghi')
    sim_event.connect(weather, heatpump, ('T_air', 'T_source'))
    sim_event.connect(heatpump, building, ('dot_Q_hp', 'dot_Q_heat'))
    sim_event.connect(controller, heatpump, ('state', 'state'), triggers=['state'])

    sim_event.connect(building, heatpump, ('T_room1', 'T_sink'), time_shifted=True, init_values={'T_room1': 21})
    sim_event.connect(building, controller, ('T_room1', 'T_is'), time_shifted=True, init_values={'T_room1': 21})

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')

    sim_event.run(times)

    df_event = sim_event.df

    df_event =df_event.ffill()

    # time discrete simulation
    sim_discrete = Simulation()

    building   = BuildingModel('building')
    heatpump   = HeatPump('heatpump')
    controller = HystController('controller', hyst=1.2)
    weather    = LocalWeather('weather')

    sim_discrete.add_model(building,   watch_values=['T_room1', 'T_room2', 'T_amb', 'apparent_zenith', 'dhi', 'ghi']  )
    sim_discrete.add_model(heatpump,   watch_values=['P_el', 'dot_Q_hp'], watch_heavy=['state'])
    sim_discrete.add_model(controller, watch_values=['T_is'], watch_heavy=['state'])
    sim_discrete.add_model(weather,    watch_values=['apparent_zenith', 'dhi', 'ghi'])

    sim_discrete.connect(weather, building, ('T_air', 'T_amb'), 'apparent_zenith', 'azimuth', 'dni', 'dhi', 'ghi')
    sim_discrete.connect(weather, heatpump, ('T_air', 'T_source'))
    sim_discrete.connect(heatpump, building, ('dot_Q_hp', 'dot_Q_heat'))
    sim_discrete.connect(controller, heatpump, ('state', 'state'))

    sim_discrete.connect(building, heatpump, ('T_room1', 'T_sink'), time_shifted=True, init_values={'T_room1': 21})
    sim_discrete.connect(building, controller, ('T_room1', 'T_is'), time_shifted=True, init_values={'T_room1': 21})

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')

    sim_discrete.run(times)

    df_discrete = sim_discrete.df
    
    df_discrete = df_discrete.ffill()


    assert df_event.compare(df_discrete).empty, 'Time discrete simulation differs from event discrete!'