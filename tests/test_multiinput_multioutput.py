import numpy as np
import pandas as pd
import os
from simplec import Simulation

def test_multiinput_list():
    class ExampleModel():
        inputs = []
        outputs = ['value_out']
        delta_t = 60 # s

        def __init__(self, name):
            self.name = name

        def step(self, time) -> dict:
            value_out = 20
            return {'value_out': value_out}

    class ExampleModelMultiinputList():
        inputs = ['value_in_']
        outputs = ['value_out']
        delta_t = 60 # s

        def __init__(self, name):
            self.name = name

        def step(self, time, value_in_) -> dict:
            value_out = sum(value_in_)
            return {'value_out': value_out} # value out will be the sum of values in

    sim = Simulation()

    model1a   = ExampleModel('example_1a')
    model1b   = ExampleModel('example_1b')
    model2    = ExampleModelMultiinputList('example_2')

    sim.add_model(model1a)
    sim.add_model(model1b)
    sim.add_model(model2, watch_values=['value_out'])

    sim.connect(model1a, model2, ('value_out', 'value_in_'))
    sim.connect(model1b, model2, ('value_out', 'value_in_'))

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-01 00:01:00', freq='1min', tz='UTC+01:00')

    sim.simulate(times)
    timestamp_1 = pd.to_datetime('2021-01-01 00:00:00+0100')

    assert sim.df.loc[timestamp_1, ('example_2', 'outputs', 'value_out')] == 40, \
        'multiinput list not working correctly'


def test_multiinput_dict():
    class ExampleModel():
        inputs = []
        outputs = ['value_out']
        delta_t = 60 # s

        def __init__(self, name):
            self.name = name

        def step(self, time) -> dict:
            value_out = 20
            return {'value_out': value_out}

    class ExampleModelMultiinputDict():
        inputs = ['value_in_dict']
        outputs = ['value_out']
        delta_t = 60 # s

        def __init__(self, name):
            self.name = name

        def step(self, time, value_in_dict) -> dict:
            value_out = value_in_dict
            return {'value_out': value_out} # value out will be a dict

    sim = Simulation()

    model1a   = ExampleModel('example_1a')
    model1b   = ExampleModel('example_1b')
    model2    = ExampleModelMultiinputDict('example_2')

    sim.add_model(model1a)
    sim.add_model(model1b)
    sim.add_model(model2, watch_heavy=['value_out'])

    sim.connect(model1a, model2, ('value_out', 'value_in_dict'))
    sim.connect(model1b, model2, ('value_out', 'value_in_dict'))

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-01 00:01:00', freq='1min', tz='UTC+01:00')

    sim.simulate(times)
    timestamp_1 = pd.to_datetime('2021-01-01 00:00:00+0100')

    assert isinstance(sim.data, dict), 'sim.data is not a dictionary'
    assert isinstance(sim.data[('example_2', 'outputs', 'value_out')][timestamp_1], dict), \
        'multiinput dict not returning dict'
    assert 'example_1a' in sim.data[('example_2', 'outputs', 'value_out')][timestamp_1], \
        'multiinput structure is off'
    assert sim.data[('example_2', 'outputs', 'value_out')][timestamp_1]['example_1a'] == 20, \
        'multiinput output is unexpected'


def test_multioutput_dict():
    class ExampleModel():
        inputs = ['value_in']
        outputs = ['value_out']
        delta_t = 60 # s

        def __init__(self, name):
            self.name = name

        def step(self, time, value_in) -> dict:
            return {'value_out': value_in}

    class ExampleModelMultioutputDict():
        inputs = []
        outputs = ['value_out_dict']
        delta_t = 60 # s

        def __init__(self, name, models:list):
            self.models = models
            self.name = name

        def step(self, time) -> dict:
            value_out = {m: 20 for m in self.models}
            return {'value_out_dict': value_out} # value out will be a dict

    sim = Simulation()

    model1a   = ExampleModel('example_1a')
    model1b   = ExampleModel('example_1b')
    model2    = ExampleModelMultioutputDict('example_2', ['example_1a', 'example_1b'])

    sim.add_model(model1a, watch_values=['value_in'])
    sim.add_model(model1b, watch_values=['value_out'])
    sim.add_model(model2, watch_heavy=['value_out_dict'])

    sim.connect(model2, model1a, ('value_out_dict', 'value_in'))
    sim.connect(model2, model1b, ('value_out_dict', 'value_in'))

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-01 00:01:00', freq='1min', tz='UTC+01:00')

    sim.simulate(times)
    timestamp_1 = pd.to_datetime('2021-01-01 00:00:00+0100')

    assert isinstance(sim.data[('example_2', 'outputs', 'value_out_dict')][timestamp_1], dict), \
        'multioutput dict not returning dict'
    assert len(sim.data[('example_2', 'outputs', 'value_out_dict')][timestamp_1]) == 2, \
        'multioutput dict length is off'
    assert sim.df.loc[timestamp_1, ('example_1a', 'inputs', 'value_in')] == 20
    assert sim.df.loc[timestamp_1, ('example_1b', 'outputs', 'value_out')] == 20
