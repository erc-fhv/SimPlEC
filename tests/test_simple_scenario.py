"""Tests for a simple scenario with two connected models, one with time-shifted input."""

import pandas as pd
import numpy as np
from simplec import Simulation

class ExampleModel():
    """Example model for simple scenario test"""

    inputs = ['value_in']
    outputs = ['value_out']
    delta_t = 60 # s

    def __init__(self, name):
        self.name = name
        # init parameters here

    def step(self, time, value_in) -> dict:
        # model logic here
        value_out = value_in + 1
        return {'value_out': value_out}

def test_simple_scenario():
    """Test a simple scenario with two connected models, one with time-shifted input."""

    sim = Simulation()

    model1   = ExampleModel('example_1')
    model2   = ExampleModel('example_2')

    sim.add_model(model1, watch_values=['value_in'])
    sim.add_model(model2, watch_values=['value_out'])

    sim.connect(model1, model2, ('value_out', 'value_in'))
    sim.connect(model2, model1, ('value_out', 'value_in'), time_shifted=True,
        init_values={'value_out': 1})

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-01 00:03:00', freq='1min', tz='UTC+01:00')

    sim.simulate(times)

    expected_output = np.array([[1, 3],
                                [3, 5],
                                [5, 7],
                                [7, 9]])
    assert np.allclose(sim.df.values, expected_output), 'output of simple example seems to be off'
    assert sim.df.index[0] == pd.to_datetime('2021-01-01 00:00:00+01:00'), 'index seems to be off'
    assert sim.df.columns[0] == ('example_1', 'inputs', 'value_in'), \
        'columns of dateframe seem to be off'


def test_watch_two_inputs_same_model():
    class SourceModel():
        inputs = []
        outputs = ['out']
        delta_t = 60

        def __init__(self, name, value):
            self.name = name
            self.value = value

        def step(self, time) -> dict:
            return {'out': self.value}

    class SinkModel():
        inputs = ['in1', 'in2']
        outputs = ['sum_out']
        delta_t = 60

        def __init__(self, name):
            self.name = name

        def step(self, time, in1, in2) -> dict:
            return {'sum_out': in1 + in2}

    sim = Simulation()
    src1 = SourceModel('src1', 2)
    src2 = SourceModel('src2', 3)
    sink = SinkModel('sink')

    sim.add_model(src1)
    sim.add_model(src2)
    sim.add_model(sink, watch_values=['in1', 'in2'])

    sim.connect(src1, sink, ('out', 'in1'))
    sim.connect(src2, sink, ('out', 'in2'))

    times = pd.date_range('2021-01-01 00:00:00', '2021-01-01 00:01:00', freq='1min', tz='UTC+01:00')
    sim.simulate(times, enable_progress_bar=False)

    assert ('sink', 'inputs', 'in1') in sim.df.columns
    assert ('sink', 'inputs', 'in2') in sim.df.columns
