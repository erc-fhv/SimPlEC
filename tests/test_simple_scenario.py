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

    sim.run(times)

    expected_output = np.array([[1, 3],
                                [3, 5],
                                [5, 7],
                                [7, 9]])
    assert np.allclose(sim.df.values, expected_output), 'output of simple example seems to be off'
    assert sim.df.index[0] == pd.to_datetime('2021-01-01 00:00:00+01:00'), 'index seems to be off'
    assert sim.df.columns[0] == ('example_1', 'inputs', 'value_in'), \
        'columns of dateframe seem to be off'
