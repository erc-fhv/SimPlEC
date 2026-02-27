"""Tests for topology export and public helper methods."""

import numpy as np
import pandas as pd
import pytest

import simplec


class Producer:
    """A producer model for topology tests."""

    inputs = []
    outputs = ['out']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time):  # noqa: ARG002
        return {'out': 1}


class Consumer:
    """A consumer model for topology tests."""

    inputs = ['in_data', 'in_const']
    outputs = ['out2']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, in_data, in_const):  # noqa: ARG002
        return {'out2': in_data + in_const}


@simplec.capture_params
class ParamModel:
    """A model with captured init params."""

    inputs = []
    outputs = ['x']
    delta_t = None

    def __init__(self, name, alpha=1.0, profile=None):
        self.name = name
        self.alpha = alpha
        self.profile = profile

    def step(self, time):  # noqa: ARG002
        return {'x': self.alpha}


class PlainModel:
    inputs = []
    outputs = ['x']
    delta_t = None

    def __init__(self, name):
        self.name = name

    def step(self, time):  # noqa: ARG002
        return {'x': 0}


def test_get_model_and_execution_order_public_helpers():
    sim = simplec.Simulation()
    a = Producer('a')
    b = Consumer('b')

    sim.add_model(a)
    sim.add_model(b)
    sim.connect(a, b, ('out', 'in_data'))
    sim.connect_constant(2, b, 'in_const')

    assert sim.get_model('a') is a
    assert sim.get_model('b') is b

    with pytest.raises(simplec.SimulationError, match="not found"):
        sim.get_model('missing')

    execution_order = sim.get_execution_order()
    assert execution_order == [a, b]


def test_get_topology_contains_all_connection_types():
    sim = simplec.Simulation()
    a = Producer('a')
    b = Consumer('b')

    sim.add_model(a)
    sim.add_model(b)
    sim.connect(a, b, ('out', 'in_data'))
    sim.connect_constant(5, b, 'in_const')
    sim.connect_nothing(a, b)

    topology = sim.get_topology()

    model_names = {m['name'] for m in topology['models']}
    assert model_names == {'a', 'b'}

    connection_types = {c['connection_type'] for c in topology['connections']}
    assert connection_types == {'data', 'constant', 'execution_order'}

    data_conn = [c for c in topology['connections'] if c['connection_type'] == 'data'][0]
    assert data_conn['source'] == 'a'
    assert data_conn['target'] == 'b'
    assert data_conn['attr_out'] == 'out'
    assert data_conn['attr_in'] == 'in_data'
    assert data_conn['constant_value'] is None

    const_conn = [c for c in topology['connections'] if c['connection_type'] == 'constant'][0]
    assert const_conn['target'] == 'b'
    assert const_conn['attr_in'] == 'in_const'
    assert const_conn['constant_value'] == 5

    exec_conn = [c for c in topology['connections'] if c['connection_type'] == 'execution_order'][0]
    assert exec_conn['source'] == 'a'
    assert exec_conn['target'] == 'b'


def test_get_topology_params_and_serialization_behavior():
    series = pd.Series([1, 2, 3])

    decorated = ParamModel('decorated', alpha=2.5, profile=series)
    plain = PlainModel('plain')

    sim = simplec.Simulation()
    sim.add_model(decorated)
    sim.add_model(plain)

    topology = sim.get_topology()
    models = {m['name']: m for m in topology['models']}

    assert models['decorated']['params']['alpha'] == 2.5
    assert models['decorated']['params']['profile'] == '<pd.Series len=3>'
    assert models['plain']['params'] is None


def test_get_topology_filters_default_object_docstring():
    class NoDocModel:
        inputs = []
        outputs = ['out']
        delta_t = 60

        def __init__(self, name):
            self.name = name

        def step(self, time):  # noqa: ARG002
            return {'out': 0}

    sim = simplec.Simulation()
    model = NoDocModel('nodoc')
    sim.add_model(model)

    topology = sim.get_topology()
    nodoc = [m for m in topology['models'] if m['name'] == 'nodoc'][0]
    assert nodoc['docstring'] == ''


def test_capture_params_decorator_captures_defaults_and_values():
    profile = np.array([1, 2])
    model = ParamModel('m', profile=profile)

    assert hasattr(model, '_init_params')
    assert model._init_params['name'] == 'm'
    assert model._init_params['alpha'] == 1.0
    assert model._init_params['profile'] is profile
