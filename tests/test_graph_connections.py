"""Tests for graph connection storage and execution-order behavior."""

from simplec import Simulation


class SourceModel:
    inputs = []
    outputs = ['a', 'b']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time):
        return {'a': 1, 'b': 2}


class TargetModel:
    inputs = ['x', 'y']
    outputs = ['out']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, x, y):
        return {'out': x + y}


class LoopModelA:
    inputs = ['in_a']
    outputs = ['out_a']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, in_a):
        return {'out_a': in_a}


class LoopModelB:
    inputs = ['in_b']
    outputs = ['out_b']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, in_b):
        return {'out_b': in_b}


def test_multi_edges_between_same_models_are_stored():
    sim = Simulation()
    src = SourceModel('src')
    tgt = TargetModel('tgt')

    sim.add_model(src)
    sim.add_model(tgt)
    sim.connect(src, tgt, ('a', 'x'))
    sim.connect(src, tgt, ('b', 'y'))

    edge_data = list(sim.graph.get_edge_data(src, tgt).values())

    assert len(edge_data) == 2
    assert {
        (edge['attr_out'], edge['attr_in'], edge['connection_type'])
        for edge in edge_data
    } == {
        ('a', 'x', 'data'),
        ('b', 'y', 'data'),
    }


def test_compute_execution_order_still_works_with_multidigraph():
    sim = Simulation()
    model_a = LoopModelA('a')
    model_b = LoopModelB('b')

    sim.add_model(model_a)
    sim.add_model(model_b)

    sim.connect(model_a, model_b, ('out_a', 'in_b'))
    sim.connect(model_b, model_a, ('out_b', 'in_a'),
                time_shifted=True, init_values={'out_b': 0})

    execution_order = sim._compute_execution_order_from_graph(sim.graph)

    assert execution_order == [model_a, model_b]


def test_draw_exec_graph_does_not_crash(monkeypatch):
    sim = Simulation()
    src = SourceModel('src')
    tgt = TargetModel('tgt')

    sim.add_model(src)
    sim.add_model(tgt)
    sim.connect(src, tgt, ('a', 'x'))

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    sim.draw_exec_graph()
