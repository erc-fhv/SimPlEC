"""Tests for graph connection storage and execution-order behavior."""

from simplec import Simulation, ConstantNode


class SourceModel:
    inputs = []
    outputs = ['a', 'b']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time):  # noqa: ARG002
        return {'a': 1, 'b': 2}


class TargetModel:
    inputs = ['x', 'y']
    outputs = ['out']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, x, y):  # noqa: ARG002
        return {'out': x + y}


class LoopModelA:
    inputs = ['in_a']
    outputs = ['out_a']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, in_a):  # noqa: ARG002
        return {'out_a': in_a}


class LoopModelB:
    inputs = ['in_b']
    outputs = ['out_b']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time, in_b):  # noqa: ARG002
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


def test_constant_source_represented_in_graph():
    sim = Simulation()
    tgt = TargetModel('tgt')

    sim.add_model(tgt)
    sim.connect_constant(10, tgt, 'x')
    sim.connect_constant(20, tgt, 'y')

    # Find constant sentinel nodes
    const_nodes = [n for n in sim.graph.nodes if isinstance(n, ConstantNode)]
    assert len(const_nodes) == 2

    node_names = {n.name for n in const_nodes}
    assert 'tgt__const__x' in node_names
    assert 'tgt__const__y' in node_names

    # Check edges exist
    const_edges = []
    for u, v, k, d in sim.graph.edges(data=True, keys=True):
        if isinstance(u, ConstantNode):
            const_edges.append((u.name, v.name, d['attr_in'], d['connection_type']))

    assert len(const_edges) == 2
    assert ('tgt__const__x', 'tgt', 'x', 'constant') in const_edges
    assert ('tgt__const__y', 'tgt', 'y', 'constant') in const_edges


def test_execution_order_filters_out_constant_nodes():
    sim = Simulation()
    a = LoopModelA('a')
    b = LoopModelB('b')

    sim.add_model(a)
    sim.add_model(b)
    
    sim.connect_constant(5, a, 'in_a')
    sim.connect(a, b, ('out_a', 'in_b'))

    execution_order = sim._compute_execution_order_from_graph(sim.graph)

    # Should only include model nodes, not constant sentinels
    assert len(execution_order) == 2
    assert all(not isinstance(n, ConstantNode) for n in execution_order)
    assert a in execution_order
    assert b in execution_order
