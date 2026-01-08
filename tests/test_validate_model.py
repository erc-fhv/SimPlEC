import pytest
from simplec import Simulation

def test_missing_name_raises():
	class Model:
		inputs = ['in']
		outputs = ['out']

		def step(self, time, in_):
			return {'out': None}

	sim = Simulation()
	with pytest.raises(AttributeError, match="has no attribute name"):
		sim.validate_model(Model())


def test_missing_inputs_attr_raises():
	class Model:
		name = 'm'
		outputs = ['out']

		def step(self, time):
			return {'out': None}

	sim = Simulation()
	with pytest.raises(AttributeError, match="no atribute 'inputs'"):
		sim.validate_model(Model())


def test_inputs_not_list_raises():
	class Model:
		name = 'm'
		inputs = 'notalist'
		outputs = ['out']

		def step(self, time):
			return {'out': None}

	sim = Simulation()
	with pytest.raises(AttributeError, match="'inputs' need to be a list"):
		sim.validate_model(Model())


def test_missing_outputs_attr_raises():
	class Model:
		name = 'm'
		inputs = []

		def step(self, time):
			return {}

	sim = Simulation()
	with pytest.raises(AttributeError, match="no atribute 'outputs'"):
		sim.validate_model(Model())


def test_outputs_not_list_raises():
	class Model:
		name = 'm'
		inputs = []
		outputs = 'notalist'

		def step(self, time):
			return {}

	sim = Simulation()
	with pytest.raises(AttributeError, match="'outputs' need to be a list"):
		sim.validate_model(Model())


def test_missing_step_raises():
	class Model:
		name = 'm'
		inputs = []
		outputs = []

	sim = Simulation()
	with pytest.raises(AttributeError, match="has no 'step' function"):
		sim.validate_model(Model())


def test_step_not_callable_raises():
	class Model:
		name = 'm'
		inputs = []
		outputs = []
		step = 'notcallable'

	sim = Simulation()
	with pytest.raises(AttributeError, match="'step' not a callable"):
		sim.validate_model(Model())


def test_duplicate_input_output_raises():
	class Model:
		name = 'dup'
		inputs = ['a']
		outputs = ['a']

		def step(self, time, a):
			return {'a': a}

	sim = Simulation()
	with pytest.raises(AttributeError, match="contains the folowing duplicate value"):
		sim.validate_model(Model())


def test_step_first_arg_not_time_raises():
	class Model:
		name = 'm'
		inputs = ['x']
		outputs = ['y']

		def step(self, t, x):
			return {'y': x}

	sim = Simulation()
	with pytest.raises(AttributeError, match="First argument of step function needs to be 'time'"):
		sim.validate_model(Model())


def test_missing_input_in_step_args_raises():
	class Model:
		name = 'm'
		inputs = ['x']
		outputs = ['y']

		def step(self, time):
			return {'y': None}

	sim = Simulation()
	with pytest.raises(AttributeError, match="Not all inputs of"):
		sim.validate_model(Model())