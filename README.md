[[Hub4FlECs]]

# Introduction

SimPlEC is a Python framework to connect and simulate time discrete and event discrete models.

SimPlEC is currently meant to be used rather as a script than a finished package. Therefore the focus of the framework lays in simplicity. We encourage the user to adapt the framework to their own needs.
Many concepts regarding time and interfaces are based on the cosimulation framework MOSAIK.

The current status of the CI is:

[![Linting](https://github.com/erc-fhv/SimPlEC/actions/workflows/tests.yml/badge.svg?job=lint)](https://github.com/erc-fhv/SimPlEC/actions/workflows/tests.yml)
[![Unit Tests](https://github.com/erc-fhv/SimPlEC/actions/workflows/tests.yml/badge.svg?job=test)](https://github.com/erc-fhv/SimPlEC/actions/workflows/tests.yml)

# How to use the framework

## Installation

To reuse SimPlEC either install it with pip
```bash
pip install -e git+https://github.com/erc-fhv/SimPlEC
```
or clone it from this repository.
```bash
git clone https://github.com/erc-fhv/SimPlEC
```

## Getting Started
To conduct a simulation with SimPlEC, one needs a number of models and a scenario, where theese models will be connected.
The models are connected on a 'per-time-step' basis. This means that components which are closely coupled should be simulated as one componenent instead of connecting them via SimPlEC.
The models are simple python clases which will be instantiated in the scenario file and then added to the SimPlEC Simulation.
See the following examples:

The models need to follow the following structure:
```python
class ExampleModel():
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
```

The simulation can then be run following this simple example (utilizing the `ExampleModel` from above):
```python
from simplec import Simulation
import pandas

model1   = ExampleModel('example_1')
model2   = ExampleModel('example_2')

sim = Simulation()

sim.add_model(model1, watch_values=['value_in'])
sim.add_model(model2, watch_values=['value_out'])

sim.connect(model1, model2, ('value_out', 'value_in'))
sim.connect(model2, model1, ('value_out', 'value_in'),
			time_shifted=True, init_values={'value_out': 1})

times = pandas.date_range('2021-01-01 00:00:00',
						  '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')

sim.run(times)
```
Reccords of the simulation which are specified for each model with the `watch_value` attribute can be retrieved by accessing the pandas dataframe with `sim.df`.

## Requirements for the model objects:
Every model instance needs
- to have a unique `name` attribute (it makes sense, to pass this to `__init__` if you plan to create more instances of the model class)
- to implement the `inputs` list, specifying the input attributes of the step function as strings
- to implement the `outputs` list, specifying the output attributes of the step function as strings
- to implement the `delta_t` attribute (in seconds) if its a time discrete simulation model
- to implement a step function:
   - The step functions first positional argument needs to be `time`. `time` ist the current simulation timestamp of type `pandas.DateTime`.
   - Further,  all arguments specified in `inputs` need to be accepted as keyword arguments
   - The step function needs to return a dictionary with the outputs as specified in `self.outputs` as keys.
   - If the step function returns an argument `next_exec_time`,  `delta_t` is overruled and the model becomes event discrete.
## Adding models to the simulation
Models can be added to the Simulation by calling `sim.add_model(model, watch_values=['value_in'])`
Parameters are:
- `model` : The model object to add to the simulation following the above requirements
- `watch_values` : any inputs or outputs (corresponding to the models input and output lists) that need to be recorded for the output
  This output will then be stored in a `pandas.Dataframe` which can be accessed after the simulation as finished with `sim.df`
  A Word of Warning: even though, arbitrarily objects can be passed in between the models, **only watch values that can safely be stored in a `pandas.DataFrame` (e.g. numeric values)** for other object see the next keyword.
- `watch_heavy` This works similar to `watch_values` but stores 'heavy objects' like lists or pandas DataFrames which are exchanged in between models, the reccords can be accesed after the simulation with `sim.data`.

## Connecting models
Models can be connected by utilizing the `sim.connect()` method.
outputs of one model can be connected to inputs of another model.
One output can be connected to several other models however one input can only be provided by one other model or connection (see hints below if you think you need to do so)

Parameters are:
`model1` : the model which outputs are to be connected (needs to be added to the simulation before it can be connected)
`model2` : the model which inputs are to be connected (needs to be added to the simulation before it can be connected)
`*connections`,  str or tuple
 if  string is provided: this is the attribute which can be found in `model1.output` and `model2.input`
 if tuple is provided: the tuple specifies attribute out of model1 and attribute in of model2 (attr_out, attr_in) specifying output (of model1) and input (of model2) attributes of the models that need to be connected.
 `time_shifted` : bool, to prevent computational loops (eg. in controll loops), time_shifted connections pass the output provided by model1  at the last time to model2.
`init_values` : dict, Time shifted connections need an initial value for the outputs of model1 (attr_out !) to function in the first step.
`triggers` : list, output attributes (attr_out) of model1, that trigger the execution of model2 when the value changes. This makes the simulation event based! **Make sure you know what you are doing, there are no internal checks for the model types, any model can be triggered at any time!** 'Big' items should probably be avoided as triggers. For performance reasons numerical values or strings are preferable.

Connecting several models to one input:
If you think you need to connect several inputs to one model, the model needs to be able to handle this internally. One option of SimPlEC to handle several inputs for one attribute is to specify an input attribute ending with an underscore '\_' such as 'P_el_'.Then, SimPlEC expects one or more inputs for this attribute and wraps them into a list, see the following example:
```python
class ExampleModel()
	def __init__(self, n_powers):
		self.input = ['P_el_']
		...
	def step(self, time, P_el_):
        # P_el_ is a list here
		P_tot = sum(P_el_)
        ...
```
The order of the list is not guaranteed to follow any logic and using this feature only really makes sense, if the order does not matter and the values get agregated (e.g. summed or averaged).
If the order matters, the models need to be adjusted, to explicitly accept theese inputs individually, as we want to avoid nested structures of inputs and outputs.
If you want to change the number of Inputs parametrically, you can adjust the models `__init__` so that the number of inputs can be changed dynamically on creation for example as such
```python
class ExampleModel()
	def __init__(self, n_powers):
		self.input = [f'P_el_{n}' for n in range n_powers]
		...
	# The signatue of step() in this case would then be somethig like
	def step(self, time, **P_el):
		...
```
However we discurage the use of this dynamic creation and suggest to use models as 'scripts' and adujst the code to ones specific need.

## Running the simulation
The simulation can be run with the `sim.run()` method.
Parameters are:
`datetimes` : a `pandas.DatetimeIndex`, the simulation is run over this index. It can for example be created by calling `pandas.date_range(...)` representing time as a human readable format might be limiting the performance, however when writing realistic simulation scenarios, time awareness can probably reduce error sources significantly.

# How it works
When modelling energy systems in python, one might start by implementing models as simple functions or plain procedural code, stepping through time in a four loop.
This however becomes quite messy when the models get more and more complex, therefore one might start implementing the models as classes, so that the parameters and states of the models are contained and protected as such:
```python
class ExampleModel():
    delta_t = 1
   def __init__(self):
		pass

   def step(self, time, value_in):
        value_out = value_in + 1
        return value_out
```
The model would then already look like the model classes used within this framework.
One could easily write a four loop to step those models through time as such:
```python
model1   = ExampleModel()
model2   = ExampleModel()
m2_v = 3
results = []
for i in range(20):
	m1_v = model1.step(i, m2_v)
	m2_v = model2.step(i, m1_v)
	results.append(m2_v)
```
Note, that in this example:
- model1 provides the input of model2 and
- model2 provides the input of model1 for the next time step.
Therefore we need to initialize the output of model2 (m2_v) before the simulation is started.

When there are more models, however, the writing of the four loop becomes quite difficult and not very flexible when the scenario (the interconnections between models) change.
That is where the framework comes into play.
## Execution order and connections
One major question that pops up when implementing the before shown for loop is the order of the models. As can be seen, model1 needs to be stepped before model2 as it provides the output of the latter. The connection between model2 and model1 is then only taking place in the next time step (so time shifted).
However, when the scenario becomes more complex, the question on the order becomes more difficult.
It turned out, that graphs are a good representation of the connection between the models and that there are frameworks to represent graphs in python. We used networkx and add the models as nodes and the connections as directed edges.
Networkx then provides the `topological_sort()` function, which allows to find an order of the models, where all models can be provided with the input from the model before.
An additional trick, we uses is, to add the time shifted connections in different direction to the graph. This ensures that model2, providing an input for model1 in the next time step is stepped after model1 in the execution order.

Now that we know the execution order of our modesl (which is trivial for our example case), the next question is, how to map the outputs to the inputs of another model.
We decided, to store all outputs in a dictionary, with a key, that combines the model name and the attribute name of the output as such:
```python
model1   = ExampleModel()
model2   = ExampleModel()

outputs = {'model1':{'value_out': np.nan}, 'model2':{'value_out': 3}}
results = []
for i in range(20):
	outputs['model1']['value_out'] = model1.step(i, outputs['model2']['value_out'])
	outputs['model2']['value_out'] = model2.step(i, outputs['model1']['value_out'])
	results.append(outputs['model2']['value_out'])
```
```python
model1   = ExampleModel()
model2   = ExampleModel()

outputs = {'model1':{'value_out': np.nan}, 'model2':{'value_out': 3}}
results = []
input_map = {'model1': {'value_in': ('model2', 'value_out')},
              'model2': {'value_in': ('model1', 'value_out')}}

model_execution_list = [model1, model2]
for i in range(20):
    for model in model_execution_list:
        inputs = outputs[input_map[model]['value_in'][0]][input_map[model]['value_in'][1]]
        outputs[model].update(model.step(i, inputs))
```

```python
model1   = ExampleModel()
model2   = ExampleModel()

outputs = {'model1':{'value_out': np.nan}, 'model2':{'value_out': 3}}
model_next_exec_time['model1'] = -1
model_next_exec_time['model2'] = -1
results = []
model_execution_list = [model1, model2]
for i in range(20):
    for model in model_execution_list:
        if model_next_exec_time[model] <= i:
            model_next_exec_time[model] += model.deta_t
    	    outputs[model].update(model.step(i, outputs['model2']['value_out']))
```
A model then only needs to know, which of the outputs belong to its inputs. Therefore a dictionary is attached to each model object that provides this mapping. The dictionary of model2 could then look like this: `model2_input_map = {'value_in': 'model2_value_out'}`. By iterating over this dict, we can collect all the inputs for model2 from the ouptut dict (here we only have one input but it could of course be many). The iteration as a dict comprehension looks like this:
`model2_inputs = {input_key: outputs[output_key] for input_key, output_key in model2_input_map.items()}`
This dict can then be passed to model2 as such: `model2.step(time, **model2_inputs)`.
The input map of each model is computed in the `sim.connect()` method and assigned to each model directly (no proxies, no more nested structures than needed) as such `model._sim_input_map`. Naming convention: All attributes added to the model by the simulation start with `_sim_`.
## Different time resolutions of models
As models can have different time resolutions, each iteration, the question arises, if a model should be stepped or not. This is determined by the `delta_t` attribute of the model or if the model is event based by the return value of `next_exec_time`.
Therefore, an attribute `model._sim_next_exec_time` is assigned to each model. If a model returns a `next_exec_time` this value is used, if not, this value is computed based on the `delta_t`  attribute of the model. With this, we can decide if a model needs to be stepped, if the current simulation time is >= a models `next_exec_time`.
This implies, that if a model provides the input of another model but the first one is not stepped, the previous output is  retrieved from the `output` dict as input of the latter model.
## Triggering attributes for event based models
For event based simulation, it might be necessary, to trigger a models execution, based on the output of another model. An example could be a heat pump, which is in an off state but should start computing output values, as soon as the controller tells it to start.
To achieve this, each model is assigned a `model._sim_triggers_model` dict. This dict looks like this: `{'attr_out': [modeltotrigger1, modeltotrigger2]}`. In the simulation loop, this dict is iterated, and a comparison between the previous output and the current output of the model is made for each `attr_out` in the `_sim_triggers_model` dict. If a difference is detected, the models (`[modeltotrigger1, modeltotrigger2]`)  need to be executed as soon as possible but without breaking the execution direction. This is done by setting `modeltotrigger1._sim_next_exec_time` to the current simulation time. Therefore it gets executed in this time step if the connection is not time shifted or executed in the next time step if the connection is time shifted based on the execution direction.

# More advanced / other concepts
## Independent interfaces
When modelling controll methods or data infrastructure, one might quickly find time discrete simulation limiting. And even the event discrete simulation, as it is implemented in SimPlEC, might not fullfil all needs for simulation. Therefore one might start implementing custom methods for the models which interact with each other independent of the framework (one could call this agent based).
The following example ilustrates a simple example of an interaction between a GridOperator and a SmartMeter:
```python
import pandas as pd

class SmartMeter():
    def __init__(self, name, n_loads=1):
        self.name = name
        self.delta_t = 60  # s

        self.inputs = [f'P_{n}' for n in range(n_loads)]
        self.outputs = ['P_Grid', 'P_quarterhourly_data']

        self._P_quarter_hourly_data = pd.DataFrame([])

    def step(self, time, *P_behind_meter):
        P_grid = sum(P_behind_meter)
        self._P_quarter_hourly_data[time, 'P'] = P_grid

        return {'P_Grid': P_grid}

    def retrieve_data(self):
        return self._P_quarter_hourly_data # seach for / implement 'pop method' which empties dataframe but returns the values

class GridOperator():
    def __init__(self, *smart_meter_models):
        self.smart_meter_models = smart_meter_models

    def step(self, time, inputs):
        for smart_meter_model in self.smart_meter_models:
            self.smart_meter_data = smart_meter_model.retrieve_data()
```
To support this style of modelling, SimPlEC provides the `sim.connect_nothing()` method. This method connects no attributes but can be used to maintain the execution order of models, that interact independently of the framework.
This provides flexibility for modelling in an unintrusive way.
The given example models from above would then be connected as such, to ensure, the `SmartMeter` is stepped before the `GridOperator`:
```python
sim = Simulation()
sm = SmartMeter()
go = GridOperator(sm)

sim.add(sm)
sim.add(go)

sim.connect_nothing(sm, go)

times = pandas.date_range('2021-01-01 00:00:00',
						  '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')
sim.run(times)
```

# Why another framework?
- We found other frameworks to be complex for most of our use cases.
- Network sockets are overkill for most situations. Most interfaces can probably be accessed by python much easier many protocols such as Sunspec or simulation tools such as Ida Ice actually provide python libraries and APIs (no network sockets).
