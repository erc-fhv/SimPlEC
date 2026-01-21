'''
Simple Simulation framework for the time-discrete simulation of energy models

The framework can be used to connect and simulate time discrete (and maybe event
discrete) models and handle the model outputs.

The models need to follow the following structure:
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

Requirements for the model objects:
Every model instance needs
 - to have a unique name
 - to implement the inputs list, specifying the inputs of the step function as
   strings
 - to implement the outputs list, specifying the outputs of the step function as
   strings
 - to implement the delta_t attribute (in seconds)
 - to implement a step function
    - the step function needs to accept an argument time and all arguments
      specified in \'inputs\'
    - the step function needs to return a dict with the outputs as keys

The simulation can then be run following this simple example (utilizing the
example model from above):

from simulation import Simulation
import pandas

sim = Simulation()

model1   = ExampleModel('example_1')
model2   = ExampleModel('example_2')

sim.add_model(model1, watch_values=['value_in'])
sim.add_model(model2, watch_values=['value_out'])

sim.connect(model1, model2, ('value_out', 'value_in'))
sim.connect(model2, model1, ('value_out', 'value_in'), time_shifted=True,
    init_values={'value_out': 1})

times = pandas.date_range('2021-01-01 00:00:00', '2021-01-03 00:00:00',
                          freq='1min', tz='UTC+01:00')

sim.simulate(times)

outputs of the simulation can then be retrieved by accessing sim.df
'''

from pathlib import Path
from copy import copy
import logging
import inspect
import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class SimulationError(Exception):
    '''Exception raised for errors in the Simulation class.'''


class Simulation():
    '''Simulation class to connect and simulate time discrete models'''

    def __init__(self,
                logger_name: str | None = None,
                time_resolution: str = 'sec',
                multiinput_symbol: str = '_',
                multiinput_dict_symbol: str = '_dict'
                ) -> None:
        '''
        Creates a simulation object

        Parameter
        ----------
        logger_name : str or None, log some information about the simulation
            progress if a name is provided (loging needs to be configured, see
            Python documentation standard library logging)
        time_resolution : str, Time resolution / unit of the models.delta_t,
            default: 'sec', (keyword strings according to pandas.Timedelta)
        multiinput_symbol : str, suffix for model input names (default: '_'),
            that accept multiple inputs. Input values ending with this character
            will be wrapt in a list.
        multiinput_dict_symbol : str, suffix for model input names (default: '_dict'),
            that accept multiple inputs. Input values ending with this character
            will be wrapt in a dict with the input model name as key.
        '''
        self.time_resolution = time_resolution
        self.multiinput_symbol = multiinput_symbol
        self.multiinput_dict_symbol = multiinput_dict_symbol

        # set up a logger
        self.log = logging.getLogger(logger_name)
        self.log.disabled = logger_name is None

        # this dict contains all models(names) and their output (if any)
        # structure: nested dict e.g. {'model1name':{'output1': np.nan},
        # 'model2': {}}. The models will collect their inputs from this output
        self._outputs = {}

        # create execution graph (the graph is needed to calculate the model
        # execution order based on the connections between the models)
        self.graph = nx.DiGraph()

        # setup the data logging capabilities
        self.model_watch_attributes = {}
        self.model_heavy_watch_attributes = {}
        # {model: {inputs: [(attr, index), (attr, index)]}}
        self.model_watch_attributes_index = {}

        # heavy watch values are stored here {attr: {datetime: value}}
        self.data = {}

        # # set up the internal values for the model connection handling

        # input_map answers the question for the model "where do i get my inputs
        # from?" It maps the models inputs to the simulation self._outputs dict
        # {'modelname': {'attribute_in': (model1name, attribute_out)}}
        # It gets filled in 'connect'.
        self._model_input_map = {}

        # datetime timedeltas of the models for fast next_exec_time calculation,
        # it contains the models timedelta converted to pd.Timedelta
        self._model_timedelta_t = {}

        # next_exec_time answers the question 'when will the model be executed
        # next time?' initialized with 1970 date to make sure all models are
        # stepped in the first timestep
        self._model_next_exec_time = {}

        # triggers_model answers the question 'which models are triggert by a
        # certain model output (attr_out)?'
        # structure: {model: {'attr_out': [modeltotrigger1, modeltotrigger2]}}
        # It gets filled in 'connect', model not in dict, if model triggers no
        # model, output not in inner dict if output triggers no models
        self._model_output_triggers_models = {}

        # Initialize missing member variables
        self.df: pd.DataFrame
        self._df_multiindex: pd.MultiIndex
        self.array: np.ndarray

        self.log.info('Initialized Simulation sucessfully')

    def add_model(self, model, watch_values=None, watch_heavy=None):
        '''
        Adds a model to the simulation.

        Parameters
        ----------
        model : User defined models need to follow the following example:
        """
        class ExampleModel():
            inputs = ['value_in']
            outputs = ['value_out']

            def __init__(self, name):
                self.name = name

            def step(self, time, value_in) -> dict:
                value_out = value_in + 1
                return {'value_out': value_out}
        """

        watch_values : adds a models input or output to the reccorded values
            (pandas.DataFrame). Access after simulation with: sim.df. Only watch
            numeric values that can safely be cast into a pandas DataFrame!
        watch_heavy : adds a models input or output to the reccorded data (dict).
            Access values after simulation with: sim.data. Anything can be
            reccorded but performance might be limited. Only suggested for
            debugging purposes!
        '''

        if watch_values is None:
            watch_values = []
        if watch_heavy is None:
            watch_heavy = []

        # validate model structure
        self.validate_model(model)

        # check for duplicate models
        if model.name in self._outputs:
            raise SimulationError(
                f'Model.name \'{model.name}\' not unique, which is required ' +
                'for the simulation to work')
        # add model outputs to the simulation output dict
        self._outputs.update(
            {model.name: {output_attr: np.nan
                          for output_attr in model.outputs}})
        # add model to the execution graph
        self.graph.add_node(model)

        # Add connection and event based attributes
        self._model_input_map[model.name] = {}
        # initialize _model_timedelta_t, for event discrete models: delta_t = None
        # therefore _model_timedelta_t = NaT which allways compares False
        # (Nat >= time), therefore models dont get executed when _model_timedelta_t = NaT
        self._model_timedelta_t[model.name] = pd.Timedelta(
            value = model.delta_t,
            unit = self.time_resolution)    # type: ignore

        # Add data logging capabilities
        self.add_watch_values_to_model(model, watch_values)
        self.add_heavy_watch_values_to_model(model, watch_heavy)

        return model

    @staticmethod
    def validate_model(model):
        '''Check if model fulfills the requirements for the simulation otherwise
        raise Error'''
        if not hasattr(model, 'name'):
            raise AttributeError(
                f'Model of type \'{type(model)}\' has no attribute name, which ' +
                'is required for the simulation')
        if not hasattr(model, 'delta_t'):
            raise AttributeError(
                f'Model of type \'{type(model)}\' has no attribute delta_t, which ' +
                'is required for the simulation '+
                '(if the model is event based, set delta_t=None)')
        if not hasattr(model, 'inputs'):
            raise AttributeError(
                f'Model \'{model.name}\' has no atribute \'inputs\', which is ' +
                'required for the simulation')
        if not isinstance(model.inputs, list):
            raise AttributeError(
                f'Model \'{model.name}\' \'inputs\' need to be a list of ' +
                'strings for the simulation to work')
        if not hasattr(model, 'outputs'):
            raise AttributeError(
                f'Model \'{model.name}\' has no atribute \'outputs\', which is ' +
                'required for the simulation')
        if not isinstance(model.outputs, list):
            raise AttributeError(
                f'Model \'{model.name}\' \'outputs\' need to be a list of ' +
                'strings for the simulation to work')
        if not hasattr(model, 'step'):
            raise AttributeError(
                f'Model \'{model.name}\' has no \'step\' function')
        if not callable(model.step):
            raise AttributeError(
                f'Model \'{model.name}\' \'step\' not a callable, which is ' +
                'required for the simulation to work')
        # Validation of duplicate input and outputs needed for watch values
        duplicate_input_outputs = list(set(model.inputs) & set(model.outputs))
        if duplicate_input_outputs:
            raise AttributeError(
                f'Model \'{model.name}\' contains the folowing duplicate value(s) ' +
                f'in inputs and outputs: {duplicate_input_outputs}')

        # parameters of the step function
        step_function_params = inspect.signature(model.step).parameters
        step_function_args = list(step_function_params.keys())
        if not step_function_args[0] == 'time':
            raise AttributeError(
                "First argument of step function needs to be 'time' " +
                f"(model '{model.name}')")
        # if variable kwargs in the signature, further checks are useless.
        # proceed at own risk
        if (not any([param.kind == param.VAR_KEYWORD
                     for param in step_function_params.values()])
            and not all([inp in step_function_args
                         for inp in model.inputs])):
            raise AttributeError(
                f"Not all inputs of '{model.name}' as specified in " +
                "'model.inputs' can be found in step function arguments " +
                "(make sure all inputs are accepted as keyword arguments, " +
                "including variable length kwargs)")

    def add_watch_values_to_model(self, model, watch_values: list):
        '''Adds provided inputs and/or outputs to the list of reccorded values.
        After complete simulation, the values can be found in sim.df

        Parameters
        ----------
        model : model of which inuts and/or output will be reccorded
        watch_values : list of str, input and/or output attributes of the model
            to be reccorded
        '''
        self.models_in_sim(model)
        for watch_value in watch_values:
            # Add model to watched models dict if needed
            # (only add if watch values present!!)
            if not model.name in self.model_watch_attributes:
                self.model_watch_attributes[model.name] = {}

            input_or_output = False

            if watch_value in model.inputs:
                if not 'inputs' in self.model_watch_attributes[model.name]:
                    self.model_watch_attributes[model.name]['inputs'] = []
                self.model_watch_attributes[model.name]['inputs'].append(
                    watch_value)
                input_or_output = True

            if watch_value in model.outputs:
                if not 'outputs' in self.model_watch_attributes[model.name]:
                    self.model_watch_attributes[model.name]['outputs'] = []
                self.model_watch_attributes[model.name]['outputs'].append(
                    watch_value)
                input_or_output = True

            if (watch_value.endswith(self.multiinput_symbol)
                or watch_value.endswith(self.multiinput_dict_symbol)):
                raise SimulationError(
                    f'Multiinput {watch_value} of model {model.name} cannot ' +
                    'be watched, consider watching individual model outputs.')

            if not input_or_output:
                raise SimulationError(
                    f'Non existant input or output {watch_value} of model ' +
                    f'{model.name} cannot be watched')

    # TODO: Those two methods could be unified or made less redundant
    def add_heavy_watch_values_to_model(self, model, watch_heavy: list):
        '''Adds provided inputs and/or outputs to the list of reccorded values.
        After complete simulation, the values can be found in sim.data

        Parameters
        ----------
        model : model of which inuts and/or output will be reccorded
        watch_heavy : list of str, input and/or output attributes of the model
            to be reccorded
        '''
        self.models_in_sim(model)
        for watch_value in watch_heavy:
            # Add model to watched models dict if needed
            # (only add if watch values present!!)
            if not model.name in self.model_heavy_watch_attributes:
                self.model_heavy_watch_attributes[model.name] = {}

            input_or_output = False

            if watch_value in model.inputs:
                if not 'inputs' in (
                        self.model_heavy_watch_attributes[model.name]):
                    self.model_heavy_watch_attributes[
                        model.name]['inputs'] = []
                self.model_heavy_watch_attributes[model.name]['inputs'].append(
                    watch_value)
                self.data[(model.name, 'inputs', watch_value)] = {}
                input_or_output = True

            if watch_value in model.outputs:
                if not 'outputs' in (
                        self.model_heavy_watch_attributes[model.name]):
                    self.model_heavy_watch_attributes[
                        model.name]['outputs'] = []
                self.model_heavy_watch_attributes[model.name]['outputs'].append(
                    watch_value)
                self.data[(model.name, 'outputs', watch_value)] = {}
                input_or_output = True

            if (watch_value.endswith(self.multiinput_symbol)
                or watch_value.endswith(self.multiinput_dict_symbol)):
                raise SimulationError(
                    f'Multiinput {watch_value} of model {model.name} cannot ' +
                    'be watched, consider watching individual model outputs.')

            if not input_or_output:
                raise SimulationError(
                    f'Non existant input or output {watch_value} of model ' +
                    f'{model.name} cannot be watched')


    def set_model_first_exec_time(self, model, first_exec_time: pd.Timestamp):
        """Sets the first execution time of a model"""
        self.models_in_sim(model)
        self._model_next_exec_time[model.name] = first_exec_time

    def connect(self, model1, model2, *connections, time_shifted=False,
                 init_values=None, triggers=None):
        """
        Connects attribute(s) of model1 to model2

        Parameters
        ----------
        model1 : model of which the output is connected
        model2 : model of which the input is provided
        *connections : str or tuple, str 'atr' or tuple (attr_out,
            attr_in) specifying output (of model1) and input (of model2)
            attributes of the models that need to be connected. if the
            attributes have the same name, a string can be passed
        time_shifted : bool, To prevent computational loops, time_shifted
            connections take the input provided at the last step.
        init_values : dict, Time shifted connections need an initial value
            for the outputs of model1 (attr_out !) to function in the first
            step.
        triggers : list, output attributes (attr_out) of model1, that
            trigger the execution of model2 when the value changes. 'Big'
            items should probably be avoided as triggers for performance
            reasons
        """

        if triggers is None:
            triggers = []

        self.models_in_sim(model1, model2)

        for connection in connections:  # TODO: refactor!
            # convert to tuple, if str is passed (if input and output have the
            # same 'name', only a string can be passed)
            if isinstance(connection, str):
                connection = (connection, connection)

            attribute_out, attribute_in = connection

            # Check if model1 provides this output
            if not attribute_out in model1.outputs:
                raise SimulationError(
                    f'Error when connecting {model1.name} to {model2.name}: ' +
                    f'\'{attribute_out}\' not an output of model ' +
                    f'\'{model1.name}\'')

            # Check if model2 accepts this input
            if not attribute_in in model2.inputs:
                raise SimulationError(
                    f'Error when connecting {model1.name} to {model2.name}: ' +
                    f'\'{attribute_in}\' not an input of model ' +
                    f'\'{model2.name}\'')

            # fill self._model_input_map[model2.name]
            if (attribute_in.endswith(self.multiinput_symbol) or
                attribute_in.endswith(self.multiinput_dict_symbol)):
                if attribute_in in self._model_input_map[model2.name]:
                    if (model1.name, attribute_out) in \
                        self._model_input_map[model2.name][attribute_in]:

                        raise SimulationError("Duplicate connection detected: "+\
                            f"'{model1.name}', '{attribute_out}' to" +\
                            f" '{model2.name}', '{attribute_in}'")

                    self._model_input_map[model2.name][attribute_in].append((model1.name,
                        attribute_out))
                else:
                    self._model_input_map[model2.name][attribute_in] = \
                        [(model1.name, attribute_out)]
                    # same connection!! raise!
            elif attribute_in in self._model_input_map[model2.name]:
                raise SimulationError("Trying to set multiple inputs for " + \
                    f"'{model2.name}' '{attribute_in}'")
            else:
                self._model_input_map[model2.name][attribute_in] = (model1.name, attribute_out)

            # initialize time shifted connection (start value)
            if time_shifted:
                if not init_values or not attribute_out in init_values:
                    raise ValueError(
                        'Time shifted connection requires init_value for ' +
                        f'\'{attribute_out}\' e.g.: {{\'{attribute_out}\': ' +
                        'value}}')
                # TODO: check if this sounds missleading: output attribute
                # provided but not seen in watch items
                self._outputs[model1.name][attribute_out] = (
                    init_values.pop(attribute_out))
                # TODO Check for duplicate init value if several models
                # ininitlaize connections to the same output, the value is just
                # overridden. Solution: raise Error if the value is different!

            # Add triggering attributes (model1 triggers execution of model2)
            if attribute_out in triggers:
                triggers.remove(attribute_out)
                if not model1.name in self._model_output_triggers_models:
                    # create empty dict
                    self._model_output_triggers_models[model1.name] = {}
                if (not attribute_out
                        in self._model_output_triggers_models[model1.name]):
                    self._model_output_triggers_models[model1.name][
                        attribute_out] = []

                self._model_output_triggers_models[model1.name][
                    attribute_out].append(model2)

            # Add connection to graph
            self.graph.add_edge(model1, model2, name=connection,
                             time_shifted=time_shifted)

        if init_values:
            raise ValueError(
                'Found init_value(s) without corresponding connection: ' +
                f'{init_values}')
        if triggers:
            raise ValueError(
                'Found trigger(s) without corresponding connection: ' +
                f'{triggers}')

    def connect_constant(self, constant, model, *attributes):
        """Connects constant value(s) to model input(s)"""

        self.models_in_sim(model)

        for attr in attributes:
            # fill self._model_input_map[model.name]
            if attr in self._model_input_map[model.name]:
                raise SimulationError(
                    f"Trying to set multiple inputs for '{model.name}' " +
                    f"'{attr}'")
            else:
                self._model_input_map[model.name][attr] = (
                    model.name+'_const', attr)

            # add inputs to the output dict, this value will then not be
            # changed during simulation
            if not model.name+'_const' in self._outputs:
                self._outputs[model.name+'_const'] = {}
            self._outputs[model.name+'_const'][attr] = constant

    def connect_nothing(self, model1, model2, time_shifted=False):
        '''
        This method ensures correct execution order when models interact
        independently (e.g. interfaces not handled via simplec).
        '''
        self.models_in_sim(model1, model2)
        # Add connection to graph
        self.graph.add_edge(model1, model2, name='nothing',
                         time_shifted=time_shifted)

    def models_in_sim(self, *models):
        '''Raises SimulationError if the models are not added to the
        simulation'''
        for model in models:
            if model.name not in self._outputs:
                raise SimulationError(
                    f'Model {model.name} needs to be added to the ' +
                    'simulation first before it can be connected with ' +
                    'sim.add_model(...)')

    @staticmethod
    def _compute_execution_order_from_graph(graph):
        '''Computes the models execution order from the initialized nx.graph
        with models at the node and connections between the models represented
        as edges'''

        # Prepare execution graph
        edges_same_time = [(u, v) for u, v, d in graph.edges(data=True)
                           if not d['time_shifted']]
        edges_time_shifted = [(u, v) for u, v, d in graph.edges(data=True)
                              if d['time_shifted']]

        graph_same_time = graph.edge_subgraph(edges_same_time)
        graph_time_shifted = graph.edge_subgraph(edges_time_shifted)

        # check if model connections are feasible (no cyclic connections)
        if not nx.is_directed_acyclic_graph(graph_same_time):
            raise SimulationError('There must be a cycle in the connections')

        # Reverse time shifted edges, to ensue, that models containing input
        # for another model in the next step are steped last.
        # The concept has to be double checked! (Is this needed?)
        graph_direction_check = graph_same_time.copy()
        graph_direction_check.add_edges_from([(v, u) for u, v
                                          in graph_time_shifted.edges])

        if not nx.is_directed_acyclic_graph(graph_direction_check):
            raise SimulationError(
                'There might be a problem with the time-shifted connections')

        # determine model execution order
        # return sorted model execution_list
        return list(nx.topological_sort(graph_direction_check))

    def _get_model_inputs_from_outputs(self, model):
        '''Get inputs of the model from outputs of the simulation'''
        inputs = {}
        for input_key, output_model_and_attr in (
                self._model_input_map[model.name].items()):
            if input_key.endswith(self.multiinput_symbol):  # multiinput
                inputs[input_key] = []
                for output_model_name, output_attr in output_model_and_attr:
                    inputs[input_key].append(
                        self._outputs[output_model_name][output_attr])
            elif input_key.endswith(self.multiinput_dict_symbol):  # multiinput_dict
                inputs[input_key] = {}
                for output_model_name, output_attr in output_model_and_attr:
                    inputs[input_key][output_model_name] = \
                        self._outputs[output_model_name][output_attr]
            else:
                output_model_name, output_attr = output_model_and_attr
                inputs[input_key] = (
                    self._outputs[output_model_name][output_attr])
        return inputs

    def _initialize_watch_value_dataframe(self, datetimes):
        # TODO: find nice solution for different time resolutions to avoid
        # sparsity (might aswell not be a storage issue)
        # TODO: refactor (exchanging df with array was a quickfix, therefore
        # messy code)
        if self.model_watch_attributes:
            ind = 0
            # make dataframe MultiIndex from watch attributes dict
            dataframe_index_structure = []
            for mname, inpoutpdict in self.model_watch_attributes.items():
                self.model_watch_attributes_index[mname] = {}
                for inpoutp, attrs in inpoutpdict.items():
                    self.model_watch_attributes_index[mname][inpoutp] = []
                    for attr in attrs:
                        self.model_watch_attributes_index[mname][
                            inpoutp].append((attr, ind))
                        ind += 1
                        dataframe_index_structure.append(
                            (mname, inpoutp, attr))

            self._df_multiindex = pd.MultiIndex.from_tuples(
                dataframe_index_structure,
                names=['model', 'i/o', 'attribute'])
            # index=datetimes, columns=pd.MultiIndex.from_tuples(
            # dataframe_index_structure,
            # names=['model', 'i/o', 'attribute'])
            self.df = pd.DataFrame([])

            self.array = np.full((len(datetimes), ind), np.nan, np.float64)

    def _update_model_next_exec_time(self, model, model_outputs, time):
        '''get (and remove) next model execution time. If model is event
        based ('next_exec_time' in model_outputs) use this value, else use
        delta_t'''
        self._model_next_exec_time[model.name] = model_outputs.pop(
            'next_exec_time',
            self._model_timedelta_t[model.name] + time)

    def _set_next_exec_time_for_dependend_models(self, model, model_outputs,
                                                  time):
        if model.name in self._model_output_triggers_models:
            for trigger_attribute, trigger_models in (
                    self._model_output_triggers_models[model.name].items()):
                # if triggering outputs have changed
                if (trigger_attribute in model_outputs
                    and self._outputs[model.name][trigger_attribute]
                        != model_outputs[trigger_attribute]):
                    # execute the triggered model as soon as possible
                    # (does not break execution order!)
                    for trigger_model in trigger_models:
                        self._model_next_exec_time[trigger_model.name] = time

    def _log_data(self, model, model_inputs, model_outputs, time, t):
        # logg data
        if model.name in self.model_watch_attributes:
            watch_attr_index = self.model_watch_attributes_index[model.name]
            if 'inputs' in watch_attr_index:
                for wi, i in watch_attr_index['inputs']:
                    self.array[t, i] = model_inputs[wi]
            if 'outputs' in watch_attr_index:
                for wo, i in watch_attr_index['outputs']:
                    self.array[t, i] = model_outputs.get(wo, np.nan)

        # logg heavy data
        if model.name in self.model_heavy_watch_attributes:
            if 'inputs' in self.model_heavy_watch_attributes[model.name]:
                for wi in (self.model_heavy_watch_attributes[
                        model.name]['inputs']):
                    self.data[(model.name, 'inputs', wi)].update(
                        {time: model_inputs[wi]})
            if 'outputs' in self.model_heavy_watch_attributes[model.name]:
                for wi in (self.model_heavy_watch_attributes[
                        model.name]['outputs']):
                    self.data[(model.name, 'outputs', wi)].update(
                        {time: model_outputs[wi]})

    @staticmethod
    def check_all_model_inputs_provided(model, connected_inputs):
        '''Check if all inputs of the model are provided with an input, raise
        SimulationError otherwise'''
        missing_inputs = model.inputs.copy()
        for model_inp in model.inputs:  # go through all model input attrs
            # go through all connections of the model
            for connected_inp in connected_inputs:
                if connected_inp == model_inp:  # check if they are equal
                    missing_inputs.remove(model_inp)
                    break
        if missing_inputs:
            raise SimulationError(
                f'Not all inputs of model \'{model.name}\' are provided. ' +
                f'Missing inputs: {missing_inputs}')

    def simulate(self, datetimes,
                 output_data_path: str | None = None,
                 enable_progress_bar: bool = True,
                 model_first_exec_time_default:
                    pd.Timestamp = pd.to_datetime('1970-01-01 00:00:00+01:00')):
        '''Runs the simulation

        Parameter
        ---------
        datetimes : pd.DatetimeIndex specifying the simulation interval and
            steps.
        output_data_path : str or None, path for the output pandas.DataFrame to be
            saved to. The extension specifies the filetype. Options are: '.pkl',
            '.csv', '.parquet'.
        enable_progress_bar : bool show a progress bar while simulateing (disable for headless
        / background use)
        model_first_exec_time_default : pd.DateTime, Simulation-time to execute
            all models the first time (when using historic value, models get
            executed at first time step).
        '''

        # set per-run options (moved from __init__ to simulate)
        if output_data_path is not None:
            self.output_data_path = Path(output_data_path)
            self.output_data_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.output_data_path = None

        self.enable_progress_bar = enable_progress_bar
        self.model_first_exec_time_default = model_first_exec_time_default

        self.log.info('Preparing simulation...')
        sorted_model_execution_list = (
            self._compute_execution_order_from_graph(self.graph))

        # initialize model first execution times for this run
        # Do not overwrite values that were manually set prior to running
        for model in sorted_model_execution_list:
            if model.name not in self._model_next_exec_time:
                self.set_model_first_exec_time(model, self.model_first_exec_time_default)

        for model in sorted_model_execution_list:
            self.check_all_model_inputs_provided(
                model, self._model_input_map[model.name])

        self._initialize_watch_value_dataframe(datetimes)

        # logging and status updates
        self.log.info('Running simulation')

        # init local variables
        time = pd.Timestamp('1970-01-01 00:00:00+01:00')
        model = sorted_model_execution_list[0]

        try:
            # add progress bar
            for t, time in tqdm.tqdm(
                    enumerate(datetimes), total=len(datetimes),
                    desc='Progress', unit='Steps', maxinterval=60,
                    disable=not self.enable_progress_bar):
                for model in sorted_model_execution_list:
                    # determine if the model needs to be steped in this time
                    # step
                    if time >= self._model_next_exec_time[model.name]:

                        model_inputs = (
                            self._get_model_inputs_from_outputs(model))

                        # calculate next model step
                        # copy for safety
                        model_outputs = copy(model.step(time, **model_inputs))

                        self._update_model_next_exec_time(
                            model, model_outputs, time)

                        self._set_next_exec_time_for_dependend_models(
                            model, model_outputs, time)

                        # write model outputs to the ouput dict
                        self._outputs[model.name].update(model_outputs)

                        self._log_data(
                            model, model_inputs, model_outputs, time, t)

        except Exception as e:
            self.log.error(
                "Error occured at sim-time '%s' and during the " +
                "processing of model '%s'", time, model.name, exc_info=e)
            raise SimulationError(
                f"Error occured at sim-time '{time}' and during the " +
                f"processing of model '{model.name}'") from e
        finally:
            # create DataFrame
            if hasattr(self, 'array'):
                self.df = pd.DataFrame(
                    self.array, index=datetimes, columns=self._df_multiindex)
            # Save DataFrame
            if (self.model_watch_attributes and self.output_data_path
                    and not self.df.empty):
                self.log.info("Saving output to %s!", self.output_data_path)
                if self.output_data_path.suffix == '.pkl':
                    self.df.to_pickle(self.output_data_path)
                elif self.output_data_path.suffix == '.csv':
                    self.df.to_csv(self.output_data_path)
                elif self.output_data_path.suffix == '.parquet':
                    self.df.to_parquet(self.output_data_path)
                else:
                    self.df.to_pickle(self.output_data_path)

            self.log.info('Simulation completed successfully!')

    def draw_exec_graph(self):
        """Draws the execution graph of the simulation using matplotlib"""

        plt.figure()
        pos = nx.spring_layout(self.graph)

        # get the model names for the edges
        node_labels = {m: m.name for m in self.graph.nodes}
        # get all edges to assign the color (safer that way)
        edges = self.graph.edges(data=True)
        edge_colors = ["green" if not d['time_shifted'] else "red"
                       for u, v, d in edges]

        nx.draw_networkx(self.graph, pos,
                         edgelist=edges, # type: ignore
                         edge_color=edge_colors,
                         connectionstyle='arc3, rad = 0.1',
                         labels=node_labels)

        plt.show()

# todo: implement visualization again (deleted at commit a08d79432da22b1f35bc2167e8697fadcd88f8a5)
