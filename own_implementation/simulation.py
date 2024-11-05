'''Simple Simulation framework for the time-discrete simulation of energy models

The framework can be used to connect and simulate time discrete (and maybe event discrete) models and handle the model outputs.

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
 - to implement the inputs list, specifying the inputs of the step function as strings
 - to implement the outputs list, specifying the outputs of the step function as strings
 - to implement the delta_t attribute (in seconds)
 - to implement a step function 
    - the step function needs to accept an argument time and all arguments specified in \'inputs\'
    - the step function needs to return a dict with the outputs as keys


The simulation can then be run following this simple example (utilizing the example model from above):

from simulation import Simulation
import pandas

sim = Simulation()

model1   = ExampleModel('example_1')
model2   = ExampleModel('example_2')

sim.add_model(model1, watch_values=['value_in'])
sim.add_model(model2, watch_values=['value_out'])

sim.connect(model1, model2, ('value_out', 'value_in'))
sim.connect(model2, model1, ('value_out', 'value_in'), time_shifted=True, init_values={'value_out': 1})

times = pandas.date_range('2021-01-01 00:00:00', '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')

sim.run(times)

outputs of the simulation can then be retrieved by accessing sim.df
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
import logging
import tqdm 
# from tqdm.contrib.logging import logging_redirect_tqdm
import inspect

from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.cm as cm

class SimulationError(Exception):
    pass

class Simulation():
    def __init__(self, output_to_file=False, output_data_path='output_data.pkl', logger=None, **kwargs) -> None:
        '''
        Creates a simulation object
        
        Parameters
        ----------
        output_data_path : str path for the output pandas.DataFrame to be saved
        log_to_file : Bool log the simulation output to a log file
        log_file_path : str file for the logging output
        '''

        self.output_to_file = output_to_file
        self.output_data_path = output_data_path
        
        self._model_names = []  # initialize lists with model_names, state_name and output_names. those get filled in add_model and is used to check for duplicate names
        self._outputs = {}  # this dict contains all models(names) and their output (if any) structure: nested dict e.g. {'model1name':{'output1': np.nan}, 'model2': {}}. The models will collect their inputs from this output
        self._G = nx.DiGraph()  # initialize execution graph (the graph is needed to calculate the model execution order)
        
        # set up a logger
        if logger:
            self.log = logger
            # logging_redirect_tqdm(self.log)
        else:
            self.log = logging.getLogger(__name__) # create unused logger
            self.log.disabled = True
        
        # setup the data logging capabilities
        self.model_watch_attributes = {}

        self.time_resolution = kwargs.pop('time_resolution', 'sec') # get time_resolution (of the models) form kwargs, default: seconds
        self.model_first_exec_time_default = kwargs.pop('model_first_exec_time_default', pd.to_datetime('1970-01-01 00:00:00+01:00'))

        # # set up the internal values for the model connection handling
        
        # input_map answers the question for the model "where do i get my inputs from?" 
        # It maps the models inputs to the simulation self._outputs dict {'modelname': {'attribute_in': (model1name, attribute_out)}}
        # It gets filled in 'connect'. 
        self._model_input_map = {}

        # datetime timedeltas of the models for fast next_exec_time calculation,
        # it contains the models timedelta converted to pd.Timedelta
        self._model_timedelta_t = {}

        # next_exec_time answers the question 'when will the model be executed next time?'
        # initialized with 1970 date to make sure all models are stepped in the first timestep
        self._model_next_exec_time = {}

        # triggers_model answers the question 'which models are triggert by a certain model output (attr_out)?'
        # structure: {model: {'attr_out': [modeltotrigger1, modeltotrigger2]}}
        # It gets filled in 'connect', 
        # model not in dict, if model triggers no model, 
        # output not in inner dict if output triggers no models 
        self._model_output_triggers_models = {}

    def _validate_model(self, model):
        '''Check if model fulfilles the requirements for the simulation otherwise raise AttributeError'''
        if not hasattr(model, 'name'): 
            raise AttributeError(f'Model of type \'{type(model)}\' has no attribute name, which is required for the simulation')
        # if not hasattr(model, 'delta_t'): # not needed for event based models
        #     raise AttributeError(f'Model \'{model.name}\' has no atribute \'delta_t\', which is required for the simulation')
        if not hasattr(model, 'inputs'): 
            raise AttributeError(f'Model \'{model.name}\' has no atribute \'inputs\', which is required for the simulation')
        if not type(model.inputs) == list: 
            raise AttributeError(f'Model \'{model.name}\' \'inputs\' need to be a list of strings for the simulation to work')
        if not hasattr(model, 'outputs'): 
            raise AttributeError(f'Model \'{model.name}\' has no atribute \'outputs\', which is required for the simulation')
        if not type(model.outputs) == list: 
            raise AttributeError(f'Model \'{model.name}\' \'outputs\' need to be a list of strings for the simulation to work')
        if not hasattr(model, 'step'): 
            raise AttributeError(f'Model \'{model.name}\' has no \'step\' function')
        if not callable(model.step): 
            raise AttributeError(f'Model \'{model.name}\' \'step\' not a callable, which is required for the simulation to work')
        
        step_function_params = inspect.signature(model.step).parameters # parameters of the step function
        step_function_args = list(step_function_params.keys())
        if not step_function_args[0] == 'time': 
            raise AttributeError(f'First argument of step function needs to be \'time\' (model \'{model.name}\')')
        if (not any([param.kind == param.VAR_KEYWORD for param in step_function_params.values()]) # if variable kwargs in the signature, further checks are useless. proceed at own risk
            and not all([inp in step_function_args for inp in model.inputs])): 
                raise AttributeError(f'Not all inputs of model.inputs can be found in step function arguments (model \'{model.name}\')')
            
    def add_model(self, model, watch_values=[]):
        '''Adds a model to the simulation.

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

        watch_values : adds a models input or output to the reccorded values (pandas.DataFrame). Only watch numeric values that can safely be cast into a pandas DataFrame 
        '''
        self._validate_model(model)

        # check for duplicate models
        if model.name in self._model_names:   ## TODO remove self._model_names and use self._outputs
            raise SimulationError(f'Model.name \'{model.name}\' not unique, which is required for the simulation to work')
        self._model_names.append(model.name)
        
        # add model outputs to the simulation output dict 
        self._outputs.update({model.name: {output_attr: np.nan for output_attr in model.outputs}})
        # add model to the execution graph
        self._G.add_node(model)

        # # Add model connection attributes
        self._model_input_map[model.name] = {}
        self._model_timedelta_t[model.name] =  pd.Timedelta(getattr(model, 'delta_t', np.nan), self.time_resolution)
        self.set_model_first_exec_time(model, self.model_first_exec_time_default)

        self.add_watch_values_to_model(model, watch_values)

    def add_watch_values_to_model(self, model, watch_values:list):
        '''Adds values to \'_sim_watch_input/output\' attribute of model'''
        for wv in watch_values:
            if not model.name in self.model_watch_attributes:
                self.model_watch_attributes[model.name] = {}
            
            input_or_output = False
            if wv in model.inputs:
                if not 'inputs' in self.model_watch_attributes[model.name]:
                    self.model_watch_attributes[model.name]['inputs'] = []
                self.model_watch_attributes[model.name]['inputs'].append(wv)
                input_or_output = True
            if wv in model.outputs:
                if not 'outputs' in self.model_watch_attributes[model.name]:
                    self.model_watch_attributes[model.name]['outputs'] = []
                self.model_watch_attributes[model.name]['outputs'].append(wv)
                input_or_output = True
            if not input_or_output: raise SimulationError(f'Non existant input or output {wv} cannot be watched')

    def set_model_first_exec_time(self, model, first_exec_time:pd.Timestamp):
        self._model_next_exec_time[model.name] = first_exec_time

    @staticmethod
    def _make_output_attribute_key(model, output_attr_name):
        '''Produces a unique key for the Simulation._ouputs dict '''
        return model.name+'_'+output_attr_name

    def connect(self, model1, model2, *connections, time_shifted=False, init_values=None, triggers=[]):
        '''Connects attribute(s) of model1 to model2
        
        Parameters
        ----------
        model1 : model of which the output is connected
        model2 : model of which the input is provided
        *connections : str or tuple, str 'atr' or tuple (attr_out, attr_in) specifying output (of model1) and input (of model2) attributes of the models that need to be connected.
        if the attributes have the same name, a string can be passed
        time_shifted : bool, To prevent computational loops, time_shifted connections take the input provided at the last step.
        init_values : dict, Time shifted connections need an initial value for the outputs of model1 (attr_out !) to function in the first step.
        triggers : list, output attributes (attr_out) of model1, that trigger the execution of model2 when the value changes. 'Big' items should probably be avoided as triggers for performance reasons
        '''
        for m in [model1, model2]:
            if m.name not in self._model_names: raise SimulationError(f'Model {m.name} needs to be added to the simulation first before it can be connected with sim.add_model(...)')

        for connection in connections: # TODO: refactor!
            # convert to tuple, if str is passed (if input and output have the same 'name', only a string can be passed)
            if type(connection) == str: 
                connection = (connection, connection)
            
            attribute_out, attribute_in = connection

            # fill self._model_input_map[model2.name]
            if attribute_in in self._model_input_map[model2.name]:
                raise SimulationError(f"Trying to set multiple inputs for '{model2.name}' '{attribute_in}'")
            else:
                self._model_input_map[model2.name][attribute_in] = (model1.name, attribute_out)
            
            # initialize time shifted connection (start value)  
            if time_shifted: 
                if not init_values or not attribute_out in init_values: 
                    raise ValueError(f'Time shifted connection requires init_value for \'{attribute_out}\' e.g.: {{\'{attribute_out}\': value}}')
                self._outputs[model1.name][attribute_out] = init_values.pop(attribute_out) # TODO: check if this sounds missleading: output attribute provided but not seen in watch items

            # Add triggering attributes (model1 triggers execution of model2)
            if attribute_out in triggers:
                triggers.remove(attribute_out)
                if not model1.name in self._model_output_triggers_models:
                    self._model_output_triggers_models[model1.name] = {} # create empty dict
                if not attribute_out in self._model_output_triggers_models[model1.name]:
                    self._model_output_triggers_models[model1.name][attribute_out] = []

                self._model_output_triggers_models[model1.name][attribute_out].append(model2)

            self._G.add_edge(model1, model2, name=connection, time_shifted=time_shifted)

        if init_values: raise ValueError(f'Found init_value(s) without corresponding connection: {init_values}')
        if triggers: raise ValueError(f'Found trigger(s) without corresponding connection: {triggers}')

    @staticmethod
    def _compute_execution_order_from_graph(G):
        '''Computes the models execution order from the initialized nx.graph with models at the node and connections between the models represented as edges'''
        
        # Prepare execution graph
        edges_same_time    = [(u,v) for u, v, d in G.edges(data=True) if not d['time_shifted']]
        edges_time_shifted = [(u,v) for u, v, d in G.edges(data=True) if d['time_shifted']]

        G_same_time = G.edge_subgraph(edges_same_time)
        G_time_shifted = G.edge_subgraph(edges_time_shifted)

        # check if model connections are feasible (no cyclic connections)
        if not nx.is_directed_acyclic_graph(G_same_time):
            raise SimulationError('There must be a cycle in the connections')
        
        # Reverse time shifted edges, to ensue, that models containing input for another model in the next step are steped last.
        # The concept has to be double checked! (Is this needed?)
        G_direction_check = G_same_time.copy()
        G_direction_check.add_edges_from([(v, u) for u, v in G_time_shifted.edges])

        if not nx.is_directed_acyclic_graph(G_direction_check):
            raise SimulationError('There might be a problem with the time-shifted connections')
        
        # determine model execution order
        return list(nx.topological_sort(G_direction_check)) # return sorted model execution_list

    def _get_model_inputs_from_outputs(self, model):
        inputs = {}
        for input_key, (output_model_name, output_attr) in self._model_input_map[model.name].items():
            inputs[input_key] = self._outputs[output_model_name][output_attr]
        return inputs
    
    def _initialize_watch_value_dataframe(self):
        # TODO: find nice solution for different time resolutions to avoid sparsity
        if self.model_watch_attributes:
            # make dataframe MultiIndex from watch attributes dict
            dataframe_index_structure = []
            for mname, inpoutpdict in self.model_watch_attributes.items():
                for inpoutp, attrs in inpoutpdict.items():
                    for attr in attrs:
                        dataframe_index_structure.append((mname, inpoutp, attr)) 
            self.df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(dataframe_index_structure, names=['model', 'i/o', 'attribute']))
    
    def run(self, datetimes):
        '''Runs the simulation on the index datetimes
        
        Arguments:
        ---------
        datetimes : pd.DatetimeIndex specifying the simulation interval and steps.
        '''
        self.log.info('Preparing simulation...')
        sorted_model_execution_list  = self._compute_execution_order_from_graph(self._G)

        # Check if all inputs of the model are provided with an input
        for m in sorted_model_execution_list:
            missing_inputs = [inp for inp in m.inputs if inp not in self._model_input_map[m.name]]
            if missing_inputs:
                raise SimulationError(f'Not all inputs of model \'{m.name}\' are provided. Missing inputs: {missing_inputs}')

        self._initialize_watch_value_dataframe()
        
        # logging and status updates
        self.log.info('Running simulation')
        n_steps = len(datetimes)
        pbar = tqdm.tqdm(total=n_steps, desc='Progress', unit='Steps', maxinterval=60)
        
        try:
            for time in datetimes:
                pbar.update(1) # update progress bar (for use in console)

                # step models
                for model in sorted_model_execution_list:
                    # determine if the model needs to be steped in this time step
                    if time >= self._model_next_exec_time[model.name]:
                        # get inputs of the model from outputs of the simulation
                        model_inputs = self._get_model_inputs_from_outputs(model)
                        
                        # calculate next model step
                        model_outputs = copy(model.step(time, **model_inputs)) # copy for safety

                        # get (and remove) next model execution time. If model is event based ('next_exec_time' in  model_outputs) use this value, else use delta_t
                        self._model_next_exec_time[model.name] = model_outputs.pop('next_exec_time', self._model_timedelta_t[model.name] + time)

                        # check for trigger outputs
                        if model.name in self._model_output_triggers_models:
                            for trigger_attribute, trigger_models in self._model_output_triggers_models[model.name].items():
                                # if triggering outputs have changed
                                if self._outputs[model.name][trigger_attribute] != model_outputs[trigger_attribute]: 
                                    # execute the triggered model as soon as possible (does not break execution order!)
                                    for trigger_model in trigger_models:
                                        self._model_next_exec_time[trigger_model.name] = time

                        self._outputs[model.name].update(model_outputs) # write model outputs to the ouput dict

                        # logg data
                        if model.name in self.model_watch_attributes:
                            if 'inputs' in self.model_watch_attributes[model.name]:
                                self.df.loc[time, (model.name, 'inputs', slice(None))] = [model_inputs[wi] for wi in self.model_watch_attributes[model.name]['inputs']]
                            if 'outputs' in self.model_watch_attributes[model.name]:
                                self.df.loc[time, (model.name, 'outputs', slice(None))] = [model_outputs[wo] for wo in self.model_watch_attributes[model.name]['outputs']]
                    
        except Exception as e:
            self.log.error(f'Error occured at sim-time \'{time}\' and during the processing of model \'{model.name}\'', exc_info=e)
            raise SimulationError(f'Error occured at sim-time \'{time}\' and during the processing of model \'{model.name}\'') from e
        finally:
            pbar.close()
            if self.model_watch_attributes and self.output_to_file and not self.df.empty:
                self.log.info(f'Saving output to {self.output_data_path}!') 
                self.df.to_pickle(self.output_data_path)
            self.log.info(f'Simulation completed successfully!') 

    def draw_exec_graph(self):
        plt.figure()
        pos=nx.spring_layout(self._G)

        node_labels = {m:m.name for m in self._G.nodes} # get the model names for the edges
        edges = self._G.edges(data=True)  # get all edges to assign the color (safer that way)
        edge_colors = ["green" if not d['time_shifted'] else "red" for u, v, d in edges]

        nx.draw_networkx(self._G, pos, edgelist=edges, edge_color=edge_colors, connectionstyle='arc3, rad = 0.1', labels=node_labels)

        plt.show()

    # def draw_execution_visualization(self):
    #     if not self._is_prepared:
    #         raise ValueError('Simulation needs to be prepared to be able to plot visualization (call prepare)')

    #     deltas = [m.delta_t for m in self.model_execution_list]  # duplicate!!! maybe move to prepare
    #     prev_time = [-60*60 for _ in self.model_execution_list]

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    #     # datetimes = pd.date_range('2021-01-01 00:00:00 +01:00', pd.to_datetime('2021-01-03 00:00:00 +01:00') + pd.Timedelta(max(deltas)*2), freq=min(deltas), tz='UTC+01:00')

    #     # fig.set_size_inches(6,6)

    #     # position in the plot for initialization dot
    #     x_pos_init = -min(deltas)
    #     y_pos_init = (len(self.model_execution_list)-1)/2

    #     pos_of_output = np.array([[0, 0] if l == np.nan else [x_pos_init, y_pos_init] for l in self.output_list]) # Positions of the output in the plot for creating the arrows


    #     cmap = cm.get_cmap('viridis', lut=len(self.model_execution_list))
    #     clist = [cmap(i) for i in range(len(self.model_execution_list))]

    #     spacing = 5

    #     times = np.arange(0, max(deltas)*2, min(deltas))

    #     for time in times:
                
    #         # determine which models need to be steped in this time step 
    #         step_models = [[i, m] for i, [m, p, d] in enumerate(zip(self.model_execution_list, prev_time, deltas)) if time-p >= d]
            
    #         # step models
    #         for i, model in step_models:
    #             # save current time as last execution time of model 
    #             prev_time[i] = time

    #             # Box for the total delta t of the model
    #             ax.add_patch( FancyBboxPatch((time-spacing/2, -i-0.4),
    #                 model.delta_t-spacing, 0.8,
    #                 boxstyle="round, pad=0.40, rounding_size=5",  #  ",
    #                 alpha=0.5,
    #                 ec="none", fc=clist[i],
    #                 mutation_aspect=1/60
    #                 ))
    #             # Box for the calculation "time"
    #             ax.add_patch( FancyBboxPatch((time-spacing/2, -i-0.4),
    #                 spacing, 0.8,
    #                 boxstyle="round, pad=0.40, rounding_size=5",  #  rounding_size=0.15",
    #                 ec="none", fc=clist[i],
    #                 mutation_aspect=1/60
    #                 ))

    #             # plot arrows from the output positions to the current model
    #             i_inp = getattr(model, 'input_selector', 'no_inp')
    #             if i_inp != 'no_inp':
    #                 pos_of_output_i = pos_of_output[i_inp, :]
    #                 for xin, yin in pos_of_output_i:
    #                     ls = '-' if time == xin else '--'
    #                     arrow_c = clist[int(yin)] if xin >= 0 else 'k'
    #                     ax.annotate("", xy=(time, -i), xytext=(xin, -yin), arrowprops=dict(arrowstyle='-|>', linestyle=ls, color=arrow_c, linewidth=2))  # arrowstyle='-|>', , headwidth=5

                
    #             pos_of_output[model.output_selector, :] = np.array([[time, i] for _ in model.output_selector])

    #     ax.plot([x_pos_init], [-y_pos_init], marker = 'o', markersize=10, color='k')

    #     ax.set_xticks(times)
    
    #     ax.set_yticks([-i for i in range(len(self.model_execution_list))])
    #     ax.set_yticklabels([f'{i+1:03} {m.name:15}' for i, m in enumerate(self.model_execution_list)])

    #     ax.autoscale()  
    #     # fig.show()