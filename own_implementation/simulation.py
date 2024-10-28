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
from tqdm.contrib.logging import logging_redirect_tqdm
import inspect

from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.cm as cm

class SimulationError(Exception):
    pass

class Simulation():
    def __init__(self, output_to_file=False, output_data_path='output_data.pkl', log_to_file=False, log_file_path='simulation.log') -> None:
        '''Creates a simulation object
        
        Attributes:
        output_data_path:str -- path for the output pandas.DataFrame to be saved
        log_to_file:Bool -- log the simulation output to a log file
        log_file_path:str -- file for the logging output
        '''
        self.output_to_file = output_to_file
        self.output_data_path = output_data_path
        
        self._model_names = []  # initialize lists with model_names, state_name and output_names. those get filled in add_model and is used to check for duplicate names
        self._outputs = {}  # this dict contains all the output values of the models, the models will collect their inputs from this output
        self._G = nx.DiGraph()  # initialize execution graph (the graph is needed to calculate the model execution order)
        
        # set up a logger
        if log_to_file:
            self._log_file_name = log_file_path
            logging.basicConfig(filename=self._log_file_name, encoding='utf-8', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        self.log = logging.getLogger(__name__)
        self.log.disabled = not log_to_file
        logging_redirect_tqdm(self.log)

        # setup the data logging capabilities
        self.dataframe_index_structure = []

    def _validate_model(self, model):
        '''Check if model fulfilles the requirements for the simulation'''
        if not hasattr(model, 'name'): raise AttributeError(f'Model of type \'{type(model)}\' has no attribute name, which is required for the simulation')
        if not hasattr(model, 'delta_t'): raise AttributeError(f'Model \'{model.name}\' has no atribute \'delta_t\', which is required for the simulation')
        if not hasattr(model, 'inputs'): raise AttributeError(f'Model \'{model.name}\' has no atribute \'inputs\', which is required for the simulation')
        if not type(model.inputs) == list: raise AttributeError(f'Model \'{model.name}\' \'inputs\' need to be a list of strings for the simulation to work')
        if not hasattr(model, 'outputs'): raise AttributeError(f'Model \'{model.name}\' has no atribute \'outputs\', which is required for the simulation')
        if not type(model.outputs) == list: raise AttributeError(f'Model \'{model.name}\' \'outputs\' need to be a list of strings for the simulation to work')
        if not hasattr(model, 'step'): raise AttributeError(f'Model \'{model.name}\' has no \'step\' function')
        if not callable(model.step): raise AttributeError(f'Model \'{model.name}\' \'step\' not a callable, which is required for the simulation to work')
        step_args = list(inspect.signature(model.step).parameters.keys())
        if not step_args[0] == 'time': raise AttributeError(f'First argument of step function needs to be \'time\' (model \'{model.name}\')')
        if not all([inp in step_args for inp in model.inputs]): raise AttributeError(f'Not all inputs of model.inputs can be found in step function arguments (model \'{model.name}\')')

    def add_model(self, model, watch_values=[]):
        '''Adds a model to the simulation.

        Arguments:
        model -- User defined models need to follow the following example:  
            class ExampleModel():
                inputs = ['value_in']
                outputs = ['value_out']

                def __init__(self, name):
                    self.name = name
                    # init parameters here

                def step(self, time, value_in) -> dict:
                    # model logic here
                    value_out = value_in + 1
                    return {'value_out': value_out}

        watch_values -- adds a models input or output to the reccorded values (pandas.DataFrame). Only watch numeric values that can safely be cast into a pandas DataFrame 
        '''
        self._validate_model(model)

        # check for dupicate models
        if model.name in self._model_names: raise SimulationError(f'Model.name \'{model.name}\' not unique, which is required for the simulation to work')
        self._model_names.append(model.name)
        
        # add model outputs to the simulation output dict 
        self._outputs.update({model.name +  '_' + output_name: np.nan for output_name in model.outputs})
        # add model to the execution graph
        self._G.add_node(model)
        
        # Add extra attributes to the model
        model._sim_input_map =  {}  # input_map maps a models input to the simulation outputs dict {attribute_in: attribute_out}, gets filled in 'connect'
        self._add_watch_values_to_model(model, watch_values)

    def _add_watch_values_to_model(self, model, watch_values):
        '''Adds values to \'_sim_watch_input/output\' attribute of model'''
        if not hasattr(model, '_sim_watch_inputs'):
            model._sim_watch_inputs = []
        if not hasattr(model, '_sim_watch_outputs'):
            model._sim_watch_outputs = []

        for wv in watch_values:
            input_or_output = False
            if wv in model.inputs:
                model._sim_watch_inputs.append(wv)
                input_or_output = True 
                self.dataframe_index_structure.append([model.name, 'inputs', wv]) # Add to MultiIndex of DataFrame
            if wv in model.outputs:
                input_or_output = True
                model._sim_watch_outputs.append(wv)
                self.dataframe_index_structure.append([model.name, 'outputs', wv]) # Add to MultiIndex of DataFrame
            if not input_or_output: raise SimulationError(f'Non existant input or output {wv} cannot be watched')

    def connect(self, model1, model2, *connections, time_shifted=False, init_values=None):
        '''Connects attribute(s) of model1 to model2
        
        Arguments:
        model1 -- model of which the output is connected
        model2 -- model of which the input is provided
        *connections:str or tuple -- str 'atr' or tuple (attr_o, attr_i) specifying input and output attribute of the models that need to be connected.
        time_shifted:bool -- To prevent computational loops, time_shifted connections take the input provided at the last step
        init_values:dict -- Time shifted connections need an initial value for the outputs of model1 to function in the first step
        '''
        for connection in connections:
            if type(connection) == str: # Input and output have the same 'name', only a string can be passed
                connection = (connection, connection)
            attribute_out, attribute_in = connection

            if attribute_in in model2._sim_input_map:
                raise SimulationError(f'Trying to set multiple inputs for {model2.name} {attribute_in}')
            else:
                model2._sim_input_map.update({attribute_in: attribute_out})
            
            if time_shifted: # initialize time shifted connection (start value)
                if not init_values or not attribute_out in init_values: 
                    raise ValueError(f'Time shifted connection requires init_value for \'{attribute_out}\' e.g.: {{\'{attribute_out}\': value}}')
                self._outputs.update({attribute_out: init_values[attribute_out]})

            self._G.add_edge(model1, model2, name=connection, time_shifted=time_shifted)

    def compute_execution_order_from_graph(self, G):
        '''Computes the models execution order from the initialized nx.graph'''
        
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
        return list(nx.topological_sort(G_direction_check)) # sorted_model_execution_list
    
    def _check_if_model_inputs_connected(self, model):
        '''Check if all inputs of the model are provided with an input'''
        if not all([inp in model._sim_input_map for inp in model.inputs]): raise SimulationError(f'Not all inputs of model \'{model.name}\' are provided')
    
    def run(self, datetimes):
        '''Runs the simulation on the index datetimes
        
        Arguments:
        datetimes -- pd.DatetimeIndex specifying the simulation interval and steps.
        '''
        self.log.info('Preparing simulation...')
        sorted_model_execution_list  = self.compute_execution_order_from_graph(self._G)

        for m in sorted_model_execution_list: self._check_if_model_inputs_connected(m)

        self.log.info('Running simulation')
        delta_index  = datetimes.freq.delta.seconds
        n_steps = len(datetimes)

        # date collection should be moved to its own place but remains here now for debugging purposes
        # np. arrays should be used, as they are several times faster. (cast to df after simulation)
        # considder chunk processing for scale up
        # find nice solution for different deltas
        # data_array = np.full((len(datetimes), len(self.output_names)), np.nan, dtype=float) # somehow does not work
        self.df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(self.dataframe_index_structure, names=['model', 'i/o', 'attribute']))

        # setup lists to perform check, which model needs to be steped
        deltas = [pd.Timedelta(m.delta_t, 'sec') for m in sorted_model_execution_list]
        prev_time = [pd.to_datetime('1990-01-01 00:00:00+01:00') for _ in sorted_model_execution_list]

        if not all([d.seconds%delta_index == 0 for d in deltas]):
            raise ValueError('At least one model has a time resolution (delta), that is not a multiple of the frequency of the datetimes provided')
        
        five_percent = int(n_steps/20)

        try:
            pbar = tqdm.tqdm(total=n_steps, desc='Progress', unit='Steps', maxinterval=60)

            for s, time in enumerate(datetimes):
                pbar.update(1) # update progress bar (for use in console)
                # write progress to logger (when running in background)
                if s % five_percent == 0: self.log.info(f'Progress: {int(s/five_percent*5): 3}%')

                # determine which models need to be steped in this time step (order remains the same)
                step_models = [[i, m] for i, [m, p, d] in enumerate(zip(sorted_model_execution_list, prev_time, deltas)) if time-p >= d]
                
                # step models
                for i, model in step_models:
                    # save current time as last execution time of model 
                    prev_time[i] = time
                    
                    # get inputs of the model from outputs of the simulation
                    model_inputs = {input_key: self._outputs[output_key] for input_key, output_key in model._sim_input_map.items()}                   
                    # calculate next model step
                    model_outputs = copy(model.step(time, **model_inputs)) # copy for safety
                    self._outputs.update(model_outputs) # write model outputs to the ouput dict!!

                    # logg data
                    if model._sim_watch_inputs:  # if inputs to watch
                        self.df.loc[time, (model.name, 'inputs', slice(None))] = [model_inputs[wi] for wi in model._sim_watch_inputs]
                    if model._sim_watch_outputs: # if outputs to watch
                        self.df.loc[time, (model.name, 'outputs', slice(None))] = [model_outputs[wo] for wo in model._sim_watch_outputs]
                
        except Exception as e:
            self.log.error(f'Error occured at sim-time \'{time}\' and during the processing of model \'{model.name}\'', exc_info=e)
            raise SimulationError(f'Error occured at sim-time \'{time}\' and during the processing of model \'{model.name}\'') from e
        finally:
            if self.output_to_file and not self.df.empty:
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