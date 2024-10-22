import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import logging
import tqdm 
from tqdm.contrib.logging import logging_redirect_tqdm

from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.cm as cm


class Simulation():
    def __init__(self) -> None:
        self.models = []  # initialize lists with models, state_name and output_names. those get filled in add_model
        self.state_names = []
        self.output_names = []

        # set up a logger 
        self.log_file_name = 'simulation.log'
        logging.basicConfig(filename=self.log_file_name, encoding='utf-8', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        self.log = logging.getLogger(__name__)
        logging_redirect_tqdm(self.log)

        self.state_list = []  # this will be casted to a numpy array in prepare 
        self.output_list = []  # this will be casted to a numpy array in prepare 

        self.G = nx.DiGraph()  # initalize execution graph

        self._is_prepared = False
    

    def add_model(self, model):
        # TODO: check for dupicates!
        self.models.append(model)
        model.state_selector = []  # maybe seperate this from the actual model (create struct or dict for the models sim data)
        init_states = model.init()
        for state_name, init_state in zip(model.states, init_states):
                self.state_names.append(model.name + '_' + state_name)
                model.state_selector.append(self.state_names.index(model.name + '_' + state_name))
                self.state_list.append(init_state)

        model.output_selector = []
        for output_name in model.outputs:
            self.output_names.append(model.name +  '_' + output_name)
            model.output_selector.append(self.output_names.index(model.name + '_' + output_name))
            self.output_list.append(np.nan)

        self.G.add_node(model)


    def connect(self, model1, model2, *connections, time_shifted=False, init_values=None):
        if not hasattr(model2, 'input_selector'):
            model2.input_selector =  [np.nan]*len(model2.inputs)  # 

        for connection in connections:
            if type(connection) == str:
                connection = (connection, connection)
            attribute_out, attribute_in = connection
            
            model_attribute_index = model2.inputs.index(attribute_in)
            world_output_index = self.output_names.index(model1.name + '_' + attribute_out)

            if model2.input_selector[model_attribute_index] is np.nan or model2.input_selector[model_attribute_index] is world_output_index:
                model2.input_selector[model_attribute_index] = world_output_index
            else:
                raise ValueError(f'Trying to set multiple inputs for {model2.name} {attribute_in}')
            
            if time_shifted: # initialize time shifted connection (start value)
                self.output_list[self.output_names.index(model1.name + '_' + attribute_out)] = init_values[attribute_out]


            self.G.add_edge(model1, model2, name=connection, time_shifted=time_shifted)


    def prepare(self):
        self.log.info('Preparing...')

        # prepare simulation state and output vector
        self.state = np.array(self.state_list)
        self.output = np.array(self.output_list)

        # Prepare execution graph
        edges_same_time    = [(u,v) for u, v, d in self.G.edges(data=True) if not d['time_shifted']]
        edges_time_shifted = [(u,v) for u, v, d in self.G.edges(data=True) if d['time_shifted']]

        self.G_current = self.G.edge_subgraph(edges_same_time)
        self.G_shifted = self.G.edge_subgraph(edges_time_shifted)


        if not nx.is_directed_acyclic_graph(self.G_current):
            raise ValueError('There must be a cycle in the connections')
        
        # Reverse time shifted edges, to ensue, that models containing input for another model in the next step are steped last.
        # The concept has to be double checked!!!!!!!!!!!!!
        self.G_direction_check = self.G_current.copy()
        self.G_direction_check.add_edges_from([(v, u) for u, v in self.G_shifted.edges])

        if not nx.is_directed_acyclic_graph(self.G_direction_check):
            raise ValueError('There must be a problem with the time-shifted connections')
        
        # determine model execution order
        self.model_execution_list = list(nx.topological_sort(self.G_direction_check))

        self._is_prepared = True


    def draw_exec_graph(self):
        plt.figure()
        pos=nx.spring_layout(self.G)

        node_labels = {m:m.name for m in self.G.nodes}    

        nx.draw_networkx(self.G_shifted, pos, edge_color="red", connectionstyle='arc3, rad = 0.1', labels=node_labels)
        nx.draw_networkx(self.G_current, pos, edge_color="green", connectionstyle='arc3, rad = 0.1', labels=node_labels)

        plt.show()


    def draw_execution_visualization(self):
        if not self._is_prepared:
            raise ValueError('Simulation needs to be prepared to be able to plot visualization (call prepare)')

        deltas = [m.delta_t for m in self.model_execution_list]  # duplicate!!! maybe move to prepare
        prev_time = [-60*60 for _ in self.model_execution_list]

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # datetimes = pd.date_range('2021-01-01 00:00:00 +01:00', pd.to_datetime('2021-01-03 00:00:00 +01:00') + pd.Timedelta(max(deltas)*2), freq=min(deltas), tz='UTC+01:00')

        # fig.set_size_inches(6,6)

        # position in the plot for initialization dot
        x_pos_init = -min(deltas)
        y_pos_init = (len(self.model_execution_list)-1)/2

        pos_of_output = np.array([[0, 0] if l == np.nan else [x_pos_init, y_pos_init] for l in self.output_list]) # Positions of the output in the plot for creating the arrows


        cmap = cm.get_cmap('viridis', lut=len(self.model_execution_list))
        clist = [cmap(i) for i in range(len(self.model_execution_list))]

        spacing = 5

        times = np.arange(0, max(deltas)*2, min(deltas))

        for time in times:
                
            # determine which models need to be steped in this time step 
            step_models = [[i, m] for i, [m, p, d] in enumerate(zip(self.model_execution_list, prev_time, deltas)) if time-p >= d]
            
            # step models
            for i, model in step_models:
                # save current time as last execution time of model 
                prev_time[i] = time

                # Box for the total delta t of the model
                ax.add_patch( FancyBboxPatch((time-spacing/2, -i-0.4),
                    model.delta_t-spacing, 0.8,
                    boxstyle="round, pad=0.40, rounding_size=5",  #  ",
                    alpha=0.5,
                    ec="none", fc=clist[i],
                    mutation_aspect=1/60
                    ))
                # Box for the calculation "time"
                ax.add_patch( FancyBboxPatch((time-spacing/2, -i-0.4),
                    spacing, 0.8,
                    boxstyle="round, pad=0.40, rounding_size=5",  #  rounding_size=0.15",
                    ec="none", fc=clist[i],
                    mutation_aspect=1/60
                    ))

                # plot arrows from the output positions to the current model
                i_inp = getattr(model, 'input_selector', 'no_inp')
                if i_inp != 'no_inp':
                    pos_of_output_i = pos_of_output[i_inp, :]
                    for xin, yin in pos_of_output_i:
                        ls = '-' if time == xin else '--'
                        arrow_c = clist[int(yin)] if xin >= 0 else 'k'
                        ax.annotate("", xy=(time, -i), xytext=(xin, -yin), arrowprops=dict(arrowstyle='-|>', linestyle=ls, color=arrow_c, linewidth=2))  # arrowstyle='-|>', , headwidth=5

                
                pos_of_output[model.output_selector, :] = np.array([[time, i] for _ in model.output_selector])

        ax.plot([x_pos_init], [-y_pos_init], marker = 'o', markersize=10, color='k')

        ax.set_xticks(times)
    
        ax.set_yticks([-i for i in range(len(self.model_execution_list))])
        ax.set_yticklabels([f'{i+1:03} {m.name:15}' for i, m in enumerate(self.model_execution_list)])

        ax.autoscale()  
        # fig.show()

        


    # def add_data_logging(self, model, *outputs):
    #     # self.data_logging_setup = {delta: (attr_name1, attr_name2, ...)}
    #     # This dict is later used to create np.arrays to store the data
    #     if model.delta_t in self.data_logging_setup:
    #         self.data_logging_setup[model.delta_t].append([model.name + '_' + o for o in outputs])
    #     else:
    #         self.data_logging_setup[model.delta_t] = [model.name + '_' + o for o in outputs]


    def run(self, datetimes):
        self.log.info('Running simulation')
        # times = pd.date_range('2021-01-01 00:00:00', '2021-01-03 00:00:00', freq='1min', tz='UTC+01:00')

        delta_index  = datetimes.freq.delta.seconds
        n_steps = len(datetimes)

        # date collection should be moved to its own place but remains here now for debugging purposes
        # np. arrays should be used, as they are several times faster. (cast to df after simulation)
        # considder chunk processing for scale up
        # find nice solution for different deltas
        self.df = pd.DataFrame([], index=datetimes, columns=self.output_names, dtype=float)
        # data_array = np.full((len(datetimes), len(self.output_names)), np.nan, dtype=float) # somehow does not work

        # setup lists to perform check, which model needs to be steped
        deltas = [pd.Timedelta(m.delta_t, 'sec') for m in self.model_execution_list]
        prev_time = [pd.to_datetime('1990-01-01 00:00:00+01:00') for _ in self.model_execution_list]

        if not all([d.seconds%delta_index == 0 for d in deltas]):
            raise ValueError('At least one model has a time resolution (delta), that is not a multiple of the frequency of the datetimes you provided')
        
        five_percent = int(n_steps/20)

        try:
            pbar = tqdm.tqdm(total=n_steps, desc='Progress', unit='Steps', maxinterval=60)

            for s, time in enumerate(datetimes):
                pbar.update(1)
                if s % five_percent == 0:
                    self.log.info(f'Progress: {int(s/five_percent*5): 3}%')

                # determine which models need to be steped in this time step 
                step_models = [[i, m] for i, [m, p, d] in enumerate(zip(self.model_execution_list, prev_time, deltas)) if time-p >= d]
                
                # step models
                for i, model in step_models:
                    # save current time as last execution time of model 
                    prev_time[i] = time

                    # get inputs of the model from output vector of the simulation
                    i_inp = getattr(model, 'input_selector', 0) # maybe change default below
                    output_i = self.output[i_inp]
                    
                    # get the previous states of the model from the state vector of the simulation
                    i_state = getattr(model, 'state_selector', 0)
                    state_i = self.state[i_state]
                    
                    # calculate next model step 
                    self.state[model.state_selector], self.output[model.output_selector] = model.step(time, state_i, output_i)

                    # logg data
                    # data_array[i][model.output_selector] = self.output[model.output_selector]
                    self.df.loc[time, [model.name + '_' + o for o in model.outputs]] = self.output[model.output_selector]
        finally:
            # self.df = pd.DataFrame(data_array, index=datetimes, columns=self.output_names, dtype=float) 
            self.df.to_pickle('output_data.pkl')
            # self.df.to_pickle('own_implementation/output_data_1.pkl')
            
                    