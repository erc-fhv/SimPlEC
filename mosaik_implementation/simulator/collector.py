"""
A simple data collector that prints all data when the simulation finishes.

"""
import collections

import mosaik_api

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

import pandas as pd


META = {
    'type': 'event-based',
    'models': {
        'Monitor': {
            'public': True,
            'any_inputs': True,
            'params': [],
            'attrs': [],
        },
    },
}


class Collector(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = None
        self.data = collections.defaultdict(lambda:
                                            collections.defaultdict(dict))

    def init(self, sid, time_resolution):
        return self.meta

    def create(self, num, model):
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of Monitor.')

        self.eid = 'Monitor'
        return [{'eid': self.eid, 'type': model}]

    def step(self, time, inputs, max_advance):
        data = inputs.get(self.eid, {})
        for attr, values in data.items():
            for src, value in values.items():
                self.data[src][attr][time] = value

        return None

    # def finalize(self):
    #     print('Collected data:')
    #     for sim, sim_data in sorted(self.data.items()):
    #         print('- %s:' % sim)
    #         for attr, values in sorted(sim_data.items()):
    #             print('  - %s: %s' % (attr, values))

    # def finalize(self):
    #     for sim, sim_data in sorted(self.data.items()):
    #         fig, ax = plt.subplots(len(sim_data), 1, squeeze=False)
    #         for i, (attr, values) in enumerate(sorted(sim_data.items())):
    #             x, y = zip(*sorted(values.items()))
    #             ax[i, 0].plot(x, y)
    #             ax[i, 0].set_ylabel(attr)
    #             ax[i, 0].set_xlabel('time')
    #         fig.savefig(f'figures/{sim}.png')

    def finalize(self):
        for sim, sim_data in sorted(self.data.items()):
            df = pd.DataFrame.from_dict(sim_data)
            df.to_pickle(f'data/output/df_{sim.replace(".", "_")}.pkl')



if __name__ == '__main__':
    mosaik_api.start_simulation(Collector())