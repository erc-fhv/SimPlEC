import mosaik_api
import pandas as pd


META = {
    'type': 'time-based',
    'models': {
        'Clock': {
            'public': True,
            'any_inputs': True,
            'params': ['startdatetime'],
            'attrs': ['datetime'],
        },
    },
}


class Clock(mosaik_api.Simulator):
    Δ_T = 60
    def __init__(self):
        super().__init__(META)
        self.time = 0
        self.eid = None
        self.startdatetime = 0
        self.datetime = 0

    def init(self, sid, time_resolution):
        if float(time_resolution) != self.Δ_T:
            raise ValueError(f'BuildingSim only supports time_resolution={self.Δ_T}., but'
                             ' %s was set.' % time_resolution)
        return self.meta
    
    def create(self, num, model, startdatetime='2022-01-01 00:00:00+01:00'):
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of Clock.')

        self.eid = 'Clock'
        self.startdatetime = pd.to_datetime(startdatetime)
        return [{'eid': self.eid, 'type': model}]

    def step(self, time, inputs, max_advance):
        self.time = time
        self.datetime = self.startdatetime + pd.Timedelta(time, 'sec')

        return time + self.Δ_T  # None  # Step size is 1 second
    
    def get_data(self, outputs):
        # print('Clock speaking: ', outputs)
        data = {}
        for eid, attrs in outputs.items():
            data['time'] = self.time
            data[eid] = {}
            for attr in attrs:
                if attr != 'datetime':
                    raise ValueError('Unknown output attribute "%s"' % attr)
                
                data[eid][attr] = getattr(self, attr)

        # print('Clock speaking: ', data)
        return data
    

def main():
    return mosaik_api.start_simulation(Clock())


if __name__ == '__main__':
    main()

