import mosaik_api

import models.weather_model as weather_model

META = {
    'type': 'time-based',
    'models': {
        'LocalWeather': {
            'public': True,
            'params': ['latitude', 'longitude'],
            'attrs': ['T_air', 'datetime', 'dni', 'dhi', 'ghi', 'solar_pos'],
        },
    },
}


class WeatherSim(mosaik_api.Simulator):
    Δ_T = 60*60 # sec
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'Weather_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.time = 0

    def init(self, sid, time_resolution, eid_prefix=None):
        # if float(time_resolution) != self.Δ_T:
        #     raise ValueError(f'WeatherSim only supports time_resolution={self.Δ_T}., but'
        #                      ' %s was set.' % time_resolution)
        if eid_prefix is not None:
            self.eid_prefix = eid_prefix
        return self.meta
    
    def create(self, num, model):
        next_eid = len(self.entities)
        entities = []

        for i in range(next_eid, next_eid + num):
            model_instance = weather_model.LocalWeather()
            eid = '%s%d' % (self.eid_prefix, i)
            self.entities[eid] = model_instance
            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        self.time = time
        # do step for each model instance:
        for eid, model_instance in self.entities.items():
            if eid in inputs:
                model_instance.datetime = list(inputs[eid]['datetime'].values())[0]

            model_instance.step()

        return time + self.Δ_T  # Step size is 1 second

    def get_data(self, outputs):
        data = {}
        # print('weather speaking: ', outputs)
        for eid, attrs in outputs.items():
            model = self.entities[eid]
            data['time'] = self.time
            data[eid] = {}
            for attr in attrs:
                if attr not in self.meta['models']['LocalWeather']['attrs']:
                    raise ValueError('Unknown output attribute: %s for LocalWeather' % attr)
                data[eid][attr] = model.data[attr]
        # print('weather speaking: ', model.data)

        return data
    

def main():
    return mosaik_api.start_simulation(WeatherSim())


if __name__ == '__main__':
    main()

