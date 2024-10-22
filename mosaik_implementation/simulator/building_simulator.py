import mosaik_api

import models.building_model as building_model

META = {
    'type': 'time-based',
    'models': {
        'BuildingModel': {
            'public': True,
            'params': ['init_T_room1', 'init_T_room2', 'Δ_T', 'UA1', 'UA2', 'UA1_2', 'C1', 'C2', 'A_facade'],
            'attrs': ['T_room1', 'T_room2', 'T_amb', 'dot_Q_sol', 'dot_Q_heat', 'dni', 'dhi', 'ghi','solar_pos'],
        },
    },
}


class BuildingSim(mosaik_api.Simulator):
    Δ_T = 60
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'Model_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.time = 0

    def init(self, sid, time_resolution, eid_prefix=None):
        if float(time_resolution) != self.Δ_T:
            raise ValueError(f'BuildingSim only supports time_resolution={self.Δ_T}., but'
                             ' %s was set.' % time_resolution)
        if eid_prefix is not None:
            self.eid_prefix = eid_prefix
        return self.meta
    
    def create(self, num, model):
        next_eid = len(self.entities)
        entities = []

        for i in range(next_eid, next_eid + num):
            model_instance = building_model.BuildingModel() # enable passing through parameters
            eid = '%s%d' % (self.eid_prefix, i)
            self.entities[eid] = model_instance
            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        self.time = time
        # do step for each model instance:
        for eid, model_instance in self.entities.items():
            if eid in inputs:
                attrs = inputs[eid]
                for attr, values in attrs.items():
                    # print('Model speaking attr, value: ', attr, values)
                    if len(values) != 1:
                        raise ValueError('Can only provide one value as input for each BuildingModel instance')
                    
                    value = list(values.values())[0] # unpack dict # /len(values)
                    # print('Model speaking tamb: ', value)

                    setattr(model_instance, attr, value)
            
            # print('Model speaking tamb: ', model_instance.T_amb)

            model_instance.step()

        return time + self.Δ_T  # Step size is 1 second

    def get_data(self, outputs):
        data = {}
        # print('building speaking: ', outputs)
        for eid, attrs in outputs.items():
            model = self.entities[eid]
            data['time'] = self.time
            data[eid] = {}
            for attr in attrs:
                if attr not in self.meta['models']['BuildingModel']['attrs']:
                    raise ValueError('Unknown output attribute: %s' % attr)

                # Get model.val or model.delta:
                data[eid][attr] = getattr(model, attr)
        # print('building speaking: ', data)

        return data
    

def main():
    return mosaik_api.start_simulation(BuildingSim())


if __name__ == '__main__':
    main()

