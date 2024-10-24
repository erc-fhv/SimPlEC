import pvlib
import pandas as pd
import numpy as np


class LocalWeather():
    def __init__(self, name, delta_t=60*60, latitude  = 47.4868936, longitude = 9.7406211):
        # Parameter
        self.delta_t = delta_t

        self.latitude  = latitude
        self.longitude = longitude
        # self.altitude  = altitude

        self._weather_df = pvlib.iotools.get_pvgis_tmy(self.latitude, self.longitude)[0]
        self._weather_df.index = pd.date_range('2021-01-01 00:00:00', '2021-12-31 23:00:00', freq='1h', tz='UTC+01:00')

        self.inputs  = []
        self.outputs = ['T_air', 'dni', 'dhi', 'ghi', 'apparent_zenith', 'azimuth']
        self.name = name

    def step(self, time):
        temp_air, dni, dhi, ghi = self._weather_df.loc[time, ['temp_air', 'dni', 'dhi', 'ghi']].values
        apparent_zenith, azimuth = pvlib.solarposition.get_solarposition(time, self.latitude, self.longitude).loc[time, ['apparent_zenith', 'azimuth']].values
        
        return {'T_air': temp_air, 'dni': dni, 'dhi': dhi, 'ghi': ghi, 'apparent_zenith': apparent_zenith, 'azimuth': azimuth}

