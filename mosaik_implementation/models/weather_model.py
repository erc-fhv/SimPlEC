import pvlib
import pandas as pd


class LocalWeather():
    def __init__(self, latitude  = 47.4868936, longitude = 9.7406211):  # latitude=37.7749, longitude=-122.4194
        # Location information (latitude, longitude, and altitude)
        self.latitude  = latitude
        self.longitude = longitude
        # self.altitude  = altitude

        self.weather_df = pvlib.iotools.get_pvgis_tmy(self.latitude, self.longitude)[0].rename({'temp_air': 'T_air'}, axis='columns')
        self.weather_df.index = pd.date_range('2021-01-01 00:00:00', '2021-12-31 23:00:00', freq='1h', tz='UTC+01:00')

        self.time = 0
        self.datetime = '2021-01-01 00:00:00'
        self.data = self.weather_df.loc[self.datetime, :]

    def step(self):
        solar_position = pvlib.solarposition.get_solarposition(self.datetime, self.latitude, self.longitude)
        self.data = {**dict(self.weather_df.loc[self.datetime, :]), 'solar_pos': dict(solar_position.loc[self.datetime, :])}
        # print('weather model speaking', self.data, {'solar_pos': dict(solar_position.loc[self.datetime, :])})

