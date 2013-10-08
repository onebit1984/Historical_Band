import pandas as pd
from load_data import read_db
from load_data import read_steam_data
from load_data import read_weather_data

class Weather:
    '''
    This class performs various operations on steam data and creates covariates
    '''
    def __init__(self):
        weather_data = read_weather_data() #read raw data from db
        temps = [float(item[1]) for item in weather_data]
        dewps = [float(item[2]) for item in weather_data]
        db_dates = [item[0].replace(second=0,microsecond=0) for item in weather_data]
        weather_df = pd.DataFrame({'temps':temps,'dates':db_dates,'dewps':dewps})
        weather_df = weather_df.drop_duplicates(cols='dates',take_last=True)
        weather_ts = pd.TimeSeries(weather_df['temps'].tolist(),index=weather_df['dates'])
        dates = pd.DateRange(min(db_dates),max(db_dates),offset=pd.DateOffset(minutes=15))
        weather_ts = weather_ts.reindex(dates)
        #TODO Change this to expolation
        weather_ts = weather_ts.interpolate() 
        weather_ts = weather_ts.tshift(-6,freq='Min')
        #steam_ts[pd.isnull(steam_ts)] should return empty list


class Steam:
    '''
    This class performs various operations on steam data and creates covariates
    '''
    def __init__(self):
        data = read_db() #read raw data from db
        steam_load = [item[4] for item in data]
        steam_dates = [item[1].replace(second=0,microsecond=0) for item in data]
        steam_df = pd.DataFrame({'load':steam_load,'dates':steam_dates})
        steam_df = steam_df.drop_duplicates(cols='dates',take_last=True)
        steam_ts = pd.TimeSeries(steam_df['load'].tolist(),index=steam_df['dates'])
        dates = pd.DateRange(min(steam_dates),max(steam_dates),offset=pd.DateOffset(minutes=15))
        steam_ts = steam_ts.reindex(dates)
        #TODO Change this to expolation
        steam_ts = steam_ts.interpolate() 
        #steam_ts[pd.isnull(steam_ts)] should return empty list


