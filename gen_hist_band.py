__author__ = 'Vaibhav Bhandari <vb2317@columbia.edu>'
__module__= 'Historical Band'

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from build_hmm import Weather
from build_hmm import Steam
from build_hmm import HMM
from build_hmm import Data

DEBUG=True

class HistBand(object):
    '''
    class to generate historical band
    '''
    def __init__(self,n_states,from_date=dt.date(2012,2,1),to_date=dt.date(2013,7,9),forecast_time_delta=1,hmm_obj=None):
        steam_obj = Steam(from_date,to_date,filename='../Data/RUDINSERVER_CURRENT_STEAM_DEMAND_FX70_AUG.csv')
        weather_obj = Weather(from_date,to_date,filename='../Data/OBSERVED_WEATHER_AUG.csv')
        wobj_forecast = Weather(to_date,to_date+dt.timedelta(forecast_time_delta),filename='../Data/OBSERVED_WEATHER_AUG.csv')
        #ONLY FOR DEBUG
        if DEBUG:
            sobj_forecast = Steam(to_date,to_date+dt.timedelta(forecast_time_delta),filename='../Data/RUDINSERVER_CURRENT_STEAM_DEMAND_FX70_AUG.csv')
        if hmm_obj==None:
            hmm_obj = HMM(steam_obj=steam_obj,weather_obj=weather_obj,n_states=n_states)
            hmm_obj.build_model()
            hmm_obj.build_forecast_model() 
        #print get_similar_weather_dates(steam_obj.df,weather_obj.df,wobj_forecast.ts)
        df_weather_forecast, X_weather_forecast= hmm_obj.gen_meta_data(weather_obj=wobj_forecast)
        if DEBUG:
            df_steam_forecast, X_steam_forecast= hmm_obj.gen_meta_data(steam_obj=sobj_forecast,weather_obj=wobj_forecast)
            states_forecast_steam = hmm_obj.model.predict(X_steam_forecast)
        states_forecast = hmm_obj.model_forecast.predict(X_weather_forecast)
        self.from_date = from_date
        self.to_date = to_date
        self.steam_obj = steam_obj
        self.weather_obj = weather_obj
        self.wobj_forecast = wobj_forecast
        self.sobj_forecast = sobj_forecast
        self.hmm_obj = hmm_obj
        self.states_forecast = states_forecast
        self.states_forecast_steam = states_forecast_steam

    def gen_hist_band(self):
        hmm_obj = self.hmm_obj
        steam_obj = self.steam_obj
        weather_obj = self.weather_obj
        wobj_forecast = self.wobj_forecast
        states_forecast = self.states_forecast
        to_date_str = dt.datetime.strftime(self.to_date,'%m/%d/%Y')
        DEBUG=False
        if DEBUG:
            states_forecast = self.states_forecast_steam
        states_forecast = pd.Series(states_forecast,index=np.arange(0,24,0.25))
        states_ts = pd.Series(hmm_obj.hidden_states,index=steam_obj.ts.index)
        states_obj = Data(states_ts)
        states_obj.gen_df()
        states_df = states_obj.df
        similar_state_dates = self.get_similar_states_dates(states_df,states_forecast,start=6,end=17)
        weather_forecast = pd.Series(wobj_forecast.ts.values,index=np.arange(0,24,0.25))
        similar_weather_dates = self.get_similar_weather_dates(steam_obj.df,weather_obj.df,weather_forecast,start=4,end=15)
        fig = plt.figure()
        ax_steam = fig.add_subplot(211)
        ax_steam.set_xlabel('Time Of Day ')
        ax_steam.set_ylabel('Steam (Mlb/Hr)')
        ax_weather = fig.add_subplot(212)
        ax_weather.set_xlabel('Time Of Day')
        ax_weather.set_ylabel('Humidex (C)')
        steam_obj.df[similar_state_dates].plot(grid=True,title='HISTORICAL STEAM LOAD BASED ON LATENT STATES (DATE:%s)'%(to_date_str),ax=ax_steam)
        ax_steam.legend([dt.datetime.strftime(dt.datetime.strptime(item,'%Y%j'),'%m/%d/%Y') for item in map(str,similar_state_dates)])
        weather_obj.df[similar_state_dates].plot(grid=True,title='HISTORICAL WEATHER BASED ON LATENT STATES (DATE:%s)'%(to_date_str),ax=ax_weather)
        ax_weather.legend([dt.datetime.strftime(dt.datetime.strptime(item,'%Y%j'),'%m/%d/%Y') for item in map(str,similar_state_dates)])
        fig = plt.figure()
        ax_steam = fig.add_subplot(211)
        ax_steam.set_xlabel('Time Of Day ')
        ax_steam.set_ylabel('Steam (Mlb/Hr)')
        ax_weather = fig.add_subplot(212)
        ax_weather.set_xlabel('Time Of Day')
        ax_weather.set_ylabel('Humidex (C)')
        steam_obj.df[similar_weather_dates].plot(grid=True,title='HISTORICAL STEAM LOAD BASED ON SIMILAR WEATHER (DATE:%s)'%(to_date_str),ax=ax_steam)
        ax_steam.legend([dt.datetime.strftime(dt.datetime.strptime(item,'%Y%j'),'%m/%d/%Y') for item in map(str,similar_weather_dates)])
        weather_obj.df[similar_weather_dates].plot(grid=True,title='HISTORICAL WEATHER BASED ON SIMILAR WEATHER (DATE:%s)'%(to_date_str),ax=ax_weather)
        ax_weather.legend([dt.datetime.strftime(dt.datetime.strptime(item,'%Y%j'),'%m/%d/%Y') for item in map(str,similar_weather_dates)])
        #TEST MEDIAN
        plot_data = np.median(steam_obj.df[similar_weather_dates],axis=1)
        dev = hmm_obj.model._covars_[:,0][states_forecast]**0.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_ax = np.arange(0,24,0.25)
        ax.plot(x_ax,plot_data+dev,'--',color='green')
        ax.plot(x_ax,plot_data-dev,'--',color='red')
        if self.sobj_forecast:
            ax.plot(x_ax,self.sobj_forecast.ts.values,color='blue')
        ax.set_title('HISTORICAL BAND USING LATENT STATES (DATE:%s)'%to_date_str)
        ax.set_ylabel('Steam (Mlb/Hr)')
        ax.set_xlabel('Time Of Day')
        ax.grid(True)
        plot_data = np.median(steam_obj.df[similar_state_dates],axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_ax,plot_data+dev,'--',color='green')
        ax.plot(x_ax,plot_data-dev,'--',color='red')
        if self.sobj_forecast:
            ax.plot(x_ax,self.sobj_forecast.ts.values,color='blue')
        ax.set_title('HISTORICAL BAND BASED ON SIMILAR DAYS (DATE:%s)'%dt.datetime.strftime(self.to_date,'%m/%d/%Y'))
        ax.set_ylabel('Steam (Mlb/Hr)')
        ax.set_xlabel('Time of Day')
        ax.grid(True)
        plt.show()
        self.similar_state_dates = similar_state_dates
        self.similar_weather_dates = similar_weather_dates
        self.states_obj = states_obj
    
    def plot_historical_band(self,x_ax,actual,upper_band,lower_band,title=' '):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_ax,actual,label='Actual')
        ax.plot(x_ax,upper_band,label='Upper_Band')
        ax.plot(x_ax,lower_band,label='Lower_Band')
        ax.set_title('Historical Band using %s'%title)
        ax.legend()
        ax.set_ylabel('Load (Mlb/Hr)')
        ax.set_xlabel('Time')
        ax.grid(True)
        plt.show()

    def get_similar_states_dates(self,wdf,inputts,start=-0.1,end=24.1,num=10):
        '''
        pass
        '''
        #inputts = mydf[2013010]
        weekday_name = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        mywdf = wdf
        wdf = {}
        for x in mywdf.columns:
            if(dt.datetime.strptime(str(int(x)),'%Y%j').weekday() <= 4):
                wdf[x] = mywdf[x]
        mywdf = pd.DataFrame(wdf)
        mywdf = mywdf[(mywdf.index>=start) & (mywdf.index<=end)]
        inputts = inputts[(inputts.index>=start) & (inputts.index<=end)]
        diffdf = mywdf.apply(lambda x:(x==inputts),axis=0).sum()
        diffdf.sort()
        diffdf = diffdf[::-1]
        print diffdf
        dates = diffdf[:num].index
        #print dates
        days = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
        #print days
        #mydf[dates].plot()
        weekdays = [weekday_name[item.weekday()] for item in days ]
        self.days = days
        self.weekdays = weekdays
        return dates

    def get_similar_weather_dates(self,df,wdf,inputts,start=-0.1,end=24.1,num=10):
        '''
        For the given input day, find the most similar weather days
        using L-1 distance
        input_day format: yyyydoy
        '''
        #inputts = mydf[2013010]
        weekday_name = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        mywdf = wdf
        wdf = {}
        for x in mywdf.columns:
            if(dt.datetime.strptime(str(int(x)),'%Y%j').weekday() <= 4):
                wdf[x] = mywdf[x]
        mywdf = pd.DataFrame(wdf)
        mywdf = mywdf[(mywdf.index>=start) & (mywdf.index<=end)]
        inputts = inputts[(inputts.index>=start) & (inputts.index<=end)]
        inputts = inputts.values
        diffdf = mywdf.apply(lambda x:abs(x-inputts),axis=0).sum()
        diffdf.sort()
        print diffdf
        dates = diffdf[:num].index
        #print dates
        days = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
        #print days
        #mydf[dates].plot()
        weekdays = [weekday_name[item.weekday()] for item in days ]
        return dates

def main():
    pass

if __name__=='__main__':
    main()
