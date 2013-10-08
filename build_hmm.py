"""
Author: Vaibhav Bhandari <vb2317@columbia.edu>

===============================

"""

from os import path
from abc import ABCMeta,abstractmethod
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import copy
from sklearn.hmm import GaussianHMM
from load_data import read_weather_data,read_steam_data,read_db

class HMM(object):
    '''
    class for creating and manipulating HMM model
    '''
    def __init__(self,**kwargs):
        if 'steam_obj' not in kwargs:
            self.steam_obj = Steam()
        else:
            self.steam_obj = kwargs['steam_obj']
        if 'weather_obj' not in kwargs:
            self.weather_obj = Weather()
        else:
            self.weather_obj = kwargs['weather_obj']
        steam_obj = self.steam_obj
        weather_obj = self.weather_obj
        hour_of_day = steam_obj.ts.index.map(lambda x: x.hour + (x.minute/60.0))
        day_of_week = steam_obj.ts.index.map(lambda x: x.dayofweek)
        df_hmm = pd.DataFrame({'steam':steam_obj.ts,'weather':weather_obj.ts, \
                'hour_of_day':hour_of_day,'day_of_week':day_of_week},index=steam_obj.ts.index)
        #its imp that the order for columns is maintain 
        #while slicing the HMM model 
        self.df_hmm,self.X_hmm = self.gen_meta_data(steam_obj,weather_obj) 
        if 'n_states' not in kwargs:
            self.plot_elbow(3,15)
        else:
            self.n_states = kwargs['n_states']

    def __len__(self):
        return len(self.X_hmm)

    def build_model(self):
        n_states = self.n_states
        X_hmm = self.X_hmm
        self.model = GaussianHMM(n_states,covariance_type='diag',n_iter=1000)
        self.model.fit([X_hmm])
        self.hidden_states = self.model.predict(X_hmm)

    def build_forecast_model(self):
        model = self.model
        n_states = self.n_states
        model_forecast = copy.deepcopy(model)
        model_forecast.n_features = model.n_features-1
        model_forecast._means_ = model.means_[:,1:]
        model_forecast._covars_ = model._covars_[:,1:]
        self.model_forecast = model_forecast

    def gen_meta_data(self,steam_obj=None,weather_obj=None):
        if steam_obj!=None:
            hour_of_day = steam_obj.ts.index.map(lambda x: x.hour + (x.minute/60.0))
            day_of_week = steam_obj.ts.index.map(lambda x: x.dayofweek)           
            df_hmm = pd.DataFrame({'steam':steam_obj.ts,'weather':weather_obj.ts, \
                        'hour_of_day':hour_of_day},index=steam_obj.ts.index)
            #df_hmm = pd.DataFrame({'steam':steam_obj.ts,'weather':weather_obj.ts, \
            #            'hour_of_day':hour_of_day,'day_of_week':day_of_week},index=steam_obj.ts.index)
           # X_hmm = df_hmm.as_matrix(columns=['steam','weather'])
            X_hmm = df_hmm.as_matrix(columns=['steam','weather','hour_of_day'])
            #X_hmm = df_hmm.as_matrix(columns=['steam','weather','hour_of_day','day_of_week'])
        else:
            hour_of_day = weather_obj.ts.index.map(lambda x: x.hour + (x.minute/60.0))
            day_of_week = weather_obj.ts.index.map(lambda x: x.dayofweek)           
            df_hmm = pd.DataFrame({'weather':weather_obj.ts, \
                    'hour_of_day':hour_of_day},index=weather_obj.ts.index)
            #df_hmm = pd.DataFrame({'weather':weather_obj.ts, \
            #        'hour_of_day':hour_of_day,'day_of_week':day_of_week},index=weather_obj.ts.index)
           # X_hmm = df_hmm.as_matrix(columns=['weather'])
            X_hmm = df_hmm.as_matrix(columns=['weather','hour_of_day'])
            #X_hmm = df_hmm.as_matrix(columns=['weather','hour_of_day','day_of_week'])
        return df_hmm,X_hmm

    def plot_model(self,x_ax=None,y_ax=None):
        X_hmm = self.X_hmm
        steam_ts = self.steam_obj.ts
        if x_ax == None:
            x_ax = np.asarray([item.to_datetime() for item in steam_ts.index])
        if y_ax == None:
            y_ax = X_hmm[:,0]
        hidden_states = self.hidden_states
        n_states = self.n_states
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in xrange(n_states):
            print i
            idx = (hidden_states==i)
            if i<7:
                ax.plot(x_ax[idx],y_ax[idx],'o',label='%dth state'%i)
            elif i<14:
                ax.plot(x_ax[idx],y_ax[idx],'x',label='%dth state'%i)
            elif i<21:
                ax.plot(x_ax[idx],y_ax[idx],'+',label='%dth state'%i)
            elif i<28:
                ax.plot(x_ax[idx],y_ax[idx],'*',label='%dth state'%i)
        ax.set_title('%d State HMM'%(n_states))
        ax.legend()
        ax.set_ylabel('Load (Mlb/Hr)')
        ax.set_xlabel('Time')
        ax.grid(True)
        plt.show()


    def plot_elbow(self,start,end):
        '''
        Fit GMM and plot elbow using AIC & BIC
        '''
        from sklearn.mixture import GMM,DPGMM
        obs = self.X_hmm
        aics = []
        bics = []
        for i in range(start,end+1):
            n_iter=1000
            for j in range(1,11):
                g = GMM(n_components=i,n_iter=n_iter)
                g.fit(obs)
                print i
                converged =  g.converged_
                if converged:
                    print 'j:%d'%(j)
                    break
                n_iter += 1000
            aics.append(g.aic(obs))
            bics.append(g.bic(obs))
        if not converged:
            print 'Not Converged!!'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(start,end+1),aics,label='AIC')
        ax.plot(range(start,end+1),bics,label='BIC')
        ax.set_xlabel("No. of Clusters")
        ax.set_ylabel("Information Loss")
        ax.set_xticks(range(start,end+1),minor=True)
        ax.legend()
        ax.grid(True,which='both')
        plt.show()

class Data(object):

    def __init__(self,ts=None):
        if len(ts):
            self.ts = ts
        else:
            pass

    def gen_df(self):
        ts = self.ts
        hr = pd.Series(ts.index.map(lambda x: x.hour+(float(x.minute)/60)),index=ts.index)
        dod = pd.Series(ts.index.map(lambda x: int('%d%03d'%(x.year,x.dayofyear))),index=ts.index)
        df = pd.concat([ts,dod,hr],axis=1)
        df.columns=['ts','dod','hr']
        df = df.drop_duplicates(cols=['hr','dod'],take_last=True)
        df = df.pivot(columns='dod',index='hr',values='ts')
        self.df = df

class Steam(Data):
    
    def __init__(self,from_date=None,to_date=dt.date.today(),**keywords):
        self.READ_FROM_DB = False
        self.from_date = from_date
        self.to_date = to_date
        self.data = read_steam_data(**keywords) #read raw data from db
        #self.data = self.read_data()
        self.preprocess_data()
        self.gen_df()

    def __len__(self):
        return len(self.ts)
    
    def read_data(self):
        return self.read_db() if self.READ_FROM_DB else self.read_file()

    def read_db(self):
        pass

    def read_file(self):
        data_dir = path.join(path.dirname(__file__),'../Data')
        return pd.load('%s/ts_data'%(data_dir))

    def preprocess_data(self):
        data = self.data
        from_date = self.from_date
        to_date = self.to_date
        steam_load = [item[1] for item in data]
        steam_dates = [item[0].replace(second=0,microsecond=0) for item in data]
        steam_df = pd.DataFrame({'load':steam_load,'dates':steam_dates})
        steam_df = steam_df.drop_duplicates(cols='dates',take_last=True)
        ts = pd.TimeSeries(steam_df['load'].tolist(),index=steam_df['dates'])
        dates = pd.DateRange(min(steam_dates),max(steam_dates),offset=pd.DateOffset(minutes=15))
        ts = ts.reindex(dates)
        #TODO Change this to expolation
        self.ts = ts.interpolate() 
        #ts[pd.isnull(ts)] should return empty list
        self.ts=self.ts[from_date:to_date][:-1]

    def gen_df(self):
        super(Steam,self).gen_df()

class Weather(Data):

    def __init__(self,from_date=None,to_date=dt.date.today(),**keywords):
        self.READ_FROM_DB = False
        self.from_date = from_date
        self.to_date = to_date
        self.read_data(**keywords)
        self.preprocess_data()
        self.gen_df()

    def __len__(self):
        return len(self.ts)
    
    def read_data(self,**keywords):
        self.data = read_weather_data(**keywords) #read raw data from db
        #self.data = self.read_db() if self.READ_FROM_DB else self.read_file()

    def read_db(self):
        pass

    def read_file(self):
        data_dir = path.join(path.dirname(__file__),'../Data')
        return pd.load('%s/ts_data'%(data_dir))
    
    def preprocess_data(self):
        data = self.data
        from_date = self.from_date
        to_date = self.to_date
        temps = [float(item[1]) for item in data]
        dewps = [float(item[2]) for item in data]
        db_dates = [item[0].replace(second=0,microsecond=0) for item in data]
        weather_df = pd.DataFrame({'temps':temps,'dates':db_dates,'dewps':dewps})
        weather_df = weather_df.drop_duplicates(cols='dates',take_last=True)
        dewps_in_K = 273.16 + (weather_df['dewps'] - 32)/1.8
        temps_in_C = (weather_df['temps'] - 32)/1.8
        humidex = temps_in_C + 0.5555*(6.11 *(np.exp(5417.7530*(1/273.16-1/dewps_in_K))) - 10)
    #    weather_ts = pd.TimeSeries(weather_df['temps' ].tolist(),index=weather_df['dates'])
        weather_ts = pd.TimeSeries(humidex.tolist(),index=weather_df['dates']) 
        dates = pd.DateRange(min(db_dates),max(db_dates),offset=pd.DateOffset(minutes=15))
        weather_ts = weather_ts.reindex(dates)
        #TODO Change this to expolation
        weather_ts[weather_ts>200] = None
        weather_ts = weather_ts.interpolate() 
        #TODO change the method to interpolation
        weather_ts = weather_ts.tshift(-6,freq='Min')
        self.ts = weather_ts
        #ts[pd.isnull(ts)] should return empty list
        self.ts = self.ts[from_date:to_date][:-1]

    def gen_df(self):
        super(Weather,self).gen_df()
