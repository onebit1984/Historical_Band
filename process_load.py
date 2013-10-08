import pandas as pd
from load_data import read_db
from load_data import read_steam_data
from load_data import read_weather_data
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

class Weather:
    '''
    This class performs various operations on weather data and creates covariates
    '''
    def __init__(self,**keywords):
        weather_data = read_weather_data(**keywords) #read raw data from db
        temps = [float(item[1]) for item in weather_data]
        dewps = [float(item[2]) for item in weather_data]
        db_dates = [item[0].replace(second=0,microsecond=0) for item in weather_data]
        weather_df = pd.DataFrame({'temps':temps,'dates':db_dates,'dewps':dewps})
        weather_df = weather_df.drop_duplicates(cols='dates',take_last=True)
        dewps_in_K = 273.16 + (weather_df['dewps'] - 32)/1.8
        temps_in_C = (weather_df['temps'] - 32)/1.8
        humidex = temps_in_C + 0.5555*(6.11 *(np.exp(5417.7530*(1/273.16-1/dewps_in_K))) - 10)
        #weather_ts = pd.TimeSeries(weather_df['temps' ].tolist(),index=weather_df['dates'])
        weather_ts = pd.TimeSeries(humidex.tolist(),index=weather_df['dates']) 
        dates = pd.DateRange(min(db_dates),max(db_dates),offset=pd.DateOffset(minutes=15))
        weather_ts = weather_ts.reindex(dates)
        #TODO Change this to expolation
        weather_ts[weather_ts>200] = None
        weather_ts = weather_ts.interpolate() 
        weather_ts = weather_ts.tshift(-6,freq='Min')
        self.wts = weather_ts
        #steam_ts[pd.isnull(steam_ts)] should return empty list

    def get_weather_ts(self):
        return self.wts

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

def process_data(data):
    steam_load = [item[1] for item in data]
    steam_dates = [item[0].replace(second=0,microsecond=0) for item in data]
    steam_df = pd.DataFrame({'load':steam_load,'dates':steam_dates})
    steam_df = steam_df.drop_duplicates(cols='dates',take_last=True)
    steam_ts = pd.TimeSeries(steam_df['load'].tolist(),index=steam_df['dates'])
    dates = pd.DateRange(min(steam_dates),max(steam_dates),offset=pd.DateOffset(minutes=15))
    steam_ts = steam_ts.reindex(dates)
    #TODO Change this to expolation
#    steam_ts = steam_ts.interpolate() 
    #steam_ts[pd.isnull(steam_ts)] should return empty list
    return steam_ts

def foo(ts1,ts2):
    ts3 = ts2.combine_first(ts1)
    ts3.save('../Data/ts_data')

def gen_covariates(ts,wts):
    '''
    generate covariates from data (time series)
    '''
    from os import path
    data_dir = path.join(path.dirname(__file__),'../Data')
    ts = pd.load('%s/ts_data'%(data_dir))
    #wts = pd.load('../Data/wts_data')
    wts = pd.load('%s/wts_data'%(data_dir))
    prevday_ts = ts.tshift(1,'D')
    prevday_avg = prevday_ts.resample('D',how='mean')
    prevday_avg = prevday_avg.asfreq('15Min',method='pad')
    prevweek_ts = ts.tshift(7,'D')
    #change min_date of all timeseries to the prevweek_ts date
    min_date = max(min(ts.index),min(prevweek_ts.index),min(prevday_ts.index),min(wts.index))
    max_date = max(prevday_ts.index)
    ts = ts[min_date:max_date]
    wts = wts[min_date:max_date]
    wts[wts<-20] = None
    wts = wts.interpolate()
    prevday_ts = prevday_ts[min_date:]
    prevday_avg = prevday_avg[min_date:]
    prevweek_ts = prevweek_ts[min_date:]
    hr = pd.Series(ts.index.map(lambda x: x.hour+(float(x.minute)/60)),index=ts.index)
    dod = pd.Series(ts.index.map(lambda x: int('%d%03d'%(x.year,x.dayofyear))),index=ts.index)
    df = pd.concat([ts,dod,hr],axis=1)
    df.columns=['ts','dod','hr']
    df = df.pivot(columns='dod',index='hr',values='ts')
    wdf = pd.concat([wts,dod,hr],axis=1)
    wdf.columns = ['wts','dod','hr']
    wdf = wdf.drop_duplicates(cols=['hr','dod'],take_last=True)
    wdf = wdf.pivot(columns='dod',index='hr',values='wts')
    wdf_oper = wdf[(wdf.index>5.99) & (wdf.index<18.01)]
    bldg_oper_max_humidex = wdf_oper.max() 
    bldg_oper_min_humidex = wdf_oper.min() 
    bldg_oper_avg_humidex = wdf_oper.mean() 
    tempdf = df.unstack().reset_index()
    hour_of_day = prevday_ts.index.map(lambda x: x.hour + (x.minute/60.0))
    day_of_week = prevday_ts.index.map(lambda x: x.dayofweek)
    all_covs = pd.DataFrame({'load':ts,'prevday_load':prevday_ts,'prevweek_load':prevweek_ts,'prevday_avg':prevday_avg,'weather':wts,'hour_of_day':hour_of_day,'day_of_week':day_of_week},index=prevday_ts.index)
    return all_covs,df,wdf


def create_data_df(ts,wts):
    ts = pd.load('../Data/ts_data')
    #wts = pd.load('../Data/wts_data')
    wts = pd.load('../Data/ts_humidex')
    prevday_ts = ts.tshift(1,'D')
    prevday_avg = prevday_ts.resample('D',how='mean')
    prevday_avg = prevday_avg.asfreq('15Min',method='pad')
    prevweek_ts = ts.tshift(7,'D')
    #change min_date of all timeseries to the prevweek_ts date
    min_date = max(min(ts.index),min(prevweek_ts.index),min(prevday_ts.index),min(wts.index))
    max_date = max(prevday_ts.index)
    ts = ts[min_date:max_date]
    wts = wts[min_date:max_date]
    wts[wts<-20] = None
    wts = wts.interpolate()
    prevday_ts = prevday_ts[min_date:]
    prevday_avg = prevday_avg[min_date:]
    prevweek_ts = prevweek_ts[min_date:]
    all_covs = pd.DataFrame({'load':ts,'prevday_load':prevday_ts,'prevweek_ts':prevweek_ts,'prevday_avg':prevday_avg,'weather':wts},index=prevday_ts.index)
    hr = pd.Series(ts.index.map(lambda x: x.hour+(float(x.minute)/60)),index=ts.index)
    dod = pd.Series(ts.index.map(lambda x: int('%d%03d'%(x.year,x.dayofyear))),index=ts.index)
    mydf = pd.concat([ts,dod,hr],axis=1)
    mydf.columns=['ts','dod','hr']
    mydf = mydf.pivot(columns='dod',index='hr',values='ts')
    mywdf = pd.concat([wts,dod,hr],axis=1)
    mywdf.columns = ['wts','dod','hr']
    mywdf = mywdf.drop_duplicates(cols=['hr','dod'],take_last=True)
    mywdf = mywdf.pivot(columns='dod',index='hr',values='wts')
    return mydf,mywdf

def find_similar(mydf,input_day,num=20):
    #inputts = mydf[2013010]
    inputts = mydf[input_day]
    diffdf = mydf.apply(lambda x:abs(x-inputts),axis=0).sum()
    diffdf.sort()
    dates = diffdf[:num].index
    days = [dt.datetime.strptime(str(item),'%Y%j') for item in dates]
    #print days
    #mydf[dates].plot()
    weekday_name = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    weekdays = [weekday_name[item.weekday()] for item in days ]
    fig,axes = plt.subplots(nrows=2,ncols=2)
    wddf = mydf[dates]
    wddf.columns = weekdays
    wddf.ix[:,:5].plot(ax=axes[0,0])
    wddf.ix[:,5:10].plot(ax=axes[0,1])
    wddf.ix[:,10:15].plot(ax=axes[1,0])
    wddf.ix[:,15:20].plot(ax=axes[1,1])
    return dates
    #plt.show()


def find_similar_weather(mydf,mywdf,input_day,start=-0.1,end=24.1,num=20):
    #inputts = mydf[2013010]
    wdf = {}
    for x in mywdf.columns:
        if(dt.datetime.strptime(str(int(x)),'%Y%j').weekday() <= 4):
            wdf[x] = mywdf[x]
    mywdf = pd.DataFrame(wdf)
    mywdf = mywdf[(mywdf.index>=start) & (mywdf.index<=end)]
    inputts = mywdf[input_day]
    diffdf = mywdf.apply(lambda x:abs(x-inputts),axis=0).sum()
    diffdf.sort()
    dates = diffdf[:num].index
    print dates
    days = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
    #print days
    #mydf[dates].plot()
    weekdays = [weekday_name[item.weekday()] for item in days ]
    print weekdays
    fig,axes = plt.subplots(nrows=2,ncols=2)
    wddf = mydf[dates]
    wddf.columns = weekdays
    wddf.ix[:,:5].plot(ax=axes[0,0])
    wddf.ix[:,5:10].plot(ax=axes[0,1])
    wddf.ix[:,10:15].plot(ax=axes[1,0])
    wddf.ix[:,15:20].plot(ax=axes[1,1])
    wddf.ix[:,:10].plot()
    return dates
    #plt.show()


def plot_preheat(df,wdf,dates=None):
    mydf = df[dates]
    mywdf = wdf[dates]
    work_start = 4.99
    work_end = 21.01
    workdf = mydf[(mydf.index>=work_start) & (mydf.index<=work_end)]
    total1,total2 = workdf.sum()/4
    peak_start = 5.99
    peak_end = 11.01
    peakdf = mydf[(mydf.index>=peak_start) & (mydf.index<=peak_end)]
    charge1,charge2 = peakdf.apply(lambda x: max(x)*1629)
    if dates is not None:
        plotdates = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
    mydf.columns = plotdates
    mywdf.columns = plotdates
    fig,axes = plt.subplots(nrows=2,ncols=1)
    ax = mydf.plot(ax=axes[0])
    ax.set_ylabel('Steam (Mlb/hr)')
    ax = mywdf.plot(ax=axes[1])
    ax.set_ylabel('Humidex (Celsius)')
    return (total1,total2,charge1,charge2)

def extract_dates():
    costs = []
    costs.append([2013087,2011336])
    costs.append([2013080,2012041])
    costs.append([2013071,2012338])
    costs.append([2013070,2012349])
    costs.append([2013059,2011059])
    costs.append([2013043,2012012])
    costs.append([2013037,2011004])
    costs.append([2013030,2011059])
    costs.append([2013014,2012338])
    costs.append([2013010,2011349])
    costs.append([2013009,2011090])
    costs.append([2012362,2013029])
    costs.append([2012353,2012047])
    costs.append([2012346,2011356])
    costs.append([2012345,2011355])
    costs.append([2012340,2012033])
    costs.append([2012089,2011356])
    costs.append([2012088,2011048])
    costs.append([2012086,2012338])
    costs.append([2012051,2012348])
    costs.append([2012040,2011027])
    costs.append([2012019,2011040])
    return costs

def plot_preheat2(df,wdf):
    costs = []
    costs.append(plot_preheat(df,wdf,[2013087,2011336]))
    costs.append(plot_preheat(df,wdf,[2013080,2012041]))
    costs.append(plot_preheat(df,wdf,[2013071,2012338]))
    costs.append(plot_preheat(df,wdf,[2013070,2012349]))
    costs.append(plot_preheat(df,wdf,[2013059,2011059]))
    costs.append(plot_preheat(df,wdf,[2013043,2012012]))
    costs.append(plot_preheat(df,wdf,[2013037,2011004]))
    costs.append(plot_preheat(df,wdf,[2013030,2011059]))
    costs.append(plot_preheat(df,wdf,[2013014,2012338]))
    costs.append(plot_preheat(df,wdf,[2013010,2011349]))
    costs.append(plot_preheat(df,wdf,[2013009,2011090]))
    costs.append(plot_preheat(df,wdf,[2012362,2013029]))
    costs.append(plot_preheat(df,wdf,[2012353,2012047]))
    costs.append(plot_preheat(df,wdf,[2012346,2011356]))
    costs.append(plot_preheat(df,wdf,[2012345,2011355]))
    costs.append(plot_preheat(df,wdf,[2012340,2012033]))
    costs.append(plot_preheat(df,wdf,[2012089,2011356]))
    costs.append(plot_preheat(df,wdf,[2012088,2011048]))
    costs.append(plot_preheat(df,wdf,[2012086,2012338]))
    costs.append(plot_preheat(df,wdf,[2012051,2012348]))
    costs.append(plot_preheat(df,wdf,[2012040,2011027]))
    costs.append(plot_preheat(df,wdf,[2012019,2011040]))
    return costs
    
def plot4(df,dates=None):
    if dates is not None:
        days = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
        #print days
        #mydf[dates].plot()
        weekday_name = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        weekdays = [weekday_name[item.weekday()] for item in days ]
        print weekdays
    fig,axes = plt.subplots(nrows=2,ncols=2)
    if dates is not None:
        df.columns = weekdays
    df.ix[:,:5].plot(ax=axes[0,0])
    df.ix[:,5:10].plot(ax=axes[0,1])
    df.ix[:,10:15].plot(ax=axes[1,0])
    df.ix[:,15:20].plot(ax=axes[1,1])


#5am to 9am 
def model_ash(ts,wts):
    select_hrs = ts.index.map(lambda x: x.hour + (float(x.minute)/60))
    ts = ts[(select_hrs>4.99)&(select_hrs<9.01)]
    hr = pd.Series(ts.index.map(lambda x: x.hour+(float(x.minute)/60)),index=ts.index)
    dod = pd.Series(ts.index.map(lambda x: int('%d%03d'%(x.year,x.dayofyear))),index=ts.index)
    mydf = pd.concat([ts,dod,hr],axis=1)
    mydf.columns=['ts','dod','hr']
    mydf = mydf.pivot(columns='dod',index='hr',values='ts')
    mywdf = pd.concat([wts,dod,hr],axis=1)
    mywdf.columns = ['wts','dod','hr']
    mywdf.drop_duplicates(cols=['hr','dod'],take_last=True,in_place=True)
    mywdf = mywdf.pivot(columns='dod',index='hr',values='wts')
    obs = np.matrix(mydf).transpose()
    g = mixture.GMM(n_components=5)
    g.fit(obs)
    results = g.means_
    results_df = pd.DataFrame(results.transpose())

