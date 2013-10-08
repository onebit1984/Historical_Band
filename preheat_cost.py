import datetime as dt
import pandas as pd
import numpy as np

class Preheat:
    def __init__(self,df,wdf):
        self.df = df
        self.wdf = wdf
        self.weekday_name = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

    
    def preheat_tag(self,start_date='2011001',end_date='2013108',start=4.99,end=5.91,limit=15.01):
        '''
        tags the days that are identified as preheat days
        days are returned in format: YYYYDOY {DOY=day of year}
        '''
        df = self.df
        mydf = df.ix[:,start_date:end_date]
        df6 = mydf[(df.index>=5.9)&(df.index<6.1)]
        mydf = mydf[(df.index>=start) & (df.index<=end)]
        mydf = mydf.ix[:,mydf.apply(lambda x: any(x>limit))]
        df6 = df6[mydf.columns]
        maxdf = mydf.apply(lambda x: max(x))
        #print mydf.ix[:,(maxdf.values>df6.values)[0]]
        return mydf.columns


    def read_preheat_tag(self):
        '''
        read the csv file to identify manually tagged tags
        '''
        f = open('../Data/steam_preheat_manual.csv','rU')
        f.readline()
        preheat_dods = []
        for line in f:
            date = line.rstrip()
            if('Dec' not in date):
                print date
                date = date.split('-')
                date.append('2013')
                date[0] = '%02d'%(int(date[0]))
                preheat_dods.append(dt.datetime.strptime(''.join(date[::-1]),'%Y%b%d'))
        return preheat_dods

    def get_dates(self,start_date,end_date):
        '''
        for the given input dates, return the dates excluding holidays, 
        fridays, saturdays, sundays
        '''
        dates = pd.date_range(start_date,end_date)
        dates = dates[:-1]
        dates = [d for d in dates if (d.weekday()!=4 and d.weekday()!=5 and d.weekday()!=6)]
        holidays = [dt.datetime(2012,12,25),dt.datetime(2012,12,25),dt.datetime(2013,01,01)] 
        dates = [d for d in dates if d not in holidays]
        return dates

    def get_similar_weather_dates(self,input_day,start=-0.1,end=24.1,num=20):
        '''
        For the given input day, find the most similar weather days
        using L-1 distance
        input_day format: yyyydoy
        '''
        #inputts = mydf[2013010]
        mywdf = self.wdf
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
        #print dates
        days = [dt.datetime.strptime(str(int(item)),'%Y%j') for item in dates]
        #print days
        #mydf[dates].plot()
        weekdays = [self.weekday_name[item.weekday()] for item in days ]
        return dates


    def find_preheat_days(self,month=2,year=2013):
        '''
        for the given month & year
        find the preheat days for the days that dont have preheat
        '''
        df = self.df
        start_date = '%d-%02d-01'%(year,month)
        start_date = dt.datetime.strptime(start_date,'%Y-%m-%d')
        if month!=12:
            end_date =  '%d-%02d-01'%(year,month+1)
        else:
            end_date =  '%d-%02d-01'%(year+1,1)
        end_date = dt.datetime.strptime(end_date,'%Y-%m-%d')    
        dates = self.get_dates(start_date,end_date) #find dates for a month except weekends and holidays
        dods = [int(d.strftime('%Y%j')) for d in dates]
        start = start_date.strftime('%Y%j')
        end = end_date.strftime('%Y%j')
        preheat_dods = self.preheat_tag()
        min_cost_dates = {}
        for dod in dods:
            if dod not in preheat_dods:
                similar_dates = self.get_similar_weather_dates(int(dod))
                temp_similar_dates = similar_dates
                while not(set(preheat_dods)& set(temp_similar_dates[:5])):
                    print temp_similar_dates
                    if not list(temp_similar_dates):
                        break
                    temp_similar_dates = temp_similar_dates[5:] 
             #   print temp_similar_dates
                if not list(temp_similar_dates):
                    print 'nothing similar found!'
                    continue
            #    print temp_similar_dates
                similar_preheat_dods = (set(preheat_dods)& set(temp_similar_dates[:5]))
             #   print similar_preheat_dods
                if similar_preheat_dods:
                    dds = [dod]
                    dds += list(similar_preheat_dods)
                    dds = map(int,dds)
              #      print dds
                    df[dds].plot()
                   # min_date = find_min_steam(similar_preheat_dods)
                   # min_cost_dates[date]= min_date
        return min_cost_dates,preheat_dods

    def calc_preheat_cost(self,no_preheat_dods,preheat_dods):
        mydf = self.df
        mywdf = self.wdf
        fig,axes = plt.subplots(nrows=2,ncols=1)
        preheat_df = mydf[preheat_dods]
        preheat_df = preheat_df.unstack()
        preheat_df = preheat_df.reset_index()
        no_preheat_df = mydf[no_preheat_dods]
        no_preheat_df = no_preheat_df.unstack()
        no_preheat_df = no_preheat_df.reset_index()
        no_preheat_dates = []
        for dod in no_preheat_dods:
            date = dt.datetime.strptime(str(dod),'%Y%j')
            no_preheat_dates += list(pd.date_range(date,periods=96,freq='15Min'))

        no_preheat_dates = pd.DatetimeIndex(no_preheat_dates)
        plotdf = pd.DataFrame({'preheat':preheat_df[0].values,'no_preheat':no_preheat_df[0].values},index=no_preheat_dates)

