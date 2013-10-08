# Create your views here.
import json
import numpy as np
from django.http import HttpResponse
from django.template import Context,loader
from django.shortcuts import render

from Rudin import gen_covariates,Preheat

all_covs,df,wdf = gen_covariates(None,None)
preheat = Preheat(df,wdf)

def index(request):
    date_list = range(2012108,2012200)
    date_list = map(str,date_list)
    template = loader.get_template('hist_envelope/index.html')
    context = Context({'date_list':date_list,})
    return HttpResponse(template.render(context))
    #return HttpResponse("Hello! Plot the envelope!!")

def show_dates(request, input_day):
    dates = preheat.get_similar_weather_dates(float(input_day))
    dates = map(str,dates.values)
    dates_str = ';'.join(dates)
    context = Context({'input_day':input_day,'dates':dates,'dates_str':dates_str})
    return render(request,'hist_envelope/test.html',context) 
    #return HttpResponse("Displaying dates for day: %s"%dates)

def show_similar_weather(request, input_day):
    return HttpResponse("Similar Weather time series for day: %s"%input_day)

def show_similar_weather_load(request, input_day):
    '''
    generally to test various mechanisms of plotting
    maps to url: hist_envelope/{{input_day}}/show_load/
    Args
    request: HTTPrequest object
    input_day: date format= YYYYDOY
    '''
    load_ts = df[float(input_day)].values
    import json
    json_data =json.dumps({'load':load_ts.tolist(),'load_dates':df[float(input_day)].index.values.tolist()})
    similar_weather_dates = preheat.get_similar_weather_dates(float(input_day),num=3)
    json_data = json.dumps(list(df[similar_weather_dates].itertuples(index=True)))
    context = Context({'load':load_ts,'load_dates':df[float(input_day)].index.values,'input_day':input_day,'json_data':json_data})
    return render(request,'hist_envelope/plot_series.html',context)
    #return HttpResponse("Similar Weather steam load for day: %s"%input_day)

def show_similar_weather_envelope(request, input_day):
    '''
    show the historical band
    maps to url: hist_envelope/{{input_day}}/show_envelope/
    Args
    request: HTTPrequest object
    input_day: date format= YYYYDOY
    '''
    num=3
    start=-0.1
    end=24.1
    if 'num' in request.GET:
        num = request.GET['num']
    if 'start' in request.GET:
        start = float(request.GET['start']) - 0.01
    if 'end' in request.GET:
        end = float(request.GET['end']) + 0.01

    load_ts = df[float(input_day)].values
    humidex_ts = wdf[float(input_day)].values
    similar_weather_dates = preheat.get_similar_weather_dates(float(input_day),num=int(num),start=start,end=end)
    #similar_weather_dates = get_most_similar_n(df[similar_weather_dates],input_day,5)
    max_load = df[similar_weather_dates].max(axis=1)
    min_load = df[similar_weather_dates].min(axis=1)
    try:
        #load_dataset = json.dumps(np.column_stack((df.index.values,df[similar_weather_dates],max_load,min_load)).tolist())
        load_dataset = json.dumps(list(df[similar_weather_dates].itertuples(index=True)))
        weather_dataset = json.dumps(list(wdf[similar_weather_dates].itertuples(index=True)))
        keys = similar_weather_dates.tolist()
        keys += [1,0]
        keys = json.dumps(keys)
    except:
        print 'ERROR!'
    context = Context({'input_day':input_day,'load_dataset':load_dataset,'weather_dataset':weather_dataset,'keys':keys})
    return render(request,'hist_envelope/plot_envelope.html',context)
 
def get_most_similar_n(mydf,input_day,n=20):
    print mydf.columns
    inputts = mydf[float(input_day)]
    df_l1 = mydf.apply(lambda x: abs(inputts-x),axis=0).sum()
    df_l1.sort()
    return df_l1[:n].index.values
