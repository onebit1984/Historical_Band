"""
Author: Vaibhav Bhandari <vb2317@columbia.edu>

===============================
Build an ensemble using SVR/HMM/RF?GRADIENT BOOSTED REG??

This script creates an ensemble model from the data and predicts 
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from process_load import create_data_df


def build_covariates():
    """
    following covariates are created:

    humidex
    bldg_oper_min_humidex
    bldg_oper_max_humidex
    bldg_oper_avg_humidex
    hour
    hour_of_day
    day_of_wk
    weekend_id
    oper_hour   holiday_pct
    day_ago_steam_demand
    wk_ago_steam_demand
    prev_day_avg_steam_demand
    prev_wk_avg_steam_demand
    prev_day_hour_avg_steam_demand
    prev_wk_hour_avg_steam_demand
    """
    df,wdf = create_data_df(None,None)
    humidex = wdf.unstack().values
    wdf_oper = wdf[(wdf.index>5.99) & (wdf.index<18.01)]
    bldg_oper_max_humidex = wdf_oper.max() 
    bldg_oper_min_humidex = wdf_oper.min() 
    bldg_oper_avg_humidex = wdf_oper.mean() 
    tempdf = df.unstack().reset_index()
    hour_of_day = tempdf['hr']
    day_ago_steam_demand = 0

import process_load as pl
all_covs,_,_ = pl.gen_covariates(None,None)
allcovs = all_covs[:datetime(2013,04,18)]
all_covs = allcovs.ix[:all_covs.index[-2]] 

predict_len = 96
all_covs = all_covs[:-97]
y = pd.Series(all_covs['load'].values,index=all_covs.index)
X = all_covs[['prevday_load','prevweek_load','prevday_avg','weather','hour_of_day','day_of_week']]
n = len(all_covs)
test_idx = np.zeros(n,dtype=bool)
test_idx[(n-predict_len):] = True
y_train = y[test_idx==False]
y_test  = y[test_idx]
X_train = X[test_idx==False]
X_test = X[test_idx]

#trim the training values
start_train_idx = len(y_train) - (96*180) 
start_train_idx = 0
y_train = y_train[start_train_idx:]
X_train = X_train[start_train_idx:]

#fit various models
##############################################################################
#Run SVR
def run_svr(X_train,y_train,X_test):
    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
    svr_model = svr_rbf.fit(X_train,y_train) 
    y_svr = svr_model.predict(X_test)
    return y_svr

def run_gbr(X_train,y_train,X_test):
    from sklearn.ensemble import GradientBoostingRegressor
    gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
    y_gbr = gbr_model.predict(X_test)
    return y_gbr

import pylab as pl
y_predict = y_gbr
pl.figure()
pl.plot(range(96),y_test,color='k')
pl.hold(True)
pl.plot(range(96),y_predict,color='k')
pl.show()

##############################################################################
#
#Fit GMM
def plot_elbow(obs,start,end):
    from sklearn.mixture import GMM,DPGMM
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
##############################################################################
# Run HMM
X_hmm = np.column_stack((y_train,X_train[['hour_of_day','weather','day_of_week']]))
#X_hmm = np.column_stack((y_train,X_train[['hour_of_day','weather']]))
#X_hmm = y_train
from sklearn.hmm import GaussianHMM
n_clusters = 9
#n_clusters = 17
model = GaussianHMM(n_clusters,covariance_type='diag',n_iter=1000)
model.fit([X_hmm])
hidden_states = model.predict(X_hmm)
viterbi_states = model.decode(X_hmm)
x_ax = np.asarray(range(len(X_hmm)))
x_ax = X_train['hour_of_day'] + X_train['day_of_week']*24
#x_ax = X_train['hour_of_day']
x_ax = np.asarray([item.to_datetime() for item in X_train.index])
def plot_HMM(n_clusters,hidden_states,x_ax,y_ax):
    #PLOT HIDDEN STATES
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in xrange(n_clusters):
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
    ax.set_title('%d State HMM'%(n_clusters))
    ax.legend()
    ax.set_ylabel('Load (Mlb/Hr)')
    ax.set_xlabel('Time')
    ax.grid(True)
#PLOT VITERBI STATES
fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
x_ax = np.asarray(range(len(X_hmm)))
x_ax = X_train['hour_of_day'] + X_train['day_of_week']*24
#x_ax = X_train['hour_of_day']
#x_ax = np.asarray([item.to_datetime() for item in X_train.index])
for i in xrange(n_clusters):
    print i
    idx = (viterbi_states==i)
    ax.plot_date(x_ax[idx],y_train[idx],'o',label='%dth state'%i)
ax.legend()
ax.set_ylabel('Load (Mlb/Hr)')
ax.set_xlabel('Time')
ax.grid(True)
plt.show()

##############################################################################
#create a new model
modelw = GaussianHMM(n_clusters,covariance_type='diag',n_iter=1000)
modelw._means_ = model.means_[:,1:]
modelw._covars_ = [item[1:,1:] for item in model.covars_]
modelw.transmat_ = model.transmat_
modelw.n_features = model.n_features-1
modelw.startprob_ = model.startprob_

##############################################################################
#RUN ENSEMBLE
def run_ensemble(X_train,y_train,X_test,y_test):
    hours = X_train.index.map(lambda x: x.hour)
    test_hours = X_test.index.map(lambda x: x.hour)
    y_svr = []
    y_gbr = []
    for i in xrange(24):
        X_train_hourly = X_train[hours==i]
        y_train_hourly = y_train[hours==i]
        X_test_hourly = X_test[test_hours==i]
        from sklearn.svm import SVR
        y_svr = np.append(y_svr,run_svr(X_train_hourly,y_train_hourly,X_test_hourly))
        y_gbr = np.append(y_gbr,run_gbr(X_train_hourly,y_train_hourly,X_test_hourly))
    return y_svr,y_gbr

fig = plt.figure()
ax = fig.add_subplot(111)
x_ax = np.asarray([item.to_datetime() for item in X_test.index])
ax.plot(x_ax,y_test,label='Actual')
ax.plot(x_ax,y_svr,label='SVR+HMM')
#ax.plot(x_ax,y_svr_old,label='SVR')
ax.plot(x_ax,y_gbr,label='GBR+HMM')
#ax.plot(x_ax,y_gbr_old,label='GBR')
ax.legend()
ax.grid(True)
plt.show()
##############################################################################
#RUN FINAL MODEL
ens_svr = []
ens_gbr = []
ens_test = []
for i in range(15):
    predict_len = 96
    test_idx = np.zeros(n,dtype=bool)
    test_idx[(n-predict_len):] = True
    y_train = y[test_idx==False]
    y_test  = y[test_idx]
    X_train = X[test_idx==False]
    X_test = X[test_idx]
    #trim the training values
    start_train_idx = len(y_train) - (96*180)
    y_train = y_train[start_train_idx:]
    X_train = X_train[start_train_idx:]
    y_svr,y_gbr = run_ensemble(X_train,y_train,X_test,y_test)  
    ens_svr = np.append(y_svr,ens_svr)
    ens_gbr = np.append(y_gbr,ens_gbr)
    ens_test = np.append(y_test,ens_test)
    X = X[:-96]
    y = y[:-96]
    n = n-96


ens_svr_hmm = []
ens_gbr_hmm= []
ens_test = []
X = X[:-96]
y = y[:-96]
n = n-96
predict_len = 96
test_idx = np.zeros(n,dtype=bool)
test_idx[(n-predict_len):] = True
y_train = y[test_idx==False]
y_test  = y[test_idx]
X_train_hmm = X[test_idx==False]
X_test_hmm = X[test_idx]
#trim the training values
start_train_idx = len(y_train) - (96*179)
y_train = y_train[start_train_idx:]
X_train_hmm = X_train_hmm[start_train_idx:]
X_train_hmm['hidden_states'] = hidden_states[:-96]
X_test_hmm['hidden_states'] = hidden_states[-96:]
for i in range(15):
    y_svr,y_gbr = run_ensemble(X_train_hmm,y_train,X_test_hmm,y_test)  
    ens_svr_hmm = np.append(y_svr,ens_svr_hmm)
    ens_gbr_hmm= np.append(y_gbr,ens_gbr_hmm)
    ens_test = np.append(y_test,ens_test)
    X_test_hmm = X_train_hmm[-96:]
    X_train_hmm = X_train_hmm[:-96] 
    y_test = y_train[-96:]
    y_train = y_train[:-96]

fig = plt.figure()
ax.plot(ens_test,label='Actual')
x_ax = np.asarray([item.to_datetime() for item in X[-(15*96):].index])
ax = fig.add_subplot(111)
ax.plot(x_ax,ens_test,label='Actual')
ax.plot(x_ax,ens_svr,label='SVR')
ax.plot(x_ax,ens_svr_hmm,label='SVR+HMM')
ax.plot(x_ax,ens_gbr,label='GBR')
ax.plot(x_ax,ens_gbr_hmm,label='GBR+HMM')
ax.set_xlabel('Time')
ax.set_ylabel('Steam Load (Mlb/Hr)')
ax.legend()
ax.grid(True)
plt.show()

mape_svr = []
mape_svr_hmm = []
mape_gbr = []
mape_gbr_hmm = []
for i in range(15):
    mape_svr = np.append(mape_svr,calcMape(ens_test[(i*96):((i+1)*96)],ens_svr[(i*96):((i+1)*96)]))
    mape_svr_hmm = np.append(mape_svr_hmm,calcMape(ens_test[(i*96):((i+1)*96)],ens_svr_hmm[(i*96):((i+1)*96)]))
    mape_gbr = np.append(mape_gbr,calcMape(ens_test[(i*96):((i+1)*96)],ens_gbr[(i*96):((i+1)*96)]))
    mape_gbr_hmm = np.append(mape_gbr_hmm,calcMape(ens_test[(i*96):((i+1)*96)],ens_gbr_hmm[(i*96):((i+1)*96)]))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mape_svr,label='SVR')
ax.plot(mape_svr_hmm,label='SVR+HMM')
ax.plot(mape_gbr,label='GBR')
ax.plot(mape_gbr_hmm,label='GBR+HMM')
ax.set_xlabel('Time')
ax.set_ylabel('Steam Load (Mlb/Hr)')
ax.set_title('MAPE')
ax.legend()
ax.grid(True)
plt.show()


def calcMape(test,pred):
    err = np.abs(test-pred)
    pctError = 100*err/test
    mape = pctError[~np.isinf(pctError)].mean()
    return mape

ts = pd.Series(hidden_states,index=X_train.index)
#def use_states(ts):
y_svrs = np.zeros(shape=(0))
y_gbrs = np.zeros(shape=(0))
y_tests = np.zeros(shape=(0))
test_dates = np.zeros(shape=(0),dtype='datetime64[ns]')
test_dods = []
hr = pd.Series(ts.index.map(lambda x: x.hour+(float(x.minute)/60)),index=ts.index)
dod = pd.Series(ts.index.map(lambda x: int('%d%03d'%(x.year,x.dayofyear))),index=ts.index)
df_hs = pd.concat([ts,dod,hr],axis=1)
df_hs.columns=['ts','dod','hr']
df_hs = df_hs.pivot(columns='dod',index='hr',values='ts')

for _ in range(15):
    test_dod = max(df_hs.columns)
    if test_dod not in df_hs:
        continue
    test_dods.append(test_dod)
    input_ts = df_hs[test_dod]
    test_date = pd.date_range(start =datetime.strptime(str(long(test_dod)),'%Y%j'),periods=96,freq='15Min')
    del df_hs[test_dod]
    similar_state_dods = df_hs.apply(lambda x: np.not_equal(x,input_ts).sum())
    similar_state_dods.sort()
    similar_state_dods = similar_state_dods.index.values[:60][::-1]
    for ind,item in enumerate(similar_state_dods):
        dates_hmm = pd.date_range(start =datetime.strptime(str(long(item)),'%Y%j'),periods=96,freq='15Min')
        if ind==0:
            X_train_hmm = X.ix[dates_hmm]
            y_train_hmm = y[dates_hmm]
        else:
            X_train_hmm = X_train_hmm.append(X.ix[dates_hmm])
            y_train_hmm = y_train_hmm.append(y[dates_hmm])
    X_test_hmm = X.ix[pd.date_range(start =datetime.strptime(str(long(test_dod)),'%Y%j'),periods=96,freq='15Min')]
    y_test_hmm = y[pd.date_range(start =datetime.strptime(str(long(test_dod)),'%Y%j'),periods=96,freq='15Min')]
    y_svr,y_gbr = run_ensemble(X_train_hmm,y_train_hmm,X_test_hmm,y_test_hmm)
    y_svrs = np.append(y_svr, y_svrs)
    y_gbrs = np.append(y_gbr, y_gbrs)
    y_tests = np.append(y_test_hmm, y_tests)
    test_dates = np.append(test_date.values, test_dates)


fig = plt.figure()
ax = fig.add_subplot(111)
#x_ax = np.asarray([item.to_datetime() for item in X_test.index])
test_dates = [item.astype(datetime) for item in test_dates]
x_ax = test_dates
ax.plot(x_ax,y_tests,label='Actual')
ax.plot(x_ax,y_svrs,label='SVR+HMM')
#ax.plot(x_ax,y_svr_old,label='SVR')
ax.plot(x_ax,y_gbrs,label='GBR+HMM')
#ax.plot(x_ax,y_gbr_old,label='GBR')
ax.legend()
ax.grid(True)
plt.show()

