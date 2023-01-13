#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Input, Model
from keras import optimizers
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

## fix the figure size and axes grid
mpl.rcParams['figure.figsize'] =  (12,8)
mpl.rcParams['axes.grid'] = False

## shifts columns of dataframe df by shift
def supervised(df,cols,shift):
    n_cols=len(cols)
  ##shifts columns of dataframe df by shift
    n_cols=len(cols)
    ad=df.iloc[:,cols]
    nms=ad.columns
    cols,names=[],[]
    for i in range(shift,0,-1):
        cols.append(ad.shift(i))
        names+=[('%s(t-%d)'%(nms[j],i)) for j in range(len(nms))]
    cols.append(ad.shift(0))
    names+=[('%s(t)'%(nms[j])) for j in range(len(nms))]
    agg=pd.concat(cols,axis=1)
    agg.columns=names 
    agg.dropna(inplace=True)
    return agg

## this function removed the data from simulated and observed data wherever the observed data contains nan
def filter_nan(s,o):
    data = np.array([s.flatten(),o.flatten()])
    data = np.transpose(data)
    return data[:,0],data[:,1]

## Evaluation metrics
def NS(s,o):
    
    #Nash Sutcliffe efficiency coefficient
    #input:
        #s: simulated
        #o: observed
    #output:
        #ns: Nash Sutcliffe efficient coefficient
    
    s,o = filter_nan(s,o)
    return 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
def pc_bias(s,o):
    """
    Percent Bias
    input:
        s: simulated
        o: observed
    output:
        pc_bias: percent bias
    """
    s,o = filter_nan(s,o)
    return 100.0*sum(o-s)/sum(o)
def rmse(s,o):
    """
    Root Mean Squared Error
    input:
        s: simulated
        o: observed
    output:
        rmses: root mean squared error
    """
    s,o = filter_nan(s,o)
    return np.sqrt(np.mean((s-o)**2))
def WB(s,o):
    s,o = filter_nan(s,o)
    return 1 - abs(1 - ((sum(s))/(sum(o))))

## import data
df = pd.read_csv('keesara_catchment_daily_lumped_PIML_input_data_with_SIMHYD_output.csv')

## define training period
cal_start = '1998-01-01' 
cal_end = '2010-12-31'
training_period = len(pd.date_range(start = cal_start,end = cal_end))
df['ET_ratio_act'] = df['ET_sim']/df['PET_total']
df['ET_ratio_req'] = df['ET_total']/df['PET_total']

perform = pd.DataFrame()
perform["Station"] = ""
perform["Dropout"] = ""
perform["Epochs"] = ""
perform["Hidden_units"] = ""

perform["NSE_cal_ET"] = ""
perform["PBIAS_cal_ET"] = ""
perform["RMSE_cal_ET"] = ""
perform["WB_cal_ET"] = ""

perform["NSE_val_ET"] = ""
perform["PBIAS_val_ET"] = ""
perform["RMSE_val_ET"] = ""
perform["WB_val_ET"] = ""
perform['constraint_violation_ratio'] = ""
perform['Batch_size_ET'] = ""

perform['Batch_size_Q'] = ""
perform["NSE_cal_Q"] = ""
perform["PBIAS_cal_Q"] = ""
perform["RMSE_cal_Q"] = ""
perform["WB_cal_Q"] = ""
perform["NSE_val_Q"] = ""
perform["PBIAS_val_Q"] = ""
perform["RMSE_val_Q"] = ""
perform["WB_val_Q"] = ""

## select required data for prediction of given variable
data_ET = df[['Pptn_total','PET_total','SM_sim','ET_ratio_req']]
data_Q = df[['Pptn_total','ET_total','SM_sim','GW_sim','Q_ds']]
ndays = 2  
n_outputs = 1 
ninputs_ET = 3
ninputs_Q = 4
nobs_ET = ndays * ninputs_ET
ndays_Q = 4
nobs_Q = ndays_Q * ninputs_Q
Ntest = training_period-1

## model hyperparameters
batch_size_ET = 64 
hidden_units_ET = 90
dropout_ET = 0.1
epochs_ET = 1000
hidden_units_Q = 100
dropout_Q = 0.4
batch_size_Q = 64
epochs_Q = 600

### ET prediction
reframed_ET = supervised(data_ET,[0,1,2,3],1)
reframed_new_ET = reframed_ET[['Pptn_total(t)','PET_total(t)','SM_sim(t)','Pptn_total(t-1)','PET_total(t-1)','SM_sim(t-1)','ET_ratio_req(t)']]
reframed_new_ET[['Pptn_total(t-1)', 'PET_total(t-1)','SM_sim(t)']] = 0  
XYdata_ET = reframed_new_ET.values
yobs_train_ET = XYdata_ET[:Ntest, -n_outputs:]
yobs_test_ET = XYdata_ET[Ntest+1:, -n_outputs:]

## split into train and test datasets
## selective preprocessing of training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain_ET = scaler.fit_transform(XYdata_ET[:Ntest, :-1])
scaledXYtest_ET = scaler.transform(XYdata_ET[Ntest+1:, :-1])
scaledXYtrain_ET_final = np.append(scaledXYtrain_ET,yobs_train_ET, axis =1)
scaledXYtest_ET_final = np.append(scaledXYtest_ET ,yobs_test_ET, axis =1)

## split into input and outputs
train_X_ET, train_y_ET = scaledXYtrain_ET_final[:, :nobs_ET], scaledXYtrain_ET_final[:, -n_outputs:]
test_X_ET = scaledXYtest_ET_final[:, :nobs_ET]
PET_data = (df['PET_total']).values
PET_data_train = PET_data[:Ntest]
PET_data_test = PET_data[Ntest+2:]
ET_data = (df['ET_total']).values
ET_data_train = ET_data[:Ntest]
ET_data_test = ET_data[Ntest+2:]

## reshape input to be 3D [samples, timesteps, features]
train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], ndays, ninputs_ET))
test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], ndays, ninputs_ET))
input_shape=(train_X_ET.shape[1], train_X_ET.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_ET = Sequential()   
model_ET.add(LSTM(hidden_units_ET, input_shape=(train_X_ET.shape[1], train_X_ET.shape[2])))
model_ET.add(Dropout(dropout_ET))
model_ET.add(Dense(n_outputs, activation = 'sigmoid')) 
model_ET.compile(loss = 'mse',optimizer='adam')
history_ET = model_ET.fit(train_X_ET, train_y_ET, batch_size = batch_size_ET, epochs=epochs_ET , shuffle=True,validation_split=0.2, verbose=0)
plt.plot(history_ET.history['loss'])
plt.plot(history_ET.history['val_loss'])
plt.title('model loss for ET/PET prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_ET = model_ET.predict(train_X_ET)
train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], nobs_ET))
inv_yhat_ET = np.concatenate((train_X_ET, yhat_ET), axis=1)
NNytrain_ET = inv_yhat_ET[:,-n_outputs:]
train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], ndays, ninputs_ET))

## outputs in testing
yhat_test = model_ET.predict(test_X_ET)
test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], nobs_ET))
inv_yhat_test_ET = np.concatenate((test_X_ET, yhat_test), axis=1)
NNytest_ET = inv_yhat_test_ET[:,-n_outputs]
test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], ndays, ninputs_ET))
mean_train_ET = np.array(NNytrain_ET)
mean_test_ET = np.array(NNytest_ET)

## evaluate outputs
ratio_ET_PET_great_than_1 = np.sum(np.array((mean_test_ET.reshape(mean_test_ET.shape[0],))) > 1, axis=0)
s1_ET_train = np.multiply(PET_data_train,(mean_train_ET.reshape(mean_train_ET.shape[0],)))
s1_ET_test = np.multiply(PET_data_test,(mean_test_ET.reshape(mean_test_ET.shape[0],)))
o1_ET_train = ET_data_train
o1_ET_test = ET_data_test
NS_ET = 1 - sum((s1_ET_test-o1_ET_test)**2)/sum((o1_ET_test-np.mean(o1_ET_test))**2)
PBIAS_ET = pc_bias(s1_ET_test,o1_ET_test)
RMSE_ET = rmse(s1_ET_test,o1_ET_test)
WB_ET = 1 - abs(1 - ((sum(s1_ET_test))/(sum(o1_ET_test))))
NS_ET_cal = 1 - sum((s1_ET_train-o1_ET_train)**2)/sum((o1_ET_train-np.mean(o1_ET_train))**2)
PBIAS_ET_cal = pc_bias(s1_ET_train,o1_ET_train)
RMSE_ET_cal = rmse(s1_ET_train,o1_ET_train)
WB_ET_cal = 1 - abs(1 - ((sum(s1_ET_train))/(sum(o1_ET_train))))

# downstream streamflow prediction 
reframed_Q = supervised(data_Q,[0,1,2,3,4],3) 
reframed_new_Q = reframed_Q[['Pptn_total(t)','ET_total(t)','ET_total(t-1)','ET_total(t-2)','SM_sim(t)','GW_sim(t)','Pptn_total(t-1)','SM_sim(t-1)','GW_sim(t-1)','Pptn_total(t-2)','SM_sim(t-2)','GW_sim(t-2)','Pptn_total(t-3)', 'ET_total(t-3)','SM_sim(t-3)','GW_sim(t-3)','Q_ds(t)']]
reframed_new_Q[['Pptn_total(t-3)', 'ET_total(t-3)']] = 0 
XYdata_Q = reframed_new_Q.values

## split into train and test datasets
XYtrain_Q = XYdata_Q[3:Ntest, :]
XYtrain_Q[:,1] = s1_ET_train[3:]
XYtrain_Q[:,2] = s1_ET_train[2:-1]
XYtrain_Q[:,3] = s1_ET_train[1:-2]
XYtest_Q = XYdata_Q[Ntest+4:, :]
XYtest_Q[:,1] = s1_ET_test[5:]
XYtest_Q[:,2] = s1_ET_test[4:-1]
XYtest_Q[:,3] = s1_ET_test[3:-2]
yobs_train_Q = XYdata_Q[3:Ntest, -1:]
yobs_test_Q = XYdata_Q[Ntest+4:, -1:]
scaledXYtrain_Q = XYtrain_Q
scaledXYtest_Q = XYtest_Q

## split into input and outputs
train_X_Q, train_y_Q = scaledXYtrain_Q[:, :nobs_Q], scaledXYtrain_Q[:, -1:]
test_X_Q = scaledXYtest_Q[:, :nobs_Q]

## reshape input to be 3D [samples, timesteps, features]
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], ndays_Q, ninputs_Q))
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], ndays_Q, ninputs_Q))

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_Q = Sequential()  
model_Q.add(LSTM(hidden_units_Q, input_shape=(train_X_Q.shape[1], train_X_Q.shape[2])))
model_Q.add(Dropout(dropout_Q))
model_Q.add(Dense(1, activation = 'relu'))    
model_Q.compile(loss = 'mse',optimizer='adam')
history_Q = model_Q.fit(train_X_Q, train_y_Q, batch_size = batch_size_Q, epochs=epochs_Q, shuffle=True,validation_split=0.2, verbose=0)
plt.plot(history_Q.history['loss'])
plt.plot(history_Q.history['val_loss'])
plt.title('model loss for Q prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_Q = model_Q.predict(train_X_Q)
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nobs_Q))
inv_yhat_Q = np.concatenate((train_X_Q, yhat_Q), axis=1)
NNytrain_Q = inv_yhat_Q[:,-1:]
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], ndays_Q, ninputs_Q))

## outputs in testing
yhat_test_Q = model_Q.predict(test_X_Q)
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nobs_Q))
inv_yhat_test_Q = np.concatenate((test_X_Q, yhat_test_Q), axis=1)
NNytest_Q = inv_yhat_test_Q[:,-1:]
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], ndays_Q, ninputs_Q))
mean_train_Q = np.array(NNytrain_Q)
mean_test_Q = np.array(NNytest_Q)

## evaluate outputs
(s,o) = (mean_test_Q[:], yobs_test_Q[:])
NS_Q = NS(s,o)
PBIAS_Q = pc_bias(s,o)
RMSE_Q = rmse(s,o)
WB_Q = WB(s,o)
(s_train, o_train) = (mean_train_Q, yobs_train_Q)
NS_Q_cal = NS(s_train, o_train)
PBIAS_Q_cal = pc_bias(s_train,o_train)
RMSE_Q_cal = rmse(s_train,o_train)
WB_Q_cal = WB(s_train, o_train)
op = df[['Date','Pptn_total','ET_total','Q_ds']][Ntest+7:]
op1 = op.reset_index(drop=True)
op1['ET_piml_sim'] = s1_ET_test[5:]
op1['Q_piml_sim'] = mean_test_Q
op1.to_csv('keesara_catchment_lumped_PIML_testing_output.csv')

perform = perform.append({'Station':stn_name_us_ds,"Dropout":dropout_Q,"Epochs":epochs_Q,"Hidden_units":hidden_units_Q,'NSE_cal_ET':NS_ET_cal,'PBIAS_cal_ET':PBIAS_ET_cal ,'RMSE_cal_ET':RMSE_ET_cal,'WB_cal_ET':WB_ET_cal,'Batch_size_ET':batch_size_ET,'NSE_val_ET':NS_ET,'PBIAS_val_ET':PBIAS_ET ,'RMSE_val_ET':RMSE_ET,'WB_val_ET':WB_ET,'constraint_violation_ratio':ratio_ET_PET_great_than_1,'Batch_size_Q' : batch_size_Q, 
"NSE_cal_Q" : NS_Q_cal, 
"PBIAS_cal_Q" : PBIAS_Q_cal, 
"RMSE_cal_Q" : RMSE_Q_cal, 
"WB_cal_Q" : WB_Q_cal, 
"NSE_val_Q" : NS_Q, 
"PBIAS_val_Q" : PBIAS_Q, 
"RMSE_val_Q" : RMSE_Q, 
"WB_val_Q" : WB_Q },ignore_index = True) 
print(perform)

