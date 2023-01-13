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
df = pd.read_csv('keesara_catchment_daily_ML_input_data.csv')

## define training period
cal_start = '1998-01-01' 
cal_end = '2010-12-31'
training_period = len(pd.date_range(start = cal_start,end = cal_end))

perform = pd.DataFrame()
perform["Dropout"] = ""
perform["Epochs"] = ""
perform["Hidden_units"] = ""

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
data_Q = df[['Pptn_total','PET_total','Q_ds']]
ndays = 1  
nfuture = 1 
ninputs_Q = 2
ndays_Q = 1
nobs_Q = ndays_Q * ninputs_Q
Ntest = training_period

## model hyperparameters
hidden_units_Q = 20
dropout_Q = 0.4
batch_size_Q = 32
epochs_Q = 300

### Q prediction
reframed_Q = supervised(data_Q,[0,1,2],0) 
reframed_new_Q = reframed_Q[['Pptn_total(t)','PET_total(t)','Q_ds(t)']]

## split into train and test datasets
XYdata_Q = reframed_new_Q.values
XYtrain_Q = XYdata_Q[:Ntest, :]
XYtest_Q = XYdata_Q[Ntest:, :]
yobs_train_Q = XYdata_Q[:Ntest, -1:]
yobs_test_Q = XYdata_Q[Ntest:, -1:]
scaledXYtrain_Q = XYtrain_Q
scaledXYtest_Q = XYtest_Q

## split into input and outputs
train_X_Q, train_y_Q = scaledXYtrain_Q[:, :nobs_Q], scaledXYtrain_Q[:, -1:]
test_X_Q = scaledXYtest_Q[:, :nobs_Q]
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
op = df[['Date','Pptn_total','PET_total','Q_ds']][Ntest:] 
op1 = op.reset_index(drop=True)
op1['Q_ml_sim'] = s[:]
op1.to_csv('keesara_catchment_ML_testing_output.csv')

perform = perform.append({"Dropout":dropout_Q,"Epochs":epochs_Q,"Hidden_units":hidden_units_Q,'Batch_size_Q' : batch_size_Q, 
"NSE_cal_Q" : NS_Q_cal, 
"PBIAS_cal_Q" : PBIAS_Q_cal, 
"RMSE_cal_Q" : RMSE_Q_cal, 
"WB_cal_Q" : WB_Q_cal, 
"NSE_val_Q" : NS_Q, 
"PBIAS_val_Q" : PBIAS_Q, 
"RMSE_val_Q" : RMSE_Q, 
"WB_val_Q" : WB_Q },ignore_index = True) 
print(perform)

