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
mpl.rcParams['figure.figsize'] =  (12,12)
mpl.rcParams['axes.grid'] = False

## shifts columns of dataframe df by shift
def supervised(df,cols,shift):
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
df = pd.read_csv('brady_catchment_daily_PIML_input_data_with_SIMHYD_output.csv')

## live storage capacity of reservoir
RS_max = 109.395

## define training period
cal_start = '2003-01-01'
cal_end = '2015-12-31'
training_period = len(pd.date_range(start = cal_start,end = cal_end))

df['ET_ratio_act_us'] = df['ET_us_sim']/df['PET_us']
df['ET_ratio_req_us'] = df['ET_us']/df['PET_us']
df['ET_ratio_act_us_ds'] = df['ET_us_ds_sim']/df['PET_us_ds']
df['ET_ratio_req_us_ds'] = df['ET_us_ds']/df['PET_us_ds']
df['St_ratio_req_us'] = df['live_St_us']/RS_max

perform = pd.DataFrame()   
perform["NSE_cal_ET_us"] = ""
perform["PBIAS_cal_ET_us"] = ""
perform["RMSE_cal_ET_us"] = ""
perform["WB_cal_ET_us"] = ""
perform["NSE_val_ET_us"] = ""
perform["PBIAS_val_ET_us"] = ""
perform["RMSE_val_ET_us"] = ""
perform["WB_val_ET_us"] = ""
perform['constraint_violation_ratio_us'] = ""

perform["NSE_cal_ET"] = ""
perform["PBIAS_cal_ET"] = ""
perform["RMSE_cal_ET"] = ""
perform["WB_cal_ET"] = ""
perform["NSE_val_ET"] = ""
perform["PBIAS_val_ET"] = ""
perform["RMSE_val_ET"] = ""
perform["WB_val_ET"] = ""
perform['constraint_violation_ratio'] = ""

perform["NSE_cal_St"] = ""
perform["PBIAS_cal_St"] = ""
perform["RMSE_cal_St"] = ""
perform["WB_cal_St"] = ""
perform["NSE_val_St"] = ""
perform["PBIAS_val_St"] = ""
perform["RMSE_val_St"] = ""
perform["WB_val_St"] = ""
perform["constraint_violation_ratio_St"] = ""

perform["NSE_cal_Rt"] = ""
perform["PBIAS_cal_Rt"] = ""
perform["RMSE_cal_Rt"] = ""
perform["WB_cal_Rt"] = ""
perform["NSE_val_Rt"] = ""
perform["PBIAS_val_Rt"] = ""
perform["RMSE_val_Rt"] = ""
perform["WB_val_Rt"] = ""

perform["NSE_cal_Q_ds"] = ""
perform["PBIAS_cal_Q_ds"] = ""
perform["RMSE_cal_Q_ds"] = ""
perform["WB_cal_Q_ds"] = ""
perform["NSE_val_Q_ds"] = ""
perform["PBIAS_val_Q_ds"] = ""
perform["RMSE_val_Q_ds"] = ""
perform["WB_val_Q_ds"] = ""

## select required data for prediction of given variable
data_ET_us = df[['Pptn_us','PET_us','SM_us_sim','ET_ratio_req_us']]
data_ET_us_ds = df[['Pptn_us_ds','PET_us_ds','SM_us_ds_sim','ET_ratio_req_us_ds']]
data_St = df[['Pptn_us','ET_us','SM_us_sim','GW_us_sim','live_St_us','St_ratio_req_us']]
data_Rt = df[['Pptn_us','ET_us','SM_us_sim','GW_us_sim','live_St_us','Res_outflow']]
data_Q_ds = df[['Pptn_us_ds','ET_us_ds','SM_us_ds_sim','GW_us_ds_sim','Res_outflow','Q_ds']]
ndays = 2  
n_outputs = 1 
ninputs_ET = 3
ninputs_ET_us = 3
ninputs_St = 5
nobs_ET = ndays * ninputs_ET
nobs_ET_us = ndays * ninputs_ET_us
ndays_St = 3
nobs_St = ndays_St * ninputs_St
Ntest = training_period-1
ninputs_Rt = 5
ndays_Rt = 3
nobs_Rt = ndays_Rt * ninputs_Rt
ninputs_Q = 5
ndays_Q = 4
nobs_Q = ndays_Q * ninputs_Q    

## model hyperparameters
batch_size_ET = 32 
batch_size_ET_us = 32
hidden_units_ET = 90
hidden_units_ET_us = 40
dropout_ET = 0.1
dropout_ET_us = 0.2
epochs_ET = 1000
epochs_ET_us = 900
batch_size_St = 360
batch_size_Rt = 256
hidden_units_St = 90
hidden_units_Rt = 10
dropout_St = 0.4
dropout_Rt = 0.4
epochs_St = 100
epochs_Rt = 300
batch_size_Q_ds = 256
hidden_units_Q_ds = 80   
dropout_Q_ds = 0.3
epochs_Q_ds = 900

### ET predictions at upstream 
reframed_ET_us = supervised(data_ET_us,[0,1,2,3],1)
print('Shape of supervised datas ET_us: ', np.shape(reframed_ET_us))
reframed_new_ET_us = reframed_ET_us[['Pptn_us(t)','PET_us(t)','SM_us_sim(t)','Pptn_us(t-1)','PET_us(t-1)','SM_us_sim(t-1)','ET_ratio_req_us(t)']]
reframed_new_ET_us[['Pptn_us(t-1)', 'PET_us(t-1)','SM_us_sim(t)']] = 0    
XYdata_ET_us = reframed_new_ET_us.values
yobs_train_ET_us = XYdata_ET_us[:Ntest, -n_outputs:]
yobs_test_ET_us = XYdata_ET_us[Ntest+1:, -n_outputs:]
print('shape of yobs_train_ET_us and yobs_test_ET_us is ', yobs_train_ET_us.shape, yobs_test_ET_us.shape)
print('min and max of yobs_test_ET_us', np.min(yobs_test_ET_us), np.max(yobs_test_ET_us))

## split into train and test datasets
## selective preprocessing of training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain_ET_us = scaler.fit_transform(XYdata_ET_us[:Ntest, :-1])
scaledXYtest_ET_us = scaler.transform(XYdata_ET_us[Ntest+1:, :-1]) 
scaledXYtrain_ET_final_us = np.append(scaledXYtrain_ET_us,yobs_train_ET_us, axis =1)
scaledXYtest_ET_final_us = np.append(scaledXYtest_ET_us ,yobs_test_ET_us, axis =1)
print('shape of scaledXYtrain_us and scaledXYtest_us is ', scaledXYtrain_ET_final_us.shape, scaledXYtest_ET_final_us.shape)

## split into input and outputs
train_X_ET_us, train_y_ET_us = scaledXYtrain_ET_final_us[:, :nobs_ET_us], scaledXYtrain_ET_final_us[:, -n_outputs:]
test_X_ET_us = scaledXYtest_ET_final_us[:, :nobs_ET_us]
PET_data_us = (df['PET_us']).values
PET_data_train_us = PET_data_us[:Ntest]
PET_data_test_us = PET_data_us[Ntest+2:]
ET_data_us = (df['ET_us']).values
ET_data_train_us = ET_data_us[:Ntest]
ET_data_test_us = ET_data_us[Ntest+2:]
print('shape of train_X_ET_us, train_y_ET_us, and test_X_ET_us: ', train_X_ET_us.shape, train_y_ET_us.shape, test_X_ET_us.shape)

## reshape input to be 3D [samples, timesteps, features]
train_X_ET_us = train_X_ET_us.reshape((train_X_ET_us.shape[0], ndays, ninputs_ET_us))
test_X_ET_us = test_X_ET_us.reshape((test_X_ET_us.shape[0], ndays, ninputs_ET_us))
print('shape of train_X_ET_us and test_X_ET_us in 3D: ', train_X_ET_us.shape, test_X_ET_us.shape)
input_shape=(train_X_ET_us.shape[1], train_X_ET_us.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_ET_us = Sequential()   
model_ET_us.add(LSTM(hidden_units_ET_us, input_shape=(train_X_ET_us.shape[1], train_X_ET_us.shape[2])))
model_ET_us.add(Dropout(dropout_ET_us))
model_ET_us.add(Dense(n_outputs, activation = 'sigmoid')) 
model_ET_us.compile(loss = 'mse',optimizer='adam') 
history_ET_us = model_ET_us.fit(train_X_ET_us, train_y_ET_us, batch_size = batch_size_ET_us, epochs= epochs_ET_us, shuffle=True,validation_split=0.2, verbose=0)
plt.plot(history_ET_us.history['loss'])
plt.plot(history_ET_us.history['val_loss'])
plt.title('model loss for ET/PET at u/s prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_ET_us = model_ET_us.predict(train_X_ET_us)
train_X_ET_us = train_X_ET_us.reshape((train_X_ET_us.shape[0], nobs_ET_us))
inv_yhat_ET_us = np.concatenate((train_X_ET_us, yhat_ET_us), axis=1)
NNytrain_ET_us = inv_yhat_ET_us[:,-n_outputs:]

## outputs in testing
yhat_test_us = model_ET_us.predict(test_X_ET_us)
test_X_ET_us = test_X_ET_us.reshape((test_X_ET_us.shape[0], nobs_ET_us))
inv_yhat_test_ET_us = np.concatenate((test_X_ET_us, yhat_test_us), axis=1)
NNytest_ET_us = inv_yhat_test_ET_us[:,-n_outputs]
mean_train_ET_us = np.array(NNytrain_ET_us)
mean_test_ET_us = np.array(NNytest_ET_us)

## evaluate outputs
ratio_ET_PET_great_than_1_us = np.sum(np.array((mean_test_ET_us.reshape(mean_test_ET_us.shape[0],))) > 1, axis=0)
print('ET not following constraint', ratio_ET_PET_great_than_1_us)
s1_ET_train_us = np.multiply(PET_data_train_us,(mean_train_ET_us.reshape(mean_train_ET_us.shape[0],)))
s1_ET_test_us = np.multiply(PET_data_test_us,(mean_test_ET_us.reshape(mean_test_ET_us.shape[0],)))
o1_ET_train_us = ET_data_train_us
o1_ET_test_us = ET_data_test_us
NS_ET_us = 1 - sum((s1_ET_test_us-o1_ET_test_us)**2)/sum((o1_ET_test_us-np.mean(o1_ET_test_us))**2)
PBIAS_ET_us = pc_bias(s1_ET_test_us,o1_ET_test_us)
RMSE_ET_us = rmse(s1_ET_test_us,o1_ET_test_us)
WB_ET_us = 1 - abs(1 - ((sum(s1_ET_test_us))/(sum(o1_ET_test_us))))
NS_ET_us_cal = 1 - sum((s1_ET_train_us-o1_ET_train_us)**2)/sum((o1_ET_train_us-np.mean(o1_ET_train_us))**2)
PBIAS_ET_us_cal = pc_bias(s1_ET_train_us,o1_ET_train_us)
RMSE_ET_us_cal = rmse(s1_ET_train_us,o1_ET_train_us)
WB_ET_us_cal = 1 - abs(1 - ((sum(s1_ET_train_us))/(sum(o1_ET_train_us))))

### ET predictions at downstream
reframed_ET = supervised(data_ET_us_ds,[0,1,2,3],1)
print('Shape of supervised datas ET: ', np.shape(reframed_ET))
reframed_new_ET = reframed_ET[['Pptn_us_ds(t)','PET_us_ds(t)','SM_us_ds_sim(t)','Pptn_us_ds(t-1)','PET_us_ds(t-1)','SM_us_ds_sim(t-1)','ET_ratio_req_us_ds(t)']]
reframed_new_ET[['Pptn_us_ds(t-1)', 'PET_us_ds(t-1)','SM_us_ds_sim(t)']] = 0  
XYdata_ET = reframed_new_ET.values
yobs_train_ET = XYdata_ET[:Ntest, -n_outputs:]
yobs_test_ET = XYdata_ET[Ntest+1:, -n_outputs:]
print('shape of yobs_train_ET and yobs_test_ET is ', yobs_train_ET.shape, yobs_test_ET.shape)
print('min and max of yobs_test_ET', np.min(yobs_test_ET), np.max(yobs_test_ET))

## split into train and test datasets
## selective preprocessing of training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain_ET = scaler.fit_transform(XYdata_ET[:Ntest, :-1])
scaledXYtest_ET = scaler.transform(XYdata_ET[Ntest+1:, :-1])
scaledXYtrain_ET_final = np.append(scaledXYtrain_ET,yobs_train_ET, axis =1)
scaledXYtest_ET_final = np.append(scaledXYtest_ET ,yobs_test_ET, axis =1)
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_ET_final.shape, scaledXYtest_ET_final.shape)

## split into input and outputs
train_X_ET, train_y_ET = scaledXYtrain_ET_final[:, :nobs_ET], scaledXYtrain_ET_final[:, -n_outputs:]
test_X_ET = scaledXYtest_ET_final[:, :nobs_ET]
PET_data = (df['PET_us_ds']).values
PET_data_train = PET_data[:Ntest]
PET_data_test = PET_data[Ntest+2:]
ET_data = (df['ET_us_ds']).values
ET_data_train = ET_data[:Ntest]
ET_data_test = ET_data[Ntest+2:]
print('shape of train_X_ET, train_y_ET, and test_X_ET: ', train_X_ET.shape, train_y_ET.shape, test_X_ET.shape)

## reshape input to be 3D [samples, timesteps, features]
train_X_ET = train_X_ET.reshape((train_X_ET.shape[0], ndays, ninputs_ET))
test_X_ET = test_X_ET.reshape((test_X_ET.shape[0], ndays, ninputs_ET))
print('shape of train_X_ET and test_X_ET in 3D: ', train_X_ET.shape, test_X_ET.shape)
input_shape=(train_X_ET.shape[1], train_X_ET.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_ET = Sequential()   
model_ET.add(LSTM(hidden_units_ET, input_shape=(train_X_ET.shape[1], train_X_ET.shape[2])))
model_ET.add(Dropout(dropout_ET))
model_ET.add(Dense(n_outputs, activation = 'sigmoid')) 
model_ET.compile(loss = 'mse',optimizer='adam')
history_ET_ds = model_ET.fit(train_X_ET, train_y_ET, batch_size = batch_size_ET, epochs=epochs_ET, shuffle=True,validation_split=0.2, verbose=0)
plt.plot(history_ET_ds.history['loss'])
plt.plot(history_ET_ds.history['val_loss'])
plt.title('model loss for ET/PET at d/s prediction')
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
print('shape of mc_train is', mean_train_ET.shape)
mean_test_ET = np.array(NNytest_ET)
print('shape of NN_ytest is', NNytest_ET.shape)

## evaluate outputs
ratio_ET_PET_great_than_1 = np.sum(np.array((mean_test_ET.reshape(mean_test_ET.shape[0],))) > 1, axis=0)
print('ET not following constraint', ratio_ET_PET_great_than_1)
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

### reservoir_storage (St) prediction
n_outputs = 1 
reframed_St = supervised(data_St,[0,1,2,3,4,5],2) 
print('Shape of supervised datasSt: ', np.shape(reframed_St))
reframed_new_St = reframed_St[['ET_us(t)','ET_us(t-1)','ET_us(t-2)','Pptn_us(t)','Pptn_us(t-1)','Pptn_us(t-2)','SM_us_sim(t)','SM_us_sim(t-1)','SM_us_sim(t-2)','GW_us_sim(t)','GW_us_sim(t-1)','GW_us_sim(t-2)','live_St_us(t)','live_St_us(t-1)','live_St_us(t-2)','St_ratio_req_us(t)']]
reframed_new_St[['Pptn_us(t-2)', 'ET_us(t-2)','live_St_us(t-2)','live_St_us(t)']] = 0 

## split into train and test datasets
XYdata_St = reframed_new_St.values
XYtrain_St = XYdata_St[2:Ntest, :]
XYtrain_St[:,0] = s1_ET_train_us[2:]#t
XYtrain_St[:,1] = s1_ET_train_us[1:-1]#t-1
XYtest_St = XYdata_St[Ntest+3:, :]
XYtest_St[:,0] = s1_ET_test_us[3:]#t
XYtest_St[:,1] = s1_ET_test_us[2:-1]#t-1
yobs_train_St = XYdata_St[2:Ntest, -n_outputs:]
yobs_test_St = XYdata_St[Ntest+3:, -n_outputs:]
print('shape of yobs_train_St and yobs_test_St is ', yobs_train_St.shape, yobs_test_St.shape)
print('min and max of yobs_test_St', np.min(yobs_test_St), np.max(yobs_test_St))
scaledXYtrain_St = XYtrain_St
scaledXYtest_St = XYtest_St
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_St.shape, scaledXYtest_St.shape)

## split into input and outputs
train_X_St, train_y_St = scaledXYtrain_St[:, :nobs_St], scaledXYtrain_St[:, -n_outputs:]
test_X_St = scaledXYtest_St[:, :nobs_St]
print('shape of train_X_St, train_y_St, and test_X_St: ', train_X_St.shape, train_y_St.shape, test_X_St.shape)

## reshape input to be 3D [samples, timesteps, features]
train_X_St = train_X_St.reshape((train_X_St.shape[0], ndays_St, ninputs_St))
test_X_St = test_X_St.reshape((test_X_St.shape[0], ndays_St, ninputs_St))
print('shape of train_X_St and test_X_St in 3D: ', train_X_St.shape, test_X_St.shape)
input_shape=(train_X_St.shape[1], train_X_St.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_St = Sequential()  
model_St.add(LSTM(hidden_units_St, input_shape=(train_X_St.shape[1], train_X_St.shape[2])))
model_St.add(Dropout(dropout_St))
model_St.add(Dense(n_outputs, activation = 'sigmoid'))    
model_St.compile(loss = 'mse',optimizer='adam') 
history_St = model_St.fit(train_X_St, train_y_St, batch_size = batch_size_St, epochs=epochs_St, shuffle=False,validation_split=0.2, verbose=0)
plt.plot(history_St.history['loss'])
plt.plot(history_St.history['val_loss'])
plt.title('model loss for St/Smax prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_St = model_St.predict(train_X_St)
train_X_St = train_X_St.reshape((train_X_St.shape[0], nobs_St))
inv_yhat_St = np.concatenate((train_X_St, yhat_St), axis=1)
NNytrain_St = inv_yhat_St[:,-n_outputs:]
train_X_St = train_X_St.reshape((train_X_St.shape[0], ndays_St, ninputs_St))

## outputs in testing
yhat_test_St = model_St.predict(test_X_St)
test_X_St = test_X_St.reshape((test_X_St.shape[0], nobs_St))
inv_yhat_test_St = np.concatenate((test_X_St, yhat_test_St), axis=1)
NNytest_St = inv_yhat_test_St[:,-n_outputs:]
test_X_St = test_X_St.reshape((test_X_St.shape[0], ndays_St, ninputs_St))
mean_train_St = np.array(NNytrain_St)
mean_test_St = np.array(NNytest_St)

## evaluate outputs
St_data = (df['live_St_us']).values
St_data_train =St_data[2:Ntest]
St_data_test = St_data[Ntest+5:]
train_output_St = mean_train_St[:,0]
test_output_St = mean_test_St[:,0]
s1_St_train = train_output_St*RS_max
s1_St_test = test_output_St*RS_max
ratio_St_RSmax_great_than_1 = np.sum(np.array(test_output_St) > 1, axis=0)
print('St not following constraint', ratio_St_RSmax_great_than_1)
o1_St_train = St_data_train
o1_St_test = St_data_test
NS_St = 1 - sum((s1_St_test-o1_St_test)**2)/sum((o1_St_test-np.mean(o1_St_test))**2)
PBIAS_St = pc_bias(s1_St_test,o1_St_test)
RMSE_St = rmse(s1_St_test,o1_St_test)
WB_St = 1 - abs(1 - ((sum(s1_St_test))/(sum(o1_St_test))))
NS_St_cal = 1 - sum((s1_St_train-o1_St_train)**2)/sum((o1_St_train-np.mean(o1_St_train))**2)
PBIAS_St_cal = pc_bias(s1_St_train,o1_St_train)
RMSE_St_cal = rmse(s1_St_train,o1_St_train)
WB_St_cal = 1 - abs(1 - ((sum(s1_St_train))/(sum(o1_St_train))))

### reservoir release (Rt) prediction
n_outputs = 1 
reframed_Rt = supervised(data_Rt,[0,1,2,3,4,5],2) 
print('Shape of supervised datasRt: ', np.shape(reframed_Rt))
reframed_new_Rt = reframed_Rt[['ET_us(t)','ET_us(t-1)','ET_us(t-2)','Pptn_us(t)','Pptn_us(t-1)','Pptn_us(t-2)','SM_us_sim(t)','SM_us_sim(t-1)','SM_us_sim(t-2)','GW_us_sim(t)','GW_us_sim(t-1)','GW_us_sim(t-2)','live_St_us(t-2)','live_St_us(t-1)','live_St_us(t)','Res_outflow(t)']]
reframed_new_Rt[['Pptn_us(t-2)', 'ET_us(t-2)','live_St_us(t-2)','live_St_us(t)']] = 0 

## split into train and test datasets
XYdata_Rt = reframed_new_Rt.values
XYtrain_Rt = XYdata_Rt[2:Ntest, :]
XYtrain_Rt[:,0] = s1_ET_train_us[2:]#t
XYtrain_Rt[:,1] = s1_ET_train_us[1:-1]#t-1
XYtest_Rt = XYdata_Rt[Ntest+3:, :]
XYtest_Rt[:,0] = s1_ET_test_us[3:]#t
XYtest_Rt[:,1] = s1_ET_test_us[2:-1]#t-1
yobs_train_Rt = XYdata_Rt[2:Ntest, -n_outputs:]
yobs_test_Rt = XYdata_Rt[Ntest+3:, -n_outputs:]
print('shape of yobs_train_Rt and yobs_test_Rt is ', yobs_train_Rt.shape, yobs_test_Rt.shape)
print('min and max of yobs_test_Rt', np.min(yobs_test_Rt), np.max(yobs_test_Rt))
scaledXYtrain_Rt = XYtrain_Rt
scaledXYtest_Rt = XYtest_Rt
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_Rt.shape, scaledXYtest_Rt.shape)

## split into input and outputs
train_X_Rt, train_y_Rt = scaledXYtrain_Rt[:, :nobs_Rt], scaledXYtrain_Rt[:, -n_outputs:]
test_X_Rt = scaledXYtest_Rt[:, :nobs_Rt]
print('shape of train_X_Rt, train_y_Rt, and test_X_Rt: ', train_X_Rt.shape, train_y_Rt.shape, test_X_Rt.shape)

## reshape input to be 3D [samples, timesteps, features]
train_X_Rt = train_X_Rt.reshape((train_X_Rt.shape[0], ndays_Rt, ninputs_Rt))
test_X_Rt = test_X_Rt.reshape((test_X_Rt.shape[0], ndays_Rt, ninputs_Rt))
print('shape of train_X_Rt and test_X_Rt in 3D: ', train_X_Rt.shape, test_X_Rt.shape)
input_shape=(train_X_Rt.shape[1], train_X_Rt.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_Rt = Sequential()  
model_Rt.add(LSTM(hidden_units_Rt, input_shape=(train_X_Rt.shape[1], train_X_Rt.shape[2])))
model_Rt.add(Dropout(dropout_Rt))
model_Rt.add(Dense(n_outputs, activation = 'relu'))    
model_Rt.compile(loss = 'mse',optimizer='adam') 
history_Rt = model_Rt.fit(train_X_Rt, train_y_Rt, batch_size = batch_size_Rt, epochs=epochs_Rt, shuffle=False,validation_split=0.2, verbose=0)
plt.plot(history_Rt.history['loss'])
plt.plot(history_Rt.history['val_loss'])
plt.title('model loss for Rt prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_Rt = model_Rt.predict(train_X_Rt)
train_X_Rt = train_X_Rt.reshape((train_X_Rt.shape[0], nobs_Rt))
inv_yhat_Rt = np.concatenate((train_X_Rt, yhat_Rt), axis=1)
NNytrain_Rt = inv_yhat_Rt[:,-n_outputs:]
train_X_Rt = train_X_Rt.reshape((train_X_Rt.shape[0], ndays_Rt, ninputs_Rt))

## outputs in testing
yhat_test_Rt = model_Rt.predict(test_X_Rt)
test_X_Rt = test_X_Rt.reshape((test_X_Rt.shape[0], nobs_Rt))
inv_yhat_test_Rt = np.concatenate((test_X_Rt, yhat_test_Rt), axis=1)
NNytest_Rt = inv_yhat_test_Rt[:,-n_outputs:]
test_X_Rt = test_X_Rt.reshape((test_X_Rt.shape[0], ndays_Rt, ninputs_Rt))
mean_train_Rt = np.array(NNytrain_Rt)
mean_test_Rt = np.array(NNytest_Rt)
 
## evaluate outputs
Rt_data = (df['Res_outflow']).values
Rt_data_train =Rt_data[2:Ntest]
Rt_data_test = Rt_data[Ntest+5:]
(s1_Rt_test,o1_Rt_test) = (mean_test_Rt[:,0], yobs_test_Rt[:,0])
NS_Rt = 1 - sum((s1_Rt_test-o1_Rt_test)**2)/sum((o1_Rt_test-np.mean(o1_Rt_test))**2)
PBIAS_Rt = pc_bias(s1_Rt_test,o1_Rt_test)
RMSE_Rt = rmse(s1_Rt_test,o1_Rt_test)
WB_Rt = 1 - abs(1 - ((sum(s1_Rt_test))/(sum(o1_Rt_test))))
(s1_Rt_train,o1_Rt_train) = (mean_train_Rt[:,0], yobs_train_Rt[:,0])
NS_Rt_cal = 1 - sum((s1_Rt_train-o1_Rt_train)**2)/sum((o1_Rt_train-np.mean(o1_Rt_train))**2)
PBIAS_Rt_cal = pc_bias(s1_Rt_train,o1_Rt_train)
RMSE_Rt_cal = rmse(s1_Rt_train,o1_Rt_train)
WB_Rt_cal = 1 - abs(1 - ((sum(s1_Rt_train))/(sum(o1_Rt_train))))

### Streamflow prediction at the downstream
n_outputs = 1 
reframed_Q = supervised(data_Q_ds,[0,1,2,3,4,5],3) 
print('Shape of supervised datasQ: ', np.shape(reframed_Q))
reframed_new_Q = reframed_Q[['ET_us_ds(t)','ET_us_ds(t-1)','ET_us_ds(t-2)','ET_us_ds(t-3)','Res_outflow(t)','Res_outflow(t-1)','Res_outflow(t-2)','Res_outflow(t-3)','Pptn_us_ds(t)','Pptn_us_ds(t-1)','Pptn_us_ds(t-2)','Pptn_us_ds(t-3)','SM_us_ds_sim(t)','SM_us_ds_sim(t-1)','SM_us_ds_sim(t-2)','SM_us_ds_sim(t-3)','GW_us_ds_sim(t)','GW_us_ds_sim(t-1)','GW_us_ds_sim(t-2)','GW_us_ds_sim(t-3)','Q_ds(t)']]
reframed_new_Q[['Pptn_us_ds(t-3)', 'ET_us_ds(t-3)','Res_outflow(t-3)']] = 0 

## split into train and test datasets
XYdata_Q = reframed_new_Q.values
XYtrain_Q = XYdata_Q[5:Ntest, :]
XYtrain_Q[:,0] = s1_ET_train[5:]#t
XYtrain_Q[:,1] = s1_ET_train[4:-1]#t-1
XYtrain_Q[:,2] = s1_ET_train[3:-2]#t-2
XYtrain_Q[:,4] = s1_Rt_train[3:]#t
XYtrain_Q[:,5] = s1_Rt_train[2:-1]#t-1
XYtrain_Q[:,6] = s1_Rt_train[1:-2]#t-2
XYtest_Q = XYdata_Q[Ntest+5:, :]
XYtest_Q[:,0] = s1_ET_test[6:]#t
XYtest_Q[:,1] = s1_ET_test[5:-1]#t-1
XYtest_Q[:,2] = s1_ET_test[4:-2]#t-2
XYtest_Q[:,4] = s1_Rt_test[3:]#t
XYtest_Q[:,5] = s1_Rt_test[2:-1]#t-1
XYtest_Q[:,6] = s1_Rt_test[1:-2]#t-2
yobs_train_Q = XYdata_Q[5:Ntest, -n_outputs:]
yobs_test_Q = XYdata_Q[Ntest+5:, -n_outputs:]
print('shape of yobs_train_Q and yobs_test_Q is ', yobs_train_Q.shape, yobs_test_Q.shape)
print('min and max of yobs_test_Q', np.min(yobs_test_Q), np.max(yobs_test_Q))
scaledXYtrain_Q = XYtrain_Q
scaledXYtest_Q = XYtest_Q
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain_Q.shape, scaledXYtest_Q.shape)

## split into input and outputs
train_X_Q, train_y_Q = scaledXYtrain_Q[:, :nobs_Q], scaledXYtrain_Q[:, -n_outputs:]
test_X_Q = scaledXYtest_Q[:, :nobs_Q]
print('shape of train_X_Q, train_y_Q, and test_X_Q: ', train_X_Q.shape, train_y_Q.shape, test_X_Q.shape)

## reshape input to be 3D [samples, timesteps, features]
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], ndays_Q, ninputs_Q))
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], ndays_Q, ninputs_Q))
print('shape of train_X_Q and test_X_Q in 3D: ', train_X_Q.shape, test_X_Q.shape)
input_shape=(train_X_Q.shape[1], train_X_Q.shape[2])

## define and fit LSTM model
np.random.seed(1234)
tf.random.set_seed(1234)
model_Q = Sequential()  
model_Q.add(LSTM(hidden_units_Q_ds, input_shape=(train_X_Q.shape[1], train_X_Q.shape[2])))
model_Q.add(Dropout(dropout_Q_ds))
model_Q.add(Dense(n_outputs, activation = 'relu'))    
model_Q.compile(loss = 'mse',optimizer='adam')
history_Q_ds = model_Q.fit(train_X_Q, train_y_Q, batch_size = batch_size_Q_ds, epochs=epochs_Q_ds, shuffle=False,validation_split=0.2, verbose=0)
plt.plot(history_Q_ds.history['loss'])
plt.plot(history_Q_ds.history['val_loss'])
plt.title('model loss for Q at d/s prediction')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

## outputs in training
yhat_Q = model_Q.predict(train_X_Q)
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], nobs_Q))
inv_yhat_Q = np.concatenate((train_X_Q, yhat_Q), axis=1)
NNytrain_Q = inv_yhat_Q[:,-n_outputs:]
train_X_Q = train_X_Q.reshape((train_X_Q.shape[0], ndays_Q, ninputs_Q))

## outputs in testing
yhat_test_Q = model_Q.predict(test_X_Q)
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], nobs_Q))
inv_yhat_test_Q = np.concatenate((test_X_Q, yhat_test_Q), axis=1)
NNytest_Q = inv_yhat_test_Q[:,-n_outputs:]
test_X_Q = test_X_Q.reshape((test_X_Q.shape[0], ndays_Q, ninputs_Q))
mean_train_Q = np.array(NNytrain_Q)
mean_test_Q = np.array(NNytest_Q)

## evaluate outputs
(s1_Q_test,o1_Q_test) = (mean_test_Q[:,0], yobs_test_Q[:,0])
NS_Q = 1 - sum((s1_Q_test-o1_Q_test)**2)/sum((o1_Q_test-np.mean(o1_Q_test))**2)
PBIAS_Q = pc_bias(s1_Q_test,o1_Q_test)
RMSE_Q = rmse(s1_Q_test,o1_Q_test)
WB_Q = 1 - abs(1 - ((sum(s1_Q_test))/(sum(o1_Q_test))))
(s_train, o_train) = (mean_train_Q[:,0], yobs_train_Q[:,0])
NS_Q_cal = 1 - sum((s_train-o_train)**2)/sum((o_train-np.mean(o_train))**2)
PBIAS_Q_cal = pc_bias(s_train,o_train)
RMSE_Q_cal = rmse(s_train,o_train)
WB_Q_cal = 1 - abs(1 - ((sum(s_train))/(sum(o_train))))
op = df[['Date','Pptn_us','Pptn_us_ds','Pptn_ds','ET_us','ET_us_ds','ET_ds','live_St_us','Res_outflow','Q_ds']][Ntest+8:]
op1 = op.reset_index(drop=True)
op1['ET_us_piml_sim'] = s1_ET_test_us[6:]
op1['ET_us_ds_piml_sim'] = s1_ET_test[6:]
op1['St_piml_sim'] = mean_test_St[3:,0]*RS_max
op1['Rt_piml_sim'] = mean_test_Rt[3:,0]
op1['Q_ds_piml_sim'] = mean_test_Q[:,0]
op1.to_csv('brady_catchment_semi_distributed_PIML_with_reservoir_testing_output.csv')

perform = perform.append({ 

"NSE_cal_ET_us" : NS_ET_us_cal, 
"PBIAS_cal_ET_us" : PBIAS_ET_us_cal, 
"RMSE_cal_ET_us" : RMSE_ET_us_cal, 
"WB_cal_ET_us" : WB_ET_us_cal, 
"NSE_val_ET_us" : NS_ET_us, 
"PBIAS_val_ET_us" : PBIAS_ET_us, 
"RMSE_val_ET_us" : RMSE_ET_us, 
"WB_val_ET_us" : WB_ET_us, 
'constraint_violation_ratio_us' : ratio_ET_PET_great_than_1_us, 

"NSE_cal_ET" : NS_ET_cal, 
"PBIAS_cal_ET" : PBIAS_ET_cal, 
"RMSE_cal_ET" : RMSE_ET_cal, 
"WB_cal_ET" : WB_ET_cal, 
"NSE_val_ET" : NS_ET, 
"PBIAS_val_ET" : PBIAS_ET, 
"RMSE_val_ET" : RMSE_ET, 
"WB_val_ET" : WB_ET, 
'constraint_violation_ratio' : ratio_ET_PET_great_than_1, 

"NSE_cal_St" : NS_St_cal, 
"PBIAS_cal_St" : PBIAS_St_cal, 
"RMSE_cal_St" : RMSE_St_cal, 
"WB_cal_St" : WB_St_cal, 
"NSE_val_St" : NS_St, 
"PBIAS_val_St" : PBIAS_St, 
"RMSE_val_St" : RMSE_St, 
"WB_val_St" : WB_St, 
'constraint_violation_ratio_St':ratio_St_RSmax_great_than_1, 

"NSE_cal_Rt" : NS_Rt_cal, 
"PBIAS_cal_Rt" : PBIAS_Rt_cal, 
"RMSE_cal_Rt" : RMSE_Rt_cal, 
"WB_cal_Rt" : WB_Rt_cal, 
"NSE_val_Rt" : NS_Rt, 
"PBIAS_val_Rt" : PBIAS_Rt, 
"RMSE_val_Rt" : RMSE_Rt, 
"WB_val_Rt" : WB_Rt,

"NSE_cal_Q_ds" : NS_Q_cal, 
"PBIAS_cal_Q_ds" : PBIAS_Q_cal, 
"RMSE_cal_Q_ds" : RMSE_Q_cal, 
"WB_cal_Q_ds" : WB_Q_cal, 
"NSE_val_Q_ds" : NS_Q, 
"PBIAS_val_Q_ds" : PBIAS_Q, 
"RMSE_val_Q_ds" : RMSE_Q, 
"WB_val_Q_ds" : WB_Q},ignore_index = True) 
print(perform)

