## 2LSTM_O3_final1.py
##Ruqi Yang

print('load module ...')
import numpy as np
from numpy import concatenate
from math import sqrt

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ReduceLROnPlateau
import keras
from scipy import stats
from pdb import set_trace
import glob

nhid = 1

def rollroll(N, Nroll):
        aa = np.arange(N)
        temp = np.arange(N - Nroll + 1)
        A, B = np.meshgrid(aa, temp)
        T = (A - B)[:, -Nroll:]
        return T[::-1,:].flatten().copy()

def make_3D_data(nline,backTime): 
    df_data = pd.read_csv("../Orig/CH4_exp.csv",names=["date", "time(UTC)", "CH4", "CH4_true","T","P","RH"], header=0)

    if nline > 0:
      df_data = df_data.dropna(how='any').head(nline)
    if nline < 0:
      df_data = df_data.dropna(how='any').tail(abs(nline))
    df_data["utc_time"] = df_data['date'] + df_data['time(UTC)']
    df_data.set_index(["utc_time"], inplace=True)


    '''
    df_features = df_data[['CH4', 'T', 'P', 'RH']]
    df_features['CH4'] = df_features['CH4'] / np.nanmax(df_features['CH4'])
    df_features['T'] = df_features['T'] / np.nanmax(df_features['T'])
    df_features['P'] = df_features['P'] / np.nanmax(df_features['P'])
    df_features['RH'] = df_features['RH'] / np.nanmax(df_features['RH'])
    '''
    df_features = df_data[['CH4']]
    df_features['CH4'] = df_features['CH4'] / np.nanmax(df_features['CH4'])
    df_label = df_data['CH4_true']


    #- Standardization for features, label will be scaled after 
    arr_features_values = df_features.values
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    arr_features_scaled_values = scaler_features.fit_transform(arr_features_values)
    df_scaled_features = pd.DataFrame(arr_features_scaled_values,columns = df_features.columns)
    df_scaled_features.set_index(df_data.index, inplace=True)
    #df_data_temp: features were scaled, label wasn't
    df_data_temp = pd.concat([df_label, df_scaled_features], axis=1) # (759,20)
    #print(df_data_temp.describe())

    #- make data for LSTM model
    #backTime = 24 # can be 2,3,4 ... 24,25, ...
    #- N = total sample, Nroll = backtime(hours)
    #- This function returns sequence like: 0,1,2,3, 1,2,3,4, 2,3,4,5
    
    #- sample: a=df[df_f].values[[0,1,2,3,1,2,3,4,2,3,4,5]].reshape(3,4,len(df_f))
    #- For data in one station
    roll_list = rollroll(len(df_data_temp), backTime)
    Nsample_orig = len(df_data_temp)
    Nfeature = len(df_scaled_features.keys())
    array_feature_rolled_3D = df_scaled_features.values[roll_list].reshape(Nsample_orig - backTime + 1, backTime, Nfeature)
    array_label_rolled_not_scaled = df_data_temp.CH4_true.values[backTime-1:]

    #- Standardization for label:
    scaler_label = MinMaxScaler(feature_range=(0, 1))
    array_label_rolled = scaler_label.fit_transform(array_label_rolled_not_scaled.reshape(-1, 1))

    #- (757, 3, 19) (759, 1)
    return array_feature_rolled_3D, array_label_rolled, scaler_label

if __name__ == "__main__":

    backTime = 1 
    model_name = 'lstm_ch4.h5'

    
    train_x, train_label, scaler_test_label = make_3D_data(100000,backTime)
    test_x, test_label, scaler_test_label = make_3D_data(-10000,backTime)
    print(np.shape(train_x),np.shape(train_label))
    print(np.shape(test_x),np.shape(test_label))

    #- Build model
    model = Sequential()
    model.add(LSTM(nhid, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences = True))
    #model.add(LSTM(nhid, input_shape=(train_x.shape[1], nhid), return_sequences = True))
    model.add(LSTM(nhid, input_shape=(train_x.shape[1], nhid)))
    model.add(Dense(1)) # fully connected layer
    model.compile(loss='mae', optimizer='adam') #can try loss='mse'

    #- train model
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 1, verbose=0, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
    history = model.fit(train_x, train_label, epochs=5, batch_size=1, validation_data=(test_x, test_label),callbacks=[reduce_lr], verbose=2, shuffle=True)

    #- make prediction
    
    Predicted_label = model.predict(test_x)

    test_label = test_label.reshape(test_label.shape[0])
    Predicted_label = Predicted_label.reshape(Predicted_label.shape[0])

    r2 = np.corrcoef(Predicted_label, test_label)[0][1] **2
    rmse = sqrt(mean_squared_error(Predicted_label, test_label))

    '''
    #- plot loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    test_label = scaler_test_label.inverse_transform(test_label)
    Predicted_label = scaler_test_label.inverse_transform(Predicted_label) #?

    test_label = test_label.reshape(test_label.shape[0])
    Predicted_label = Predicted_label.reshape(Predicted_label.shape[0])

    rmse = sqrt(mean_squared_error(Predicted_label, test_label))
    r2 = np.corrcoef(Predicted_label, test_label)[0][1] **2
    '''
    print('Test RMSE: %.3f' % rmse)
    print('r^2: %.3f' %(r2))
    model.save(model_name)

    #load model:
    #aaa = keras.models.load_model('lstm.h5')

    plt.plot(test_label.reshape(test_label.shape[0]), label='predicted')
    plt.plot(Predicted_label.reshape(Predicted_label.shape[0]), label='true')
    plt.legend()
    plt.show()



