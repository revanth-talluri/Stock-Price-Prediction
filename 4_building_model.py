# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:00:22 2020
@author: revan
"""

#importing libraries
import numpy as np 
import os, sys
from pathlib import Path, PureWindowsPath

#getting data
from pandas_datareader import data as pdr
from datetime import datetime

#data processing
import pandas as pd 
pd.set_option('display.max_columns', 25)

#data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#avoid warnings
import warnings
warnings.filterwarnings('ignore')

#importing RNN libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

#DateTime
from datetime import datetime

#Setting the seed
import random
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1000)
#tf.set_random_seed(1000)

#Functions from other files
from check_overfit import get_params


#building the model for the price prediction
def build_model(train,params,scaled_data_train):    
    
    x_train, y_train = [], []
    for i in range(params['offset'],len(train)):
        x_train.append(scaled_data_train[i-params['offset']:i,0])
        y_train.append(scaled_data_train[i,0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))    
    
    #create and fit the LSTM network
    if params['units_2'] != 0:
        
        model = Sequential()
        model.add(LSTM(units=params['units_1'], return_sequences=True, 
                       input_shape=(x_train.shape[1],1)))
        model.add(Dropout(rate=params['drop_rate_1']))
        model.add(LSTM(units=params['units_2']))
        model.add(Dropout(rate=params['drop_rate_2']))
        model.add(Dense(1)) 
        
    else:
        
        model = Sequential()
        model.add(LSTM(units=params['units_1'], return_sequences=False, 
                       input_shape=(x_train.shape[1],1)))
        model.add(Dropout(rate=params['drop_rate_1']))
        model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=params['epochs'], 
                        batch_size=params['batch_size'], verbose=1)
    
    return model, history
   

def run(data_df, params):    
    
    #Plot the data and check if there are any unexpected anamolies(sudden spikes or dips)
    plt.figure(figsize=(16,8))
    plt.plot(data_df['Close'], label='Close Price history')
    plt.title('Close Price History')
    
    new_data = pd.DataFrame(index=range(0,len(data_df)),columns=['Date', 'Close'])
    for i in range(0,len(data_df)):
        new_data['Date'][i]  = data_df.index[i]
        new_data['Close'][i] = data_df['Close'][i]
        
    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    
    tl = len(new_data)
    
    dataset = new_data.values
    train   = dataset[0:tl,:]
    
    #Normalizing the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    scaled_data_train = scaler.transform(train)
    
    model, history = build_model(train,params,scaled_data_train)
    
    #return predict(params,new_data,scaler,model)
    return model


#'main' program
if __name__ == '__main__':
    
    #data_df = pd.read_csv('GOOG_2015-04-01_2020-03-31.csv', index_col='Date', parse_dates=True)
    #using pandas_datareader library to get the data from Yahoo-Finance
    start_date = datetime(2015, 4, 2)
    end_date   = datetime(2020, 3, 31)
    ticker = 'GOOG'
    
    data_df = pdr.get_data_yahoo(tickers=ticker, start=start_date, end=end_date)
    
    #Defining the initial parameters of the model    
    params = get_params() 
    
    #this 'result' contains the predicted close prices for the next 30 days (future_days)
    #here our aim is to build model that predicts the future prices
    #this model will be saved for future use
    model = run(data_df, params)    
    
    BASE_PATH = os.getcwd()
    script_folder = Path(os.getcwd())
    path = script_folder / 'model.h5'
    
    model.save(path)
    print("'model' is saved in the current directory for future use")
    
