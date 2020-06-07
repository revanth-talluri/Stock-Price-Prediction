# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 02:22:19 2020
@author: revan
"""

#importing libraries
import numpy as np 
import os, sys
from pathlib import Path, PureWindowsPath

#getting data
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

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



#predicting the 'future prices' for the next 30 days
def predict(params,new_data,scaler,model):
    
    #pred_input is the first set of values used for future-prediction
    pred_input = new_data[-params['offset']:].values
    pred_input = pred_input.reshape(-1,1)
    pred_input = scaler.transform(pred_input)
    
    #y_hat is used to store the predicted-future close price values
    y_hat  = []
    pred_input = pred_input.tolist()
    
    future_days = params['future_days'] #No. of days predicting into the future
    for _ in range(0,future_days):
        X_pred = []
        X_pred.append(pred_input)
        X_pred = np.array(X_pred)
        X_pred = np.reshape(X_pred, (X_pred.shape[0],X_pred.shape[1],1)) 
        output = model.predict(X_pred)
        y_hat.append(output[0,0])
        
        #popping the first element and adding the new predicted value at the end 
        #and using this new pred_input list for the next day prediction
        pred_input.pop(0)
        pred_input.append([output[0,0]])        
    
    y_hat = np.array(y_hat)
    y_hat = np.reshape(y_hat, (y_hat.shape[0],1))    
    y_hat = scaler.inverse_transform(y_hat)
    
    return y_hat  
   

def run(data_df, params, model):    
    
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
    
    return predict(params,new_data,scaler,model)


#'main' program
if __name__ == '__main__':
    
    #data_df = pd.read_csv('GOOG_2015-04-01_2020-03-31.csv', index_col='Date', parse_dates=True)
    #using pandas_datareader library to get the data from Yahoo-Finance
    start_date = datetime(2015, 4, 2)
    end_date   = datetime(2020, 3, 31)
    ticker = 'GOOG'
    
    data_df = pdr.get_data_yahoo(tickers=ticker, start=start_date, end=end_date)
    
    #loading the already built model for prediction
    model = load_model('model.h5')
    
    #summarize model
    model.summary()
    
    #Defining the initial parameters of the model    
    params = get_params() 
    params.update({'future_days':30})
    
    #this 'result' contains the predicted close prices for the next 30 days (future_days)
    result = run(data_df, params, model)    
    
    result = np.reshape(result,(result.shape[0]))
    result = list(result)    
    
    #Prediction for the month of April
    dates = pd.date_range(end_date.date() + timedelta(days=1), periods=params['future_days'], freq='D')
    result_df = pd.DataFrame([dates, result]).transpose()
    result_df.columns = ['Date','Predicted Close Price']
    
    BASE_PATH = os.getcwd()
    script_folder   = Path(os.getcwd())
    params_to_store = script_folder / 'future_prices.csv'
  
    result_df.to_csv (params_to_store, index = True, header=True)
    
