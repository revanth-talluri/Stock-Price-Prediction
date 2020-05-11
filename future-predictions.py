# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:00:22 2020
@author: revan
"""

#Linear algebra
import numpy as np 


# data processing
import pandas as pd 
pd.set_option('display.max_columns', 15)

# data visualization
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
    
    return predict(params,new_data,scaler,model)

if __name__ == '__main__':
    
    #company  = input('Name of the company: ')
    #file_name = input('Name of the CSV data file: ')
    
    #Readind the data and changing it into pandas dataframe
    #data_df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    data_df = pd.read_csv('Google15-20.csv', index_col='Date', parse_dates=True)
    
    #Defining the initial parameters of the model    
    params = get_params() 
    params.update({'future_days':30})
    
    #this 'result' contains the predicted close prices for the next 30 days (future_days)
    result = run(data_df, params)    
    
    result = np.reshape(result,(result.shape[0]))
    result = list(result)    
    
    dates = pd.date_range('2020-04-01', periods=params['future_days'], freq='D')
    result_df = pd.DataFrame([dates, result]).transpose()
    result_df.columns = ['Date','Predicted Close Price']
  
    result_df.to_csv (r'C:\Users\shaik\Downloads\Revanth\Project-Google\future_prices.csv', 
                      index = True, header=True)
    
