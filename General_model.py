# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:06:26 2020

@author: revan
"""

#Linear algebra
import numpy as np 
import random


# data processing
import pandas as pd 
pd.set_option('display.max_columns', 15)

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

#for normalizing data
from sklearn.preprocessing import MinMaxScaler

#avoid warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from sklearn.metrics import r2_score


#Setting the seed
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1000)



def build_model(train,valid,new_data,scaler,params,
                scaled_data_train,scaled_data_valid):
    
    
    x_train, y_train = [], []
    for i in range(params['offset'],len(train)):
        x_train.append(scaled_data_train[i-params['offset']:i,0])
        y_train.append(scaled_data_train[i,0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    #scale = MinMaxScaler(feature_range=(0,1))
    #scale.min_, scale.scale_ = scaler.min_, scaler.scale_
    
    inputs = new_data[len(new_data) - len(valid) - params['offset']:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    
    X_test, Y_test = [], []
    for i in range(params['offset'],inputs.shape[0]):
        X_test.append(inputs[i-params['offset']:i,0])
        Y_test.append(inputs[i,0])
        
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    
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
    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        verbose=1, validation_data=[X_test, Y_test])
    
    return model, history, X_test

def get_accuracy(train,valid,new_data,tl, 
                 scaler,model,X_test):

    
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    train = new_data[:tl]
    valid = new_data[tl:]
    valid['Predictions'] = closing_price
    
    #for plotting
    plt.figure(figsize=(16,8))
    plt.plot(train['Close'])
    plt.plot(valid['Close'], label='Actual Close Price')
    plt.plot(valid['Predictions'] , label='Predicted Close Price')
    plt.legend()
    
    #RMS error
    rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))
    
    #R-squared
    y_true = valid['Close']
    y_pred = valid['Predictions']
    r = r2_score(y_true, y_pred)
    
    return rms[0], r

  
   
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
    
    frac = 0.8
    tl = int(len(new_data)*frac)
    
    dataset = new_data.values
    train = dataset[0:tl,:]
    valid = dataset[tl:,:]
    
    #Normalizing the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    scaled_data_train = scaler.transform(train)
    scaled_data_valid = scaler.transform(valid)

    
    model, history, X_test = build_model(train,valid,new_data,scaler,params,
                                         scaled_data_train,scaled_data_valid)
    
    rms, r = get_accuracy(train,valid,new_data,tl,
                          scaler,model,X_test)
    
    #Changing params to dataframe to store all the data
    params_items = params.items()
    params_list  = list(params_items)    
    params_df = pd.DataFrame(params_list, index=params.keys())
    
    #Adding errors to the above dataframe    
    errors = {'RMS': rms,
              'R-square':r}
    errors_items = errors.items()
    errors_list  = list(errors_items)    
    errors_df = pd.DataFrame(errors_list, index=errors.keys())
    
    result_df = pd.concat([params_df,errors_df])
    result_df = result_df.drop([0], axis=1)
    
    return result_df



if __name__ == '__main__':
    
    #company  = input('Name of the company: ')
    #file_name = input('Name of the CSV data file: ')
    
    #Readind the data and changing it into pandas dataframe
    #data_df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    data_df = pd.read_csv('Google15-20.csv', index_col='Date', parse_dates=True)
    
    #Defining the initial parameters of the model    
    params = {'offset':60,
              'units_1':32,
              'drop_rate_1':0,
              'units_2':32,
              'drop_rate_2':0,
              'batch_size':5,
              'epochs':10}
    

    result_df = run(data_df, params)
    
    print(result_df)
    

    
 
    