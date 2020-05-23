# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:52:56 2020
@author: revanth
"""

#linear algebra
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

#importing libraries for LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Flatten

#For Statistics
from sklearn.metrics import r2_score

#avoid warnings
import warnings
warnings.filterwarnings('ignore')

#Setting the seed
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1000)
#tf.set_random_seed(1000)

#Functions from other files
from General_model import build_model, get_accuracy

    
def run(data_df, params):
    
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
    
    plt.figure(figsize=(16,8))
    plt.plot(history.history['loss'], label='Train set loss')
    plt.plot(history.history['val_loss'], label='Test set loss')
    plt.title('Checking the model fit')
    plt.legend()
    plt.savefig('model_fit.png')
    
    train_loss = history.history['loss']
    test_loss  = history.history['val_loss']
    
    rms, r = get_accuracy(train,valid,new_data,tl,
                          scaler,model,X_test)
    
    #Adding errors to the above dataframe    
    errors = {'RMS': rms,
              'R-square':r}
    errors_items = errors.items()
    errors_list  = list(errors_items)
    
    errors_df = pd.DataFrame(errors_list)
    errors_df.columns = ['Attribute','Value']
    
    return errors_df, train_loss, test_loss

#get the parameters from here
#we can change the parameters directly here
def get_params():
    
        params = {'offset':60,
              'units_1':64,
              'drop_rate_1':0,
              'units_2':32,
              'drop_rate_2':0,
              'batch_size':50,
              'epochs':100}
    
        return params

#'main' program
if __name__ == '__main__':
    
    #company  = input('Name of the company: ')
    #file_name = input('Name of the CSV data file: ')
    
    #Readind the data and changing it into pandas dataframe
    #data_df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    data_df = pd.read_csv('Google15-20.csv', index_col='Date', parse_dates=True)
    
    #Defining the initial parameters of the model    
    params = get_params()
    errors_df, train_loss, test_loss = run(data_df, params)
    
    print(errors_df)
