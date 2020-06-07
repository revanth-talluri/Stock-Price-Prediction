# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:56:05 2020
@author: revanth
"""

#importing libraries
import numpy as np 
import csv 
import os, sys
from pathlib import Path, PureWindowsPath

#getting data
from pandas_datareader import data as pdr
from datetime import datetime

#data processing
import pandas as pd 
pd.set_option('display.max_columns', 15)

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

#For LSTM RNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Flatten

#For Statistics
from sklearn.metrics import r2_score

#For Bayesian Optimization
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

#Setting the seed
import random
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1000)
#tf.set_random_seed(1000)


#getting the 'train' and 'test' inputs for the nueral network
def data(batch_size, offset, scaled_data_train, scaled_data_valid):

    #function that returns data to be fed into objective function and 
    #model is trained on it subsequently
    global data_df

    BATCH_SIZE = batch_size
    TIME_STEPS = offset

    global train
    global valid
    global new_data
    
    #creating the training set in the required format
    x_train, y_train = [], []
    for i in range(TIME_STEPS,len(train)):
        x_train.append(scaled_data_train[i-TIME_STEPS:i,0])
        y_train.append(scaled_data_train[i,0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    #creating a new dataframe which will be used to create the test set
    inputs = new_data[len(new_data) - len(valid) - TIME_STEPS:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test, Y_test = [], []
    for i in range(TIME_STEPS,inputs.shape[0]):
        X_test.append(inputs[i-TIME_STEPS:i,0])
        Y_test.append(inputs[i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    return x_train, y_train, X_test, Y_test

#defining our search space, the parameters over which the model will run 
#and give the best possible set of parameters
search_space = {
    'batch_size': hp.choice('batch_size', [1,5,10]),
    'offset': hp.choice('offset', [30,45,60,90]),
    'lstm1_nodes': hp.choice('lstm1_nodes', [32,64,100]),
    'lstm1_dropouts': hp.uniform('lstm1_dropouts',0,0.5),
    'lstm_layers': hp.choice('lstm_layers',[
        {
            'layers':'one', 
        },
        {
            'layers':'two',
            'lstm2_nodes': hp.choice('lstm2_nodes', [32,64,100]),
            'lstm2_dropouts': hp.uniform('lstm2_dropouts',0,0.5)  
        }
        ]),
    "epochs": hp.choice('epochs', [1,5,10,20])
}


#creating the model for the Bayesian Optimzation
def create_model_hypopt(params):
    
    #This method is called for each combination of parameter set to train the model 
    #and validate it against validation datato see all the results, from which 
    #best can be selected.
    print("Trying params:",params)
    batch_size = params["batch_size"]
    offset = params["offset"]

    x_train, y_train, X_test, Y_test = data(batch_size, offset,
                                            scaled_data_train, scaled_data_valid)
    
    model = Sequential()
    model.add(LSTM(params["lstm1_nodes"], input_shape=(x_train.shape[1],1),return_sequences=True,
                   dropout=params["lstm1_dropouts"], recurrent_dropout=params["lstm1_dropouts"]))
    if params["lstm_layers"]["layers"] == "two":
        model.add(LSTM(params["lstm_layers"]["lstm2_nodes"], 
                       dropout=params["lstm_layers"]["lstm2_dropouts"]))

    else:
        model.add(Flatten())
    
    model.add(Dense(1))

    epochs = params["epochs"]

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=epochs, verbose=2,
                        batch_size=batch_size, validation_data=[X_test, Y_test])
    
    val_error = np.amin(history.history['val_loss']) 
    print('Best validation error of epoch:', val_error)
    
    closing_price = model.predict(X_test)
    #closing_price = scaler.inverse_transform(closing_price)
    
    rms=np.sqrt(np.mean(np.power((scaled_data_valid-closing_price),2)))
    print('The RMS error is {}'.format(rms))
    
    return {'loss': rms, 'status': STATUS_OK, 'model': model}  


#'main' program
if __name__ == '__main__':

    #data_df = pd.read_csv('GOOG_2015-04-01_2020-03-31.csv', index_col='Date', parse_dates=True)
    #using pandas_datareader library to get the data from Yahoo-Finance
    start_date = datetime(2015, 4, 1)
    end_date   = datetime(2020, 3, 31)
    ticker = 'GOOG'
    
    data_df = pdr.get_data_yahoo(tickers=ticker, start=start_date, end=end_date)
    
    #In our model, we will try to predict the future close price of a stock using only the past
    #close prices of that particular stock. So let's a create a new dataframe with only the
    #'Date' and 'Close' price columns
    new_data = pd.DataFrame(index=range(0,len(data_df)),columns=['Date', 'Close'])
    for i in range(0,len(data_df)):
        new_data['Date'][i]  = data_df.index[i]
        new_data['Close'][i] = data_df['Close'][i]
        
    #setting 'Date' column as index and dropping the original column
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    
    #creating train and test sets
    dataset = new_data.values
    
    #80% of the data is used as training set and 20% as test set
    #'test set' here is referred to as 'validatation set'
    frac=0.8
    tl = int(len(dataset)*frac)
    train = dataset[0:tl,:]
    valid = dataset[tl:,:]
    
    #Normalizing the data. Here we fit the data only on train set, because we 
    #don't want to involve the test set here and skew our model.
    #If the test set is also used, model will become biased
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)
    scaled_data_train = scaler.transform(train)
    scaled_data_valid = scaler.transform(valid)    
    
    #Set of parameters to be searched 
    vals = {'batch_size':[1,5,10],
            'epochs':[1,5,10,20],
            'lstm1_nodes':[32,64,100],
            'lstm2_nodes':[32,64,100],
            'lstm_layers':[1,2],
            'offset':[30,45,60,90]}
    
    # Trails object let's you return and store extra information from objective function, which
    # can be analysed later. Check "trails.trails" which returns all the list of dictionaries 
    trials = Trials()
    best   = fmin(create_model_hypopt,
                  space=search_space,
                  algo=tpe.suggest,  # type random.suggest to select param values randomly
                  max_evals=100,     # max number of evaluations you want to do on objective function
                  trials=trials)  
    
    if len(best)>6:
        best_params = {'best_bs':vals['batch_size'][best['batch_size']],
                       'best_ts':vals['offset'][best['offset']],
                       'best_epochs':vals['epochs'][best['epochs']],
                       'best_nodes1':vals['lstm1_nodes'][best['lstm1_nodes']],
                       'best_nodes2':vals['lstm2_nodes'][best['lstm2_nodes']],
                       'best_layers':vals['lstm_layers'][best['lstm_layers']],
                       'best_dropout1':best['lstm1_dropouts'],
                       'best_dropout2':best['lstm2_dropouts']}
        
    else:
        best_params = {'best_bs':vals['batch_size'][best['batch_size']],
                       'best_ts':vals['offset'][best['offset']],
                       'best_epochs':vals['epochs'][best['epochs']],
                       'best_nodes1':vals['lstm1_nodes'][best['lstm1_nodes']],
                       'best_layers':vals['lstm_layers'][best['lstm_layers']],
                       'best_dropout1':best['lstm1_dropouts']}
    
    #convertng the params in dictionary to dataframe
    params_items = best_params.items()
    params_list  = list(params_items)
    
    BASE_PATH = os.getcwd()
    script_folder  = Path(os.getcwd())
    params_to_store = script_folder / 'best_params.csv'
    
    params_df = pd.DataFrame(params_list)
    params_df.to_csv (params_to_store, index = False, header=True)
    
    #initialize empty lists to store the set of parameters that our model has searched
    bs, ts, ep, nuerons_1, nuerons_2, num_layers, drop_1, drop_2, loss = [],[],[],[],[],[],[],[],[]
    
    for i in range(0,len(trials.trials)):
        
        bs.append(vals['batch_size'][trials.trials[i]['misc']['vals']['batch_size'][0]])
        ts.append(vals['offset'][trials.trials[i]['misc']['vals']['offset'][0]])
        ep.append(vals['epochs'][trials.trials[i]['misc']['vals']['epochs'][0]])
        nuerons_1.append(vals['lstm1_nodes'][trials.trials[i]['misc']['vals']['lstm1_nodes'][0]])
        drop_1.append(np.round(trials.trials[i]['misc']['vals']['lstm1_dropouts'][0],3))
        loss.append(trials.trials[i]['result']['loss'])
        
        if vals['lstm_layers'][trials.trials[i]['misc']['vals']['lstm_layers'][0]] == 2:
            
            num_layers.append(vals['lstm_layers'][trials.trials[i]['misc']['vals']['lstm_layers'][0]])
            nuerons_2.append(vals['lstm2_nodes'][trials.trials[i]['misc']['vals']['lstm2_nodes'][0]])
            drop_2.append(np.round(trials.trials[i]['misc']['vals']['lstm2_dropouts'][0],3))
            
        else:
            
            num_layers.append(vals['lstm_layers'][trials.trials[i]['misc']['vals']['lstm_layers'][0]])
            nuerons_2.append(0)
            drop_2.append(0)            
    
    hyper_params = [bs,ts,ep,num_layers,nuerons_1,nuerons_2,drop_1,drop_2,loss]
    
    df = pd.DataFrame(hyper_params).transpose()
    df.columns = ['batch_size','offset','epochs','lstm_layers','lstm1_nodes','lstm2_nodes',
                  'lstm1_dropouts','lstm2_dropouts','loss']
    
    #the dataframe is sorted in the increasing order of the 'loss' column
    df.sort_values(['loss'], axis=0, ascending=True, inplace=True)
    
    
    path_to_store  = script_folder / 'trials_data.csv'
    
    #store all the results in a .csv file for future reference
    df.to_csv(path_to_store, index = False, header = True)
                
