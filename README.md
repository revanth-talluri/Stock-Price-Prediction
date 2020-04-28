# stock-predictions
Stock 'Close' price prediction using LSTM RNN's 
(with 3% error in 3 days and 15% error in 30 days)

With the vast amount of data available in the Finance sector, it is one of the best places
to explore and apply the ML techniques. Personally, the stock price prediction is one of 
the interesting areas for me.

The project tries to predict the stock close prices of Google(Aplhabet) for the month of
April, 2020 using the LSTM (Long Short Term Memory cells) RNN's. 

The data is taken from the Yahoo Finance between 2015/04/01 and 2020/03/31. The sudden 
rise and dips in the data (if any) are assumed to be free of any external factors (Though
this is not true, and especially the given time frame includes the effect of Covid-19).
The idea here is to try and build a model that can take even a small amount of data and 
can predict the future movement of prices with good accuracy.

Though SVM's can be used for their fast and simple application, LSTM's are used in this 
project for their feedback connection ability. There are 4 main steps involved here.

1) Build a General model:
      A generic model is built with random initial parameters. The params are adjusted 
      using a trial and error method so as to bring the model to atleast 85-90% accuracy. 
      
2) Optimization:
      Next, optimization of hyperparameters (referred to as params from here on) is done.
      Since there are many params here to be optimized, 'Bayesian Optimization' using the
      'Hyperopt' library is used. All the trials are saved in a .csv file.

3) Check Overfit:
      The model is now checked for the overfit/underfit taking the params from trails.csv
      If the model is overfit/underfit, go for the next set of params
      
4) Future Predictions:
      The best set of params are chosen and the stock price for the next 30 days are predicted.
      
Results: The predicted values are then compared with the original values. It is observed that
         a) The model can predict the next 3 days values with a 3% error.
         b) The next 8 days values with a 10% error.
         c) And the next 30 days values with a 15% error.
