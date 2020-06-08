# Stock Price Prediction using LSTM (Long Short Term Memory) Nueral Networks
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
can predict the future movement of prices with good accuracy assuming that it is free of
any and all external factors.

Though SVM's can be used for their fast and simple application, LSTM's are used in this 
project for their feedback connection ability. There are 5 main steps involved here:

- Building a General model
- Optimization of the model hyperparameters
- Check for the overfit/underfit 
- Predicting the future stock price 

Let's see what is done in each step.

## Building a General model
An LSTM model is built with random initial parameters. And using a trail-and-error
method, the parameters are tweaked in each iteration as to get atleast 85-90%
accuracy. This is done because, in our next step i.e, 'Optimization', the maximum
number of evaluations are fixed at 100. So to get the best params, we employ a 
trial-and-error method to get good params so as to get a headstart in the next step.
80% of data is used as trian set and 20% as test set.

Based on the other data that I have tried, I feel that a nueral network with 2 hidden
layers with 32 nodes in each layer, dropout rate of 0 for both layers and an offset
(no. of days data used) of 60 days is used. The 'R-squared value' is used as a measure
of accuracy and with these initial params, the accuracy is around 93%. Well, our accuracy
is more than 90% and we have a good starting point. Let's proceed to the model optimization.

## Optimization of the model hyperparameters
There are multiple options for the selection of hyperparameters, Manual tuning, Grid-Search, 
Random-Search, Bayesian Optimization to name a few. We can say right off the bat that
manual tuninig is out of the question as we have to tweak the params everytime and we 
can't be sure if we are improving the model or worsening it. Grid-Search is the tuning
where we give a set of values for each hyperparameter and the algorithm runs the model
for every possible combination of those values. This is very costly when we have more
hyperparameters or huge set of values. Random-Search is an improvement over Grid-Search,
in which the algorithm searchs for the best params randomly. Both these share the same
downside, i.e the trials are independent of each other.

'Bayesian Optimzation' strategy builds a surrogate model that  tries to predict the 
metrics and at each iteration, the surrogate becomes more and more confident about
which new guesses can lead to the improvement. For more information on these, you can
have a look at this simple yet interesting blog here.
https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/

A search space is created using the params from the previous step as reference and the 
'TPE'(Tree of Parzen Estimators) algorithm is used to build the surrogate model and 
search the values in the search space. The loss function to minimize here is the 'RMSE' 
value. Bayesian Optimization is implemented using the 'HyperOpt' package here. 

The maximum number of evaluations are fixed at 100 i.e the algorithm runs for 100 times
trying to find the best params. And all the combination of params used by the model is
saved into .csv file in the ascending order of the loss value (RMSE error).

## Check for the overfit/underfit 
Even though we get the high accuracy for our model using the best params from the previous
step, we have to make sure that these set of params don't overfit or underfit our model.
In this case, the model may perform better on our training set, but will fail on test set.
To make sure that this is not the case, let's take the first set of params from the .csv
and plug into our model. The training loss and test loss is observed and if they converge,
then the model is performing well. Else, we have to ignore these set of params, and move on
to the next set of params. In our case, the training loss and test loss is converging and
we will move forward with these set of params.

## Predicting the future stock price 
Taking the params from the previous step, we will predict the stock prices for the next 30
days. Now the entire data is used to train the model. Let's say that our objective is to 
predict the price from Day 1 to Day 30. We will use the last 60 days (offset) data to predict
the price on Day 1. Since we are predicting into the future, we will not have the actual Day 1
price to compare, adjust and retrain the model to get a more accuarte Day 2 price. So, assuming
that our Day 1 price is acceptable and with-in the acceptance level, we will predict the Day 2
price. And this is continued till the Day 30 price prediction. Since there will be error from
Day 1, the error is propagated in every iteration and we can see that the prediction value
will largely be off from the original value as the days progress.

# Results
The predicted values are compared with the original values. It is observed that
- The model can predict the next 3 days values with a 3% error
- The next 8 days values with a 10% error
- And the next 30 days values with a 15% error

# Note
For this model, only the 'Close' prices are taken as input and future 'Close' price is predicted.
But in reality, there will be many more variables like 'Open' price, 'Volume' traded etc that
effect the 'Close' price. Then our model has to deal with multi-variable input/output scenario.
You can check this in my [other repository](https://github.com/revanth-talluri/Multivariate-input-price-prediction). Pelase note that those scenarios do not have an 
in-depth exploration like we have seen here.





