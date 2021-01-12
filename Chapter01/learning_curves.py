# -*- coding: utf-8 -*-
"""
## Author: Sunil Patel
## Copyright: Copyright 2018-2019, Packt Publishing Limited
## Version: 0.0.1
## Maintainer: Sunil Patel
## Email: snlpatel01213@hotmail.com
## Linkedin: https://www.linkedin.com/in/linus1/
## Contributor : {if you debug, append your name here}
## Contributor Email : {if you debug, append your email here}
## Status: active
"""
# dataset source - "Electrical Grid Stability Simulated Data Data Set"
# source url https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

if __name__ == '__main__':

    # Loading the data
    data = pd.read_csv('data/Data_for_UCI_named.csv')
    data.head()

    # converting categorical to numerical
    stav_int = []
    for i in list(data["stabf"].values):
        if i == "unstable":
            stav_int.append(0)
        else:
            stav_int.append(1)

    # assing numerical variable to pandas data frame
    data["stav_int"] = stav_int

    # defining various data fraction
    train_sizes = [1, 5, 10, 25, 20, 25, 50, 75]

    features = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4", "stab"]
    target = 'stav_int'

    # --------------------LINEAR REGRESSION MODEL--------------------------

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=LinearRegression(), X=data[features],
        y=data[target], train_sizes=train_sizes,
        scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.style.use('seaborn')

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Test error')

    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for a linear regression model', fontsize=18)
    plt.legend()
    plt.show()

    # --------------------RANDOM FOREST MODEL--------------------------

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=RandomForestRegressor(), X=data[features],
        y=data[target], train_sizes=train_sizes, cv=5,
        scoring='neg_mean_squared_error')
    # PLOTTING RESULTS
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.style.use('seaborn')

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Test error')

    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for random forest model', fontsize=18)
    plt.legend()
    plt.show()
