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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


# Run Elastic Net Regressions, Varying Alpha Levels
def elastic(alphas):
    '''
    Takes in a list of alphas. Outputs a data frame containing the coefficients of lasso regressions from each alpha.
    '''
    # Create an empty data frame
    df = pd.DataFrame()

    # Create a column of feature names
    df['Feature Name'] = names

    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        # Create a lasso regression with that alpha value,
        Ridge = ElasticNet(alpha=alpha, l1_ratio=0.5)

        # Fit the lasso regression
        Ridge.fit(X, Y)

        # Create a column name for that alpha value
        column_name = ' α = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = Ridge.coef_

    # Return the dataframe
    return df


if __name__ == '__main__':
    boston = load_boston()
    scaler = StandardScaler()
    X = scaler.fit_transform(boston["data"])
    Y = boston["target"]
    names = boston["feature_names"]
    # Run the function called, Lasso
    df = elastic([.0001, 0.25, 0.50, 0.75, 1.00])

    # plotting
    ax = df.plot()
    ax.set_title("Plot Elatic Regressions, Varying Alpha Levels")
    ax.set_xticklabels(df['Feature Name'], rotation=90)
    plt.show()
