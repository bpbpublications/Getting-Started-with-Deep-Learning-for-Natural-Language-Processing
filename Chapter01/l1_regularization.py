# -*- coding: utf-8 -*-
"""
## Author: Sunil Patel
## Copyright: Copyright 2018-2019, Packt Publishing Limited
## Version: 0.0.1
## Maintainer: sunil patel
## Email: snlpatel01213@hotmail.com
## Linkedin: https://www.linkedin.com/in/linus1/
## Contributor : {if you debug, append your name here}
## Contributor Email : {if you debug, append your email here}
## Status: active
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# Run Lasso Regressions, Varying Alpha Levels
# Create a function called lasso,
def lasso(alphas):
    '''
    Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
    '''
    # Create an empty data frame
    df = pd.DataFrame()

    # Create a column of feature names
    df['Feature Name'] = names

    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        # Create a lasso regression with that alpha value,
        lasso = Lasso(alpha=alpha)

        # Fit the lasso regression
        lasso.fit(X, Y)

        # Create a column name for that alpha value
        column_name = ' α = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = lasso.coef_

    # Return the datafram
    return df


if __name__ == '__main__':
    boston = load_boston()
    scaler = StandardScaler()
    X = scaler.fit_transform(boston["data"])
    Y = boston["target"]
    names = boston["feature_names"]

    # Run the function called, Lasso
    df = lasso([.0001, 0.25, .5, 0.75, 1.0])
    print(df)

    # plotting it
    ax = df[[' α = 0.000100', ' α = 0.250000', ' α = 0.500000', ' α = 0.750000', ' α = 1.000000']].plot()
    ax.set_title("Plot Lasso Regressions, Varying Alpha Levels")
    ax.set_xticklabels(df['Feature Name'], rotation=90)
    plt.show()
