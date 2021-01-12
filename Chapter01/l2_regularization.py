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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# Run Ridge Regressions, Varying Alpha Levels
# Create a function called Ridge_,
def ridge_(alphas):
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
        Ridge_obj = Ridge(alpha=alpha)

        # Fit the lasso regression
        Ridge_obj.fit(X, Y)

        # Create a column name for that alpha value
        column_name = ' α = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = Ridge_obj.coef_

    # Return the datafram
    return df


if __name__ == '__main__':
    boston = load_boston()
    scaler = StandardScaler()
    X = scaler.fit_transform(boston["data"])
    Y = boston["target"]
    names = boston["feature_names"]
    # Run the function called, Ridge
    df = ridge_([.0001, 25, 50, 75, 100])

    ax = df.plot()
    ax.set_title("Plot Ridge Regressions, Varying Alpha Levels")
    ax.set_xticklabels(df['Feature Name'], rotation=90)
    plt.show()
