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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)


def true_fun(X):
    """
    given X it will provide its mapping to Y by sing function np.cos(1.5 * np.pi * X)
    :param X:
    :return:
    """
    return np.cos(1.5 * np.pi * X)


if __name__ == '__main__':
    n_samples = 30
    degrees = [1, 3, 9, 15]

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        """
        Evaluating and plotting for each degree of freedom
        """
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        # Evaluate the models using cross-validation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)

        # predicting on test data
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")

        # plotting the True  and predicted function
        plt.plot(X_test, true_fun(X_test), label="True function")
        plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}\n TEST MSE = {:.2e}".format(
            degrees[i], -scores.mean()))
    plt.show()
