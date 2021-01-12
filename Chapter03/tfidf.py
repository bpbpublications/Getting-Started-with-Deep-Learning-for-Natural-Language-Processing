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
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def heatmap(tfidf_matrix, title, xlabel, ylabel, xticklabels, yticklabels):
    """
    To plot tfidf_matrix using matplotlib
    
    :param tfidf_matrix:  cooccurance matrix
    :param title: Title of the plot
    :param xlabel: x label
    :param ylabel: y label
    :param xticklabels: x ticks
    :param yticklabels: y ticlks
    :return: 
    """

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(tfidf_matrix, edgecolors='k', linestyle='dashed', cmap=plt.cm.Blues, linewidths=0.2, vmin=0.0,
                  vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(tfidf_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(tfidf_matrix.shape[1]) + 0.5, minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,tfidf_matrix.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False, rotation='vertical')
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    # plt.xlim( (0, tfidf_matrix.shape[1]) )
    for i in range(tfidf_matrix.shape[0]):
        for j in range(tfidf_matrix.shape[1]):
            c = round(tfidf_matrix[i, j], 2)
            ax.text(j, i, str(c))

    plt.show()


if __name__ == "__main__":
    corpus = [
        'this is the one document.',
        'this is the second document.',
        'and this is the third one, which is very similar to first one.',
        'is this the first document relates to politics?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(csr_matrix.todense(X))

    heatmap(np.array(csr_matrix.todense(X)), "", "", "", vectorizer.get_feature_names(), corpus)
