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


def heatmap(cooccurrence_matrix, title, xlabel, ylabel, xticklabels, yticklabels):
    """
    To plot cooccurrence_matrix using matplotlib
    
    :param cooccurrence_matrix:  cooccurance matrix
    :param title: Title of the plot
    :param xlabel: x label
    :param ylabel: y label
    :param xticklabels: x ticks
    :param yticklabels: y ticlks
    :return: 
    """

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(cooccurrence_matrix, edgecolors='k', linestyle='dashed', cmap=plt.cm.Blues, linewidths=0.2, vmin=0.0,
                  vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cooccurrence_matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(cooccurrence_matrix.shape[1]) + 0.5, minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,cooccurrence_matrix.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    # plt.xlim( (0, cooccurrence_matrix.shape[1]) )
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            c = cooccurrence_matrix[i, j]
            ax.text(i, j, str(c))

    plt.show()


def create_cooccurrence_matrix(text, context_size):
    """
    to creat coocurrence matrix
    :param text:
    :param context_size: near by word to be considered
    :return:
    """
    word_list = text.split()
    vocab = np.unique(word_list)
    w_list_size = len(word_list)
    vocab_size = len(vocab)
    w_to_i = {word: ind for ind, word in enumerate(vocab)}

    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    for i in range(w_list_size):
        for j in range(1, context_size + 1):
            ind = w_to_i[word_list[i]]
            if i - j > 0:
                lind = w_to_i[word_list[i - j]]
                cooccurrence_matrix[ind, lind] += round(1.0 / j, 2)
            if i + j < w_list_size:
                rind = w_to_i[word_list[i + j]]
                cooccurrence_matrix[ind, rind] += round(1.0 / j, 2)

    return cooccurrence_matrix


if __name__ == "__main__":
    samples = 'I am a ML scientist and I love working with data .'
    cooccurrence_matrix = create_cooccurrence_matrix(samples, context_size=3)
    print("cooccurrence_matrix : ", cooccurrence_matrix)
    heatmap(np.array(cooccurrence_matrix), "", "", "", list(set(samples.split())), list(set(samples.split())))
