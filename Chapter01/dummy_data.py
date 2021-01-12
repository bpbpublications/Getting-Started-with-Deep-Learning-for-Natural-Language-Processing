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


def make_positive(mask):
    """
    MAKING POSITIVE DATASET
    :param mask:
    :return:
    """
    mask[5, 4] = 1
    mask[4, 5] = 1
    mask[5, 6] = 1
    mask[6, 5] = 1
    mask[2, 2] = 0
    mask[2, 7] = 0
    mask[7, 2] = 0
    mask[7, 7] = 0
    return mask


def make_negative(mask):
    """
    MAKING NWGATIVE DATASET
    :param mask:
    :return:
    """
    mask[2, 2] = 1
    mask[2, 7] = 1
    mask[7, 2] = 1
    mask[7, 7] = 1
    mask[5, 4] = 0
    mask[4, 5] = 0
    mask[5, 6] = 0
    mask[6, 5] = 0
    return mask


if __name__ == '__main__':
    random_grid = np.random.uniform(0.0, 1, [10, 10])
    mask = random_grid
    # print(make_positive(mask))
    plt.subplot(121)
    plt.imshow(make_positive(mask), cmap="inferno")
    random_grid = np.random.uniform(0.0, 1.0, [10, 10])
    mask = random_grid
    # print(make_negative(mask))
    plt.subplot(122)
    plt.imshow(make_negative(mask), cmap="inferno")

    plt.show()
