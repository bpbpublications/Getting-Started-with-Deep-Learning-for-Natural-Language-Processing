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

#  Sub-sampling logic in word2vec
import math

import matplotlib.pyplot as plt
import numpy as np


def get_chances_of_being_kept(fraction):
    sample = 0.001
    return (math.sqrt(fraction / sample) + 1) * (sample / fraction)


if __name__ == "__main__":
    fractions = np.arange(0.01, 1.0, 0.001)
    chances_of_kept = [get_chances_of_being_kept(i) for i in fractions]
    plt.title("Chances being kept v/s Occurance in Fraction")
    x = chances_of_kept
    plt.ylabel("Chances being kept")
    plt.xlabel("Occurance in Fraction")
    plt.plot(x, fractions);
    plt.show()
