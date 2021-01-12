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

import math

import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    weight function o prevent influence of extremely frequent word on the algorithm
    """
    xmax = 1000
    a = 0.75
    weight = [min(1, math.pow(x / xmax, a)) for x in range(1, 2000)]
    plt.plot([x for x in range(1, 2000)], weight)
    plt.ylabel("Weight")
    plt.xlabel("Xij")
    plt.title("Weighting Function")
    plt.show()
