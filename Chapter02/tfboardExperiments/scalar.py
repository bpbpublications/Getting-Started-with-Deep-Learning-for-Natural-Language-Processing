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

import math
import random

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")

    writer = SummaryWriter()

    # writing both to separately
    for i in range(0, 100):
        writer.add_scalar('sin', math.sin(i * 0.001) + random.random(), i)
        writer.add_scalar('cos', math.cos(i * 0.001) + random.random(), i)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
