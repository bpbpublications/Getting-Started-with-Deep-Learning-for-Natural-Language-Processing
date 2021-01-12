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
import numpy as np
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter()
    i = [np.random.randint(0, 100) for i in range(0, 32)]
    writer.add_embedding(np.random.random([32, 20]), i)

    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")
    writer.export_scalars_to_json("./all_scalars_2.json")
    writer.close()
