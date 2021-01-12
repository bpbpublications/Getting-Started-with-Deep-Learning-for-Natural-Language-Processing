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

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")

    writer = SummaryWriter()
    for i in range(0, 10):
        writer.add_text('mytext', 'this is a pen_' + str(i), i)

    writer.export_scalars_to_json("./all_scalars_2.json")
    writer.close()
