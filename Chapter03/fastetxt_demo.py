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

import fasttext
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()

if __name__ == "__main__":

    # Skipgram model
    model = fasttext.skipgram('data/testdata_en.txt', 'model')
    words = model.words  # list of words in dictionary
    print("words present in the model : ", words)

    # # CBOW model
    # model = fasttext.cbow('data/testdata_en.txt', 'model')
    # print (model.words) # list of words in dictionary

    # I am using only  Skipgram model model

    # visualizing using tensorboard
    print(
        """##################################\n## Launch tensorboard as: ## \n## tensorboard --logdir=runs/ ## \n##################################""")
    all_vectors = []
    for eachword in words:
        all_vectors.append(model[eachword])
    writer.add_embedding(np.asarray(all_vectors), words)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
