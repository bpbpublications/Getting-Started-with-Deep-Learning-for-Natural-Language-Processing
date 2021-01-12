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

if __name__ == "__main__":
    # train classifier
    classifier = fasttext.supervised('data/SMSSpamCollection.train', 'model')
    # check performance on test
    result = classifier.test('data/SMSSpamCollection.test')
    print('Precision:', result.precision)
    print('Recall:', result.recall)
    print('Number of examples:', result.nexamples)
