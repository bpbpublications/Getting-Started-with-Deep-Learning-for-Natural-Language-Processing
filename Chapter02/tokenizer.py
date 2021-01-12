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

import spacy
from nltk.tokenize import sent_tokenize, word_tokenize

if __name__ == '__main__':

    print("======== using NLTK tokenizer ========")
    data = """"At a time when more soldiers are committing suicide than are dying in battle, it is well to remember that, no matter how thoroughly indoctrinated the belief in the superiority of an abstraction, there remains within each of us a powerful life-force that can never be fully repressed. What M. Gandhi called Satyagraha – a “Truth-force” or “Soul-force” – remains deep within us. For more abstracts mail at abstract.lsat@lsat.com"""
    for each_sent in sent_tokenize(data):
        print("SENTENCE : ", each_sent)
        all_token = word_tokenize(each_sent)
        print("WORDS : ", all_token)

    print("\n\n======== Using Spacy tokenizer ========")
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(data)
    for sent in doc.sents:
        print("SENTENCE : ", sent)
        print("WORDS : ", [str(token) for token in sent])
