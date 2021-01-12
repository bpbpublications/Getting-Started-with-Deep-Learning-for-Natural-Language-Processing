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

import nltk

nltk.download('popular')
lancaster = nltk.stem.lancaster.LancasterStemmer()
porter = nltk.stem.porter.PorterStemmer()
snowball = nltk.stem.snowball.EnglishStemmer()
WordNetLemmatizer = nltk.stem.WordNetLemmatizer()


def differnt_stemmars_and_lemmatizer(word):
    print("Word : ", word)
    print("Lancaster Stemmer : ", lancaster.stem(word))
    print("Porter Stemmer : ", porter.stem(word))
    print("Snowball Stemmer : ", snowball.stem(word))
    print("Snowball Lemmatizer : ", WordNetLemmatizer.lemmatize(word))


word_list = ["maximum", "cats", "seventy-one", "cacti", "geese", "better", "Agreed", "Plastered", "Motoring"]

if __name__ == '__main__':

    for each_word in word_list:
        differnt_stemmars_and_lemmatizer(each_word)
