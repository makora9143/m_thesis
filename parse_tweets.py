#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re

import MeCab
import pandas as pd
from gensim.models import word2vec


DATAPATH = ''
NAMES = [
    'tweet_id',
    'screen_name',
    'account',
    'account_id',
    'text',
    'image',
    'lat',
    'lng',
    'date'
]


DATES = [
    "2013-11",
    "2013-12",
    "2014-01",
    "2014-02",
    "2014-03",
    "2014-04",
    "2014-05",
    "2014-06",
    "2014-07",
    "2014-08",
    "2014-09",
    "2014-10",
    "2014-11",
    "2014-12",
    "2015-01",
    "2015-02",
    "2015-03",
    "2015-04",
    "2015-05",
    "2015-06",
    "2015-07",
    "2015-08",
    "2015-09",
]

ACCOUNTS = r'@[0-9a-zA-Z_]{1,15}'
SYMBOLS = r'[\\\nwｗ]'
#URL = r'http(s)?://([\w-]+\.)+[\w-]+(/[\w- ./?%&=]*)?'
HASHTAG = r'[#＃][Ａ-Ｚａ-ｚA-Za-z一-鿆0-9０-９ぁ-ヶｦ-ﾟー]+'

COMPILERS = [
        re.compile(ACCOUNTS),
        re.compile(SYMBOLS),
        re.compile(URL),
        re.compile(HASHTAG)
        ]

#DATA = pd.read_csv(DATAPATH, quotechar='', names=NAMES)


TAGGER = MeCab.Tagger('-Owakati')


def preprocessing(texts):
    for compiler in COMPILERS:
        texts = map(lambda text: compiler.sub('', text), texts)

    wakati_texts = map(TAGGER.parse, texts)
    return wakati_texts


def save_file(texts, filename):
    try:
        with open(filename, 'w') as f:
            f.writelines(texts)
        return True
    except:
        return False


def create_model(filename):
    sentences = word2vec.Text8Corpus(filename + ".txt")
    model = word2vec.Word2Vec(sentences, size=100)
    model.save(filename + ".model")


def treat_all_data(filename):
    all_data = []
    for date in DATES:
        fname = DATAPATH+date+'.csv'
        print fname
        data = pd.read_csv(fame, quotechar='', names=NAMES)
        texts = data.texts
        all_data += preprocessing(texts)

    print 'saving all data'
    save_file(all_data, filename)
    print 'create model'
    create_model(filename)

if __name__ == '__main__':
    treat_all_data('all_tweet_corpus')

# End of Line.
