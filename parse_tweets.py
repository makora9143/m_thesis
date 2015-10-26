#! /usr/bin/env python
# -*- coding: utf-8 -*-


import re
import math
import csv

import MeCab
import geohash
import jpgrid
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from gensim.models import word2vec

from m2_vae import M2_VAE


GEOHASH_DB = '/Users/makora/Dropbox/geohash.csv'
DATAPATH = '/Users/makora/Dropbox/2015-06.csv'
#DATAPATH = ''
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

GEONAMES = [
    'geohash',
    'lat',
    'lng',
    'pref',
    'branch',
    'city',
    'town',
    'postnum'
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
URL = r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+'
HASHTAG = r'[#＃][Ａ-Ｚａ-ｚA-Za-z一-鿆0-9０-９ぁ-ヶｦ-ﾟー]+'

COMPILERS = [
        re.compile(ACCOUNTS),
        re.compile(SYMBOLS),
        re.compile(URL),
        re.compile(HASHTAG)
        ]


TAGGER = MeCab.Tagger('-Owakati')


def preprocessing(texts):
    for compiler in COMPILERS:
        texts = [compiler.sub('', text) for text in texts if not math.isnan(text)]

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
        print fname + ' is ',
        data = pd.read_csv(fname, quotechar='', names=NAMES)
        print 'loaded, now preprocessing'
        texts = data.text
        all_data += preprocessing(texts)
#    data = pd.read_csv(DATAPATH, quotechar='', names=NAMES)
#    texts = data.text
#    print 'loaded, now preprocessing'
#    all_data += preprocessing(texts)

    print 'saving all data'
    save_file(all_data, filename)
    print 'create model'
    create_model(filename)


def create_location_data(mode=None):
    GS = pd.read_csv(GEOHASH_DB, names=GEONAMES)
#    for date in DATES:
#        fname = DATAPATH + date + '.csv'
    for i in range(1):
        fname = DATAPATH
#        f = open(DATAPATH + date + "_geogrid1.csv", 'ab')
#        f = open(DATAPATH + date + "_geogrid2.csv", 'ab')
#        f = open(DATAPATH + date + "_pref.csv", 'ab')
#        f = open(DATAPATH + date + "_city.csv", 'ab')
        f = open('./location.csv', 'ab')
        csvwriter = csv.writer(f)
        print fname
        datas = pd.read_csv(open(fname, 'rU'), quotechar='', names=NAMES)
        print 'load data'
        lats = datas.lat
        lngs = datas.lng
        locations = []

        
        #locations = [(lat, lng) for lat, lng in zip(lats, lngs)]
        #locations = [[jpgrid.encodeLv1(lat, lng)] for lat, lng in zip(lats, lngs)]
        #locations = [[jpgrid.encodeLv2(lat, lng)] for lat, lng in zip(lats, lngs)]
        #locations = [[GS[GS.geohash == geohash.encode(lat, lng)].pref] for lat, lng in zip(lats, lngs)]
        #locations = [[GS[GS.geohash == geohash.encode(lat, lng)].city] for lat, lng in zip(lats, lngs)]

#        for lat, lng in zip(lats, lngs):
#            geohash_code = geohash.encode(lat, lng)
#            target = GS[GS.geohash == geohash_code]
#            if mode == 'pref':
#                locations.append([target.pref])
#            elif mode == 'city':
#                locations.append([target.city])
#            elif mode =='g1':
#                locations.append([jpgrid.encodeLv1(lat, lng)])
#            elif mode == 'g2':
#                locations.append([jpgrid.encodeLv2(lat, lng)])
#            else:
#                locations.append([lat, lng])
        print "write file"
        csvwriter.writerows(locations)
        f.close()


def load_dataset_tweets(filename=None):
    pass


def get_sentence_vector(model, sentence):
    words = TAGGER.parse(sentence)
    vector = np.array([])
    for word in words:
        vector += model[word]
    return vector


def text2vector(model, texts):
    vectors = []
    for text in texts:
        vectors.append(get_sentence_vector(model, text))
    return vectors


def train_vae_model(model_file):
    w2w = word2vec.Word2Vec.load(model_file)

    tweets, locations = load_dataset_tweets()

    tweets = text2vector(model=w2w, texts=tweets)

    train_tweets, test_tweets, train_locations, test_locations = train_test_split(tweets, locations, train_size=0.8, random_state=1234)

    train_tweets, valid_tweets, train_locations, valid_tweets = train_test_split(train_tweets, train_locations, train_size=0.9, random_state=1234)

    all_params = {
        'hyper_params': {
            'rng_seed'          : 1234,
            'dim_z'             : 50,
            'n_hidden'          : [500, 500],
            'n_mc_sampling'     : 1,
            'scale_init'        : 0.01,
            'nonlinear_q'       : 'softplus',
            'nonlinear_p'       : 'softplus',
            'type_px'           : 'bernoulli',
            'optimizer'         : 'adam',
            'learning_process'  : 'early_stopping'
        },
        'optimize_params': {
            'learning_rate'        : 1e-4,
            'n_iters'              : 1000,
            'minibatch_size'       : 1000,
            'calc_history'         : 'all',
            'calc_hist'            : 'all',
            'n_mod_history'        : 100,
            'n_mod_hist'           : 100,
            'patience'             : 5000,
            'patience_increase'    : 2,
            'improvement_threshold': 1.005,
        }
    }
    vae = M2_VAE(**all_params)
    vae.fit(train_tweets, train_locations)


if __name__ == '__main__':
    treat_all_data('all_tweet_corpus')

# End of Line.
