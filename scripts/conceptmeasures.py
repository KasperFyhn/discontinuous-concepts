import re

import nltk
import sys
from corpusstats import ngramcounting, stats
from datautils import dataio, datapaths
from tqdm import tqdm

CORPUS_NAME = 'pmc'
FREQ_THRESHOLD = 5
LEN_THRESHOLD = 4
C_THRESHOLD = 3


if CORPUS_NAME.lower() == 'genia':
    corpus = dataio.load_genia_corpus()
elif CORPUS_NAME.lower() == 'craft':
    corpus = dataio.load_craft_corpus()
else:
    corpus = None


if corpus:
    gold = stats.gold_standard_concepts(corpus)
    bigram_lls = stats.calculate_ngram_log_likelihoods(
        set(((subgram[0],), (subgram[1],)) for ngram in gold
            for subgram in nltk.bigrams(ngram)), ngrams
    )
    c_values = stats.calculate_c_values(gold, C_THRESHOLD, ngrams)
    tf_idfs = stats.calculate_tf_idf_values(gold, corpus, ngrams)
