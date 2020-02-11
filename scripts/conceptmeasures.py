import re

import nltk
import sys
from corpusstats import ngramcounting, stats
from datautils import dataio, datapaths
from tqdm import tqdm

CORPUS_NAME = 'genia'
FREQ_THRESHOLD = 5
LEN_THRESHOLD = 4
C_THRESHOLD = 3


ngrams = ngramcounting.main(CORPUS_NAME, max_n=5)
ngrams_alt = ngramcounting.main(CORPUS_NAME, max_n=5, pos_tag_filter=ngramcounting.FILTERS['liberal'])

for i in range(1, 6):
    print(ngrams.total_counts(i), ngrams_alt.total_counts(i))

candidate_concepts = {c for c in ngrams_alt.generate_ngrams()
                      if 'COLLAPSED' not in c
                      and (ngrams.freq(c) > FREQ_THRESHOLD
                           or len(c) > LEN_THRESHOLD)
                      }

bigram_lls = stats.calculate_pointwise_mutual_information(
    set(((subgram[0],), (subgram[1],)) for ngram in candidate_concepts
        for subgram in nltk.bigrams(ngram)), ngrams
)
bigram_lls_alt = stats.calculate_pointwise_mutual_information(
    set(((subgram[0],), (subgram[1],)) for ngram in candidate_concepts
        for subgram in nltk.bigrams(ngram)), ngrams_alt
)

x = []
y = []
for key, value in bigram_lls.items():
    x.append(value)
    alt_value = bigram_lls_alt[key]
    y.append(alt_value)
    if abs(value - alt_value) > 1:
        print(key, value, alt_value, sep='\t')
import seaborn
seaborn.scatterplot(x=x, y=y)


CORPUS_NAME = 'asda'

if CORPUS_NAME.lower() == 'genia':
    corpus = dataio.load_genia_corpus()
elif CORPUS_NAME.lower() == 'craft':
    corpus = dataio.load_craft_corpus()
else:
    corpus = None

ngrams.save_to_file(datapaths.PATH_TO_PMC + 'bigrams_PMC1+6')

if corpus:
    gold = stats.gold_standard_concepts(corpus)
    bigram_lls = stats.calculate_ngram_log_likelihoods(
        set(((subgram[0],), (subgram[1],)) for ngram in gold
            for subgram in nltk.bigrams(ngram)), ngrams
    )
    bigram_lls = stats.calculate_ngram_log_likelihoods(
        set(((subgram[0],), (subgram[1],)) for ngram in gold
            for subgram in nltk.bigrams(ngram)), ngrams
    )
    c_values = stats.calculate_c_values(gold, C_THRESHOLD, ngrams)
    tf_idfs = stats.calculate_tf_idf_values(gold, corpus, ngrams)
