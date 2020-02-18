import re

import nltk
import sys
from corpusstats import ngramcounting, stats
from datautils import dataio as dio
from tqdm import tqdm

CORPUS = 'genia'
MODEL_SPEC = '_noskip_all'
FREQ_THRESHOLD = 5
LEN_THRESHOLD = 5
C_THRESHOLD = 2


load_corpus = dio.load_genia_corpus if CORPUS.lower() == 'genia' \
    else dio.load_craft_corpus if CORPUS.lower() == 'craft'\
    else dio.load_genia_corpus
corpus = load_corpus()
gold = stats.gold_standard_concepts(corpus)

model = stats.NgramModel.load_model(CORPUS, MODEL_SPEC)

c_values = stats.calculate_c_values(gold, C_THRESHOLD, model)
tf_idfs = stats.calculate_tf_idf_values(gold, corpus, model)
