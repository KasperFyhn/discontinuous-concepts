import re

import nltk
import sys
from stats import conceptstats
from datautils import dataio as dio
from tqdm import tqdm
import seaborn as sns
import pandas as pd

CORPUS = 'craft'
MODEL_SPEC = '_skip_all'
FREQ_THRESHOLD = 5
LEN_THRESHOLD = 5
C_THRESHOLD = 2


load_corpus = dio.load_genia_corpus if CORPUS.lower() == 'genia' \
    else dio.load_craft_corpus if CORPUS.lower() == 'craft'\
    else dio.load_genia_corpus
corpus = load_corpus()
print('Excluding discontinuous concepts: ', end='')
cont_concepts = conceptstats.gold_standard_concepts(corpus, allow_discontinuous=False)
print('Only discontinuous concepts: ', end='')
disc_concepts = conceptstats.gold_standard_concepts(corpus).difference(cont_concepts)
all_concepts = cont_concepts.union(disc_concepts)

model = conceptstats.NgramModel.load_model(CORPUS, MODEL_SPEC)

c_values_all = conceptstats.calculate_c_values(all_concepts, C_THRESHOLD, model,
                                               skipgrams=True)
c_values_cont = conceptstats.calculate_c_values(cont_concepts, C_THRESHOLD, model)
tf_idf_values = conceptstats.calculate_tf_idf_values(all_concepts, corpus, model)


concepts, c_all, c_cont, tf_idf, only_cont = [], [], [], [], []
for concept in all_concepts:
    if len(concept) > 5 or model.freq(concept, include_skipgrams=True) == 0:
        continue
    concepts.append(concept)
    c_all.append(c_values_all[concept])
    c_cont.append(c_values_cont[concept] if concept in c_values_cont else None)
    tf_idf.append(tf_idf_values[concept])
    only_cont.append(concept in cont_concepts)

data = pd.DataFrame({'concept': concepts, 'only_cont': only_cont,
                     'C(all)': c_all, 'C(excl_DC)': c_cont, 'tf-idf': tf_idf, }
                    )

sns.boxplot(x='only_cont', y='C(all)', data=data, showfliers=False)

