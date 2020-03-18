import re
import math
import nltk
import sys
from stats import conceptstats
from datautils import dataio as dio
from tqdm import tqdm
import seaborn as sns
import pandas as pd

CORPUS = 'PMC'
MODEL_NAME = CORPUS
MODEL_SPEC = '_min20'
FREQ_THRESHOLD = 5
MAX_LEN = 5
MIN_LEN = 1
C_THRESHOLD = 2


corpus = load_corpus(CORPUS.lower())
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


concepts, freq, c_all, c_cont, tf_idf, only_cont = [], [], [], [], [], []
for concept in all_concepts:
    if not MIN_LEN < len(concept) <= MAX_LEN \
            or model.freq(concept, include_skipgrams=True) == 0:
        continue
    concepts.append(concept)
    freq.append(model.freq(concept))
    c_all.append(c_values_all[concept])
    c_cont.append(c_values_cont[concept] if concept in c_values_cont else None)
    tf_idf.append(tf_idf_values[concept])
    only_cont.append(concept in cont_concepts)

data = pd.DataFrame({'concept': concepts, 'only_cont': only_cont, 'freq': freq,
                     'C(all)': c_all, 'C(excl_DC)': c_cont, 'tf-idf': tf_idf, }
                    )
data['log_freq'] = [math.log10(v) for v in data['freq']]

sns.boxplot(x='only_cont', y='log_freq', data=data, showfliers=True)

