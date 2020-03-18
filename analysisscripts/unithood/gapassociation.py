from datautils import dataio, annotations as anno
from collections import defaultdict
from stats import ngramcounting, conceptstats
from pipeline.evaluation import gold_standard_concepts
import math
import seaborn as sns
import pandas as pd

CORPUS_NAME = 'craft'
FREQ_THRESHOLD = 1
MODEL_NAME = CORPUS_NAME
MODEL_SPEC = '_noskip_all'

corpus = dataio.load_corpus(CORPUS_NAME.lower())
dcs = {dc for doc in corpus
       for dc in doc.get_annotations(anno.DiscontinuousConcept)}

gap_bigrams = defaultdict(list)
other_bigrams = defaultdict(list)
for dc in dcs:
    tokens = dc.get_concept_tokens()
    norm_concept = dc.normalized_concept()
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i+1]
        bigram = norm_concept[i:i + 2]
        if t2.span[0] - t1.span[-1] > 2:
            gap_bigrams[bigram].append(dc)
        else:
            other_bigrams[bigram].append(dc)

model = ngramcounting.NgramModel.load_model(MODEL_NAME, MODEL_SPEC)
data_dict = {'bigram': [], 'pmi': [], 'mi': [], 'll': [], 'occurrence': []}
all_bigrams = set(gap_bigrams.keys()).union(set(other_bigrams.keys()))
for bigram in all_bigrams:
    # skip if not frequent enough
    if model.freq(bigram) < FREQ_THRESHOLD:
        continue

    # normalized form of bigram
    data_dict['bigram'].append(bigram)

    # pmi
    pmi = conceptstats.ngram_pointwise_mutual_information(bigram[0], bigram[1],
                                                          model)
    data_dict['pmi'].append(pmi)

    mi = conceptstats.ngram_mutual_information(bigram[0], bigram[1], model)
    data_dict['mi'].append(mi)

    ll = conceptstats.ngram_log_likelihood_ratio(bigram[0], bigram[1], model)
    data_dict['ll'].append(math.log(ll) if not ll <= 0 else None)

    if bigram in gap_bigrams and bigram in other_bigrams: type_ = 'both'
    elif bigram in gap_bigrams: type_ = 'only_gap'
    else: type_ = 'only_cont'
    data_dict['occurrence'].append(type_)

data = pd.DataFrame(data_dict)

sns.boxplot(x='occurrence', y='ll', data=data, showfliers=False)
