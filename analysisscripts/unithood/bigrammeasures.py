from stats import conceptstats, ngramcounting
from datautils import dataio as dio
import math
import nltk
import pandas as pd
import seaborn as sns
from tqdm import tqdm

CORPUS = 'genia'
MODEL_SPEC = '_skip_all'
FREQ_THRESHOLD = 3
SKIPGRAMS = True

load_corpus = dio.load_genia_corpus if CORPUS.lower() == 'genia' \
    else dio.load_craft_corpus if CORPUS.lower() == 'craft'\
    else dio.load_genia_corpus
corpus = load_corpus()

model = ngramcounting.NgramModel.load_model(CORPUS, MODEL_SPEC)

print('Excluding discontinuous concepts: ', end='')
gold_concepts = conceptstats.gold_standard_concepts(corpus,
                                                    allow_discontinuous=False)
print('Only discontinuous concepts: ', end='')
disc_concepts = conceptstats.gold_standard_concepts(corpus).difference(
    gold_concepts)
cc_bigrams = {bigram for ngram in gold_concepts
              for bigram in nltk.bigrams(ngram)}
dc_bigrams = {bigram for ngram in disc_concepts
              for bigram in nltk.bigrams(ngram)}
both_dc_and_cc = cc_bigrams.intersection(dc_bigrams)
concept_bigrams = cc_bigrams.union(dc_bigrams)


bigrams = []
freqs = []
skip_freqs = []
is_concept_bigram = []
concept_types = []
pmis = []
mis = []
lls = []
for bigram_pattern, count in tqdm(
        model.iterate(2, threshold=FREQ_THRESHOLD, encoded_patterns=True),
        desc='Making calculations of association measures'):
    bigram = model.decode_pattern(bigram_pattern)
    bigrams.append(bigram)
    word_a = bigram_pattern[0]
    word_b = bigram_pattern[1]
    contingency_table = model.contingency_table(word_a, word_b,
                                                skipgrams=SKIPGRAMS)
    pmi = conceptstats.pointwise_mutual_information(contingency_table)
    pmis.append(pmi)
    # mi = conceptstats.mutual_information(contingency_table)
    # mis.append(mi)
    ll = conceptstats.log_likelihood_ratio(contingency_table)
    lls.append(ll)
    freqs.append(count)
    skipgram_count = sum(model.freq(skipgram)
                         for skipgram in model.skipgrams_with(bigram))
    skip_freqs.append(skipgram_count)

    is_concept_bigram.append(bigram in concept_bigrams)
    concept_types.append('both' if bigram in both_dc_and_cc
                         else 'cc' if bigram in cc_bigrams
                         else 'dc' if bigram in dc_bigrams
                         else 'neither')


data = pd.DataFrame({'bigram': bigrams, 'in_concept': is_concept_bigram,
                     'type': concept_types, 'pmi': pmis, 'll': lls,
                     'frequency': freqs, 'skipgram_frequency': skip_freqs})

sns.boxplot(x='type', y='frequency', data=data, showfliers=False)

cc_pmi = [v for v in data.where(data['type'] == 'cc')['pmi']
          if not math.isnan(v)]
both_pmi = [v for v in data.where(data['type'] == 'both')['pmi']
            if not math.isnan(v)]
neither_pmi = [v for v in data.where(data['type'] == 'neither')['pmi']
               if not math.isnan(v)]

in_concept_pmi = [v for v in data.where(data['in_concept'] == True)['pmi']
                  if not math.isnan(v)]
outside_concept_pmi = [v for v in data.where(data['in_concept'] == False)['pmi']
                       if not math.isnan(v)]


cc_ll = [v for v in data.where(data['type'] == 'cc')['ll']
          if not math.isnan(v)]
both_ll = [v for v in data.where(data['type'] == 'both')['ll']
            if not math.isnan(v)]
neither_ll = [v for v in data.where(data['type'] == 'neither')['ll']
               if not math.isnan(v)]

in_concept_ll = [v for v in data.where(data['in_concept'] == True)['ll']
                  if not math.isnan(v)]
outside_concept_ll = [v for v in data.where(data['in_concept'] == False)['ll']
                       if not math.isnan(v)]
