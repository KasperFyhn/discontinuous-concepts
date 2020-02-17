from corpusstats import stats
from datautils import dataio as dio
import nltk
import pandas as pd
import seaborn as sns
from tqdm import tqdm

CORPUS = 'genia'
MODEL_SPEC = '_skip_min1'
FREQ_THRESHOLD = 3

model = stats.NgramModel.load_model(CORPUS, MODEL_SPEC)

print('Excluding discontinuous concepts: ', end='')
gold_concepts = stats.gold_standard_concepts(
    dio.load_genia_corpus(as_generator=True), allow_discontinuous=False)
print('Only discontinuous concepts: ', end='')
disc_concepts = stats.gold_standard_concepts(
    dio.load_genia_corpus(as_generator=True)).difference(gold_concepts)
cont_concept_bigrams = {bigram for ngram in gold_concepts
                        for bigram in nltk.bigrams(ngram)}
bigrams_in_dc = {bigram for ngram in disc_concepts
                 for bigram in nltk.bigrams(ngram)}
concept_bigrams = cont_concept_bigrams.union(bigrams_in_dc)

bigrams = []
freqs = []
is_concept_bigram = []
in_dc = []
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
    if bigram in bigrams_in_dc:
        extra_count = sum(model.freq(skipgram)
                          for skipgram in model.skipgrams_with(bigram))
        # extra_count = 0
    else:
        extra_count = 0
    pmi = stats.pointwise_mutual_information(word_a, word_b, model, extra_count)
    pmis.append(pmi)
    contingency_table = model.contingency_table(word_a, word_b)
    contingency_table.a_b += extra_count
    mi = stats.mutual_information(contingency_table)
    mis.append(mi)
    ll = stats.log_likelihood_ratio(contingency_table)
    lls.append(ll)

    freqs.append(count)
    is_concept_bigram.append(bigram in cont_concept_bigrams)
    in_dc.append(bigram in bigrams_in_dc)

data = pd.DataFrame({'bigram': bigrams, 'in_concept': is_concept_bigram,
                     'in_dc': in_dc, 'pmi': pmis, 'mi': mis, 'll': lls})

sns.boxplot(x='in_concept', y='pmi', hue='in_dc', data=data, showfliers=False)

