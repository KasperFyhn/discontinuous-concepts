import math
import sys
from nltk import WordNetLemmatizer
from tqdm import tqdm
import datautils.annotations as anno
from corpusstats.ngramcounting import NgramCounter
import corpusstats.ngramcounting as ngc
import multiprocessing as mp
import corpusstats.ngrammodel as ngm
import colibricore as cc

################################################################################
# CONVENIENCE CLASSES
################################################################################


class NgramModel:
    """A wrapper of a Colibri Core n-gram model with its appropriate encoder
    and decoder."""

    def __init__(self, model: cc.UnindexedPatternModel,
                 encoder: cc.ClassEncoder, decoder: cc.ClassDecoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def load_model(cls, name, model_spec_name=''):
        model, encoder, decoder = ngm.load_model(name, model_spec_name)
        return cls(model, encoder, decoder)

    def decode_pattern(self, pattern):
        return tuple(pattern.tostring(self.decoder).split())

    def iterate(self, n, threshold=0, encoded_patterns=False):
        if encoded_patterns:
            return self.model.filter(threshold, size=n)
        else:
            return ((p.tostring(self.decoder), c)
                    for p, c in self.model.filter(threshold, size=n))

    def freq(self, ngram):
        if isinstance(ngram, tuple):
            ngram = ' '.join(ngram)
        if isinstance(ngram, str):
            ngram = self.encoder.buildpattern(ngram)
        return self.model.occurrencecount(ngram)

    def prob(self, ngram, smoothing=1):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        return (self.freq(ngram) + smoothing)\
               / (self.total_counts(len(ngram)) + smoothing)

    def total_counts(self, of_length):
        return self.model.totaloccurrencesingroup(n=of_length)

    def contingency_table(self, ngram_a, ngram_b, smoothing=1, base=None):
        if isinstance(ngram_a, tuple):
            ngram_a = self.encoder.buildpattern(' '.join(ngram_a))
        elif isinstance(ngram_a, str):
            ngram_a = self.encoder.buildpattern(ngram_a)
        if isinstance(ngram_b, tuple):
            ngram_b = self.encoder.buildpattern(' '.join(ngram_b))
        elif isinstance(ngram_b, str):
            ngram_b = self.encoder.buildpattern(ngram_b)
        if not base:
            base = len(ngram_a)
        n = self.model.totaloccurrencesingroup(n=base)
        a_b = self.model.occurrencecount(ngram_a + ngram_b) + smoothing
        a_not_b = self.model.occurrencecount(ngram_a) - a_b + smoothing * 2
        not_a_b = self.model.occurrencecount(ngram_b) - a_b + smoothing * 2
        not_a_not_b = n - a_not_b - not_a_b - a_b + smoothing * 4
        return ContingencyTable(a_b, a_not_b, not_a_b, not_a_not_b)


class ContingencyTable:

    def __init__(self, a_b, a_not_b, not_a_b, not_a_not_b):
        self.a_b = a_b
        self.a_not_b = a_not_b
        self.not_a_b = not_a_b
        self.not_a_not_b = not_a_not_b
        self.n = self.a_b + self.a_not_b + self.not_a_b + self.not_a_not_b

    def all_cells(self):
        return [self.a_b, self.a_not_b, self.not_a_b, self.not_a_not_b]

    def marginal_a(self):
        return self.a_b + self.a_not_b

    def marginal_not_a(self):
        return self.not_a_b + self.not_a_not_b

    def marginal_b(self):
        return self.a_b + self.not_a_b

    def marginal_not_b(self):
        return self.a_not_b + self.not_a_not_b


################################################################################
# STATISTICAL MEASURES
################################################################################


# LOG-LIKELIHOOD
def calculate_ngram_log_likelihoods(ngram_pairs, model, smoothing=1):
    ngram_lls = {}
    print('Calculating log-likelihood for n-gram pairs')
    for pair in tqdm(ngram_pairs, file=sys.stdout):
        ng1, ng2 = pair
        ngram_lls[pair] = ngram_log_likelihood(ng1, ng2, model, smoothing)
    return ngram_lls


def ngram_log_likelihood(ngram1, ngram2, model, smoothing=1):
    table = model.contingency_table(ngram1, ngram2, smoothing)
    k_1 = table.a_b
    k_2 = table.a_not_b
    n_1 = table.marginal_b()
    n_2 = table.marginal_not_b()

    return log_likelihood_ratio(k_1, k_2, n_1, n_2)


def log_likelihood_ratio(k_1, k_2, n_1, n_2):
    """The binomial case of the log-likelihood ratio (see Dunning 1993)."""
    p_1 = k_1 / n_1
    p_2 = k_2 / n_2
    p = (k_1 + k_2) / (n_1 + n_2)
    return 2 * (log_likelihood(p_1, k_1, n_1) + log_likelihood(p_2, k_2, n_2)
                - log_likelihood(p, k_1, n_1) - log_likelihood(p, k_2, n_2))


def log_likelihood(p, k, n, log_base=math.e):
    """The binomial case of log-likelihood, calculated as in Dunning (1993)"""
    try:
        return k * math.log(p, log_base) + (n - k) * math.log(1 - p, log_base)
    except ValueError as e:
        print('Log-likelihood calculations go wrong if p=1 or p=0. '
              'Try smoothing perhaps?')
        raise e

# # see Dunning 1993
# A_B, A_notB, notA_B, notA_notB = 110, 2442, 111, 29114
# # (A, B), (A, not B), (A, B) + (not A, B), (A, not B) + (not A, not B)
# test1 = log_likelihood_ratio(A_B, A_notB, A_B + notA_B, A_notB + notA_notB, 0)
# # (A, B), (not A, B), (A, B) + (A, not B), (not A, B) + (not A, not B)
# test2 = log_likelihood_ratio(A_B, notA_B, A_B + A_notB, notA_B + notA_notB, 0)
# print(round(test1, 2) == round(test2, 2) == 270.72)


# MUTUAL INFORMATION
def calculate_mutual_information(ngram_pairs, model):
    return {pair: ngram_mutual_information(pair[0], pair[1], model)
            for pair in ngram_pairs}


def ngram_mutual_information(ngram1, ngram2, model, smoothing=1):
    contingency_table = model.contingency_table(ngram1, ngram2, smoothing)
    return mutual_information(contingency_table)


def mutual_information(contingency_table: ContingencyTable):
    p_ngram1 = contingency_table.marginal_a() / contingency_table.n
    p_ngram2 = contingency_table.marginal_b() / contingency_table.n
    mi = 0
    for jc in contingency_table.all_cells():
        joint_prob = jc / contingency_table.n
        mi += joint_prob * math.log(joint_prob / (p_ngram1 * p_ngram2))
    return mi


def calculate_pointwise_mutual_information(ngram_pairs, model):
    return {pair: pointwise_mutual_information(pair[0], pair[1], model)
            for pair in ngram_pairs}


def pointwise_mutual_information(ngram_a, ngram_b, model, smoothing=1):
    p_x_and_y = model.prob(ngram_a + ngram_b, smoothing)
    p_x = model.prob(ngram_a, smoothing)
    p_y = model.prob(ngram_b, smoothing)
    return math.log(p_x_and_y / (p_x * p_y))


# C-VALUE
def calculate_c_values(candidate_terms: list, threshold: float,
                       counter: NgramCounter):
    """Return a dict of terms and their C-values, calculated as in Frantzi et
    al. (2000). If a term's C-value is below the threshold, it will not be used
    for further calculations, but be included in the returned dict with a value
    of -1."""
    # make sure that the candidate terms list is sorted
    print('Sorting candidate terms for C-value calculation ...')
    candidate_terms = sorted(candidate_terms, key=lambda x: len(x),
                             reverse=True)
    final_terms = {}
    nested_ngrams = {t: set() for t in candidate_terms}
    print('Calculating C-values')
    for t in tqdm(candidate_terms, file=sys.stdout):
        c = c_value(t, nested_ngrams, counter)
        if c >= threshold:
            final_terms[t] = c
            for ng in NgramCounter.make_ngrams(t, max_n=len(t)-1):
                if ng in nested_ngrams:
                    nested_ngrams[ng].add(t)
        else:
            final_terms[t] = -1
    return final_terms


def c_value(term: tuple, nested_ngrams: dict, counter: NgramCounter):
    """Calculate C-value as in Frantzi et al. (2000)."""
    if not nested_ngrams[term]:
        return math.log2(len(term)) * counter.freq(term)
    else:
        nested_in = nested_ngrams[term]
        return math.log2(len(term)) * (counter.freq(term) - sum(
            rectified_freq(s, nested_ngrams, counter) for s in nested_in
        ) / len(nested_in))


def rectified_freq(ngram: tuple, nested_ngrams: dict, counter: NgramCounter):
    """Return the frequency of the ngram occurring as non-nested."""
    if not nested_ngrams[ngram]:
        return counter.freq(ngram)
    else:
        return counter.freq(ngram) - sum(
            rectified_freq(sg, nested_ngrams, counter)
            for sg in nested_ngrams[ngram])


# TF-IDF
def calculate_tf_idf_values(candidate_terms, docs, counter, n_docs=None):
    term_frequency = {t: counter.freq(t) for t in candidate_terms}
    doc_frequency = {t: 0 for t in candidate_terms}
    if not n_docs:
        try:
            n_docs = len(docs)
        except Exception as e:
            print('Could not get number of docs for TF-IDF calculations, thus '
                  'making it impossible!')
            raise e

    print('Gathering numbers for TF-IDF calculations')
    with mp.Pool() as pool:
        for dc in tqdm(pool.imap_unordered(ngc.count_ngrams_in_doc, docs),
                       total=n_docs, file=sys.stdout):
            for term in dc.keys():
                if term in doc_frequency:
                    doc_frequency[term] += 1

    tf_idf_values = {t: tf_idf(term_frequency[t], doc_frequency[t], n_docs)
                     for t in candidate_terms}
    return tf_idf_values


def tf_idf(tf, df, n_docs):
    return tf * n_docs / (df + 1)


# PERFORMANCE MEASURES
def gold_standard_concepts(corpus, allow_discontinuous=True):
    print('Retrieving gold standard concepts ...')
    lemmatize = WordNetLemmatizer().lemmatize
    all_concepts = set()
    for doc in corpus:
        concepts = doc.get_annotations(anno.Concept)
        for c in concepts:
            # normalize to lemmaed version
            if allow_discontinuous and isinstance(c, anno.DiscontinuousConcept):
                c_tokens = [t for span in c.spans
                            for t in doc.get_annotations_at(span, anno.Token)]
            elif isinstance(c, anno.DiscontinuousConcept):
                continue  # skip DiscontinuousConcept if not allowed
            else:
                c_tokens = [t for t in doc.get_annotations_at(c.span,
                                                              anno.Token)]

            lemmaed_concept = tuple(
                lemmatize(w.get_covered_text().lower().replace(' ', '_'),
                          pos=w.mapped_pos()) if w.mapped_pos() in 'anvr'
                else lemmatize(w.get_covered_text().lower().replace(' ', '_'))
                for w in c_tokens
            )
            normalized_concept = tuple(w.lower() for w in lemmaed_concept)
            all_concepts.add(normalized_concept)
    return all_concepts


def recall_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(exp)


def precision_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(pred)


def f1_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def performance(predicted, expected):
    precision = precision_types(predicted, expected)
    recall = recall_types(predicted, expected)
    f1 = f1_measure(precision, recall)
    print('Precision:  ', round(precision, 3))
    print('Recall:     ', round(recall, 3))
    print('F1-measure: ', round(f1, 3))
