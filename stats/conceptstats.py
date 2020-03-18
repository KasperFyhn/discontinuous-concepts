import math
import sys
from nltk import WordNetLemmatizer
from tqdm import tqdm
from datautils import annotations as anno, dataio
from stats import ngramcounting
from stats.ngramcounting import NgramModel, ContingencyTable, make_ngrams
import multiprocessing as mp


################################################################################
# CONCEPT MEASURES
################################################################################


# LOG-LIKELIHOOD
def ngram_log_likelihood_ratio(ngram1, ngram2, model, smoothing=1):
    table = model.contingency_table(ngram1, ngram2, smoothing)
    return log_likelihood_ratio(table)


def log_likelihood_ratio(contingency_table: ContingencyTable):
    """The binomial case of the log-likelihood ratio (see Dunning 1993)."""
    k_1 = contingency_table.a_b
    k_2 = contingency_table.not_a_b
    n_1 = contingency_table.marginal_a()
    n_2 = contingency_table.marginal_not_a()
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
        print('Log-likelihood calculations go wrong if p = 1 or p = 0. '
              'Try smoothing perhaps?')
        raise e


# # see Dunning 1993
# A_B, A_notB, notA_B, notA_notB = 110, 2442, 111, 29114
# test1 = log_likelihood_ratio(ContingencyTable(A_B, notA_B, A_notB, notA_notB))
# test2 = log_likelihood_ratio(ContingencyTable(A_B, A_notB, notA_B, notA_notB))
# print(round(test1, 2) == round(test2, 2) == 270.72)


# MUTUAL INFORMATION
def ngram_mutual_information(ngram1, ngram2, model, smoothing=1):
    contingency_table = model.contingency_table(ngram1, ngram2, smoothing)
    return mutual_information(contingency_table)


def mutual_information(contingency_table: ContingencyTable):
    mi = 0
    for jf, mf1, mf2 in contingency_table.iterate():
        joint_prob = jf / contingency_table.n()
        marginal_p1 = mf1 / contingency_table.n()
        marginal_p2 = mf2 / contingency_table.n()
        mi += joint_prob * math.log(joint_prob / (marginal_p1 * marginal_p2))
    return mi


def ngram_pointwise_mutual_information(ngram1, ngram2, model, smoothing=1):
    p_x_and_y = model.prob(ngram1 + ngram2, smoothing=smoothing)
    p_x = model.prob(ngram1, smoothing=smoothing)
    p_y = model.prob(ngram2, smoothing=smoothing)
    return math.log(p_x_and_y / (p_x * p_y))


def pointwise_mutual_information(contingency_table: ContingencyTable):
    p_x_and_y = contingency_table.a_b / contingency_table.n()
    p_x = contingency_table.marginal_a() / contingency_table.n()
    p_y = contingency_table.marginal_b() / contingency_table.n()
    return math.log(p_x_and_y / (p_x * p_y))


# C-VALUE
def calculate_c_values(candidate_terms: list, threshold: float,
                       model: NgramModel, skipgrams=False):
    """
    Return a dict of terms and their C-values, calculated as in Frantzi et
    al. (2000). If a term's C-value is below the threshold, it will not be used
    for further calculations, but be included in the returned dict.
    :param candidate_terms: list of tuples containing candidate terms
    :param threshold: C-value treshold; can be a floating point number
    :param model: NgramModel-like object with a .freq(ngram) method
    :param skipgrams: Whether to include skipgram counts or not
    :return: dict of candidate terms and their C-value
    """
    # make sure that the candidate terms list is sorted
    print('Calculating C-values')
    candidate_terms = sorted(candidate_terms, key=lambda x: len(x),
                             reverse=True)
    final_terms = {}
    nested_ngrams = {t: set() for t in candidate_terms}
    for t in tqdm(candidate_terms, file=sys.stdout):
        c = c_value(t, nested_ngrams, model, skipgrams=skipgrams)
        final_terms[t] = c
        if c >= threshold:
            for ng in make_ngrams(t, max_n=len(t)-1):
                if ng in nested_ngrams:
                    nested_ngrams[ng].add(t)

    return final_terms


def c_value(term: tuple, nested_ngrams: dict, model, skipgrams=False):
    """Calculate C-value as in Frantzi et al. (2000)."""
    if not nested_ngrams[term]:
        return math.log2(len(term)) * model.freq(term, skipgrams)
    else:
        nested_in = nested_ngrams[term]
        return math.log2(len(term)) * (model.freq(term, skipgrams) - sum(
            rectified_freq(s, nested_ngrams, model, skipgrams)
            for s in nested_in) / len(nested_in))


def rectified_freq(ngram: tuple, nested_ngrams: dict, model, skipgrams=False):
    """Return the frequency of the ngram occurring as non-nested."""
    if not nested_ngrams[ngram]:
        return model.freq(ngram, skipgrams)
    else:
        return model.freq(ngram) - sum(
            rectified_freq(sg, nested_ngrams, model, skipgrams)
            for sg in nested_ngrams[ngram])


# TF-IDF
def calculate_tf_idf_values(candidate_terms, docs, model, n_docs=None):
    term_frequency = {t: model.freq(t) for t in candidate_terms}
    doc_frequency = {t: 0 for t in candidate_terms}
    if not n_docs:
        try:
            n_docs = len(docs)
        except TypeError as e:
            print('Could not get number of docs for TF-IDF calculations, thus '
                  'making it impossible!')
            raise e

    print('Gathering numbers for TF-IDF calculations')
    with mp.Pool() as pool:
        for dc in tqdm(
                pool.imap_unordered(ngramcounting.count_ngrams_in_doc, docs),
                total=n_docs, file=sys.stdout):
            for term in dc.keys():
                if term in doc_frequency:
                    doc_frequency[term] += 1

    tf_idf_values = {t: tf_idf(term_frequency[t], doc_frequency[t], n_docs)
                     for t in candidate_terms}
    return tf_idf_values


def tf_idf(tf, df, n_docs):
    return tf * n_docs / (df + 1)


# WEIRDNESS and GLOSSEX
_brown_model = None  # used by weirdness()


def weirdness(term, target_model, reference_model=None, smoothing=1):
    """
    Returns Weirdness measure as calculated in Ahmad et al. (1999)
    :param term: the term; can be multi-word term.
    :param target_model: NgramModel for the target corpus.
    :param reference_model: NgramModel for the reference corpus. Default is the
    Brown corpus from NLTK.
    :param smoothing: smoothing of counts; if set to 0, it may result in a
    ZeroDivisionError.
    :return: positive float
    """

    if not reference_model:
        global _brown_model
        if not _brown_model:
            print('Loading reference model for the first time.')
            _brown_model = NgramModel.load_model('brown', '_noskip_all')
        reference_model = _brown_model

    target_freq = target_model.freq(term) + smoothing
    reference_freq = reference_model.freq(term) + smoothing
    target_length = target_model.total_counts(1) + smoothing
    reference_length = reference_model.total_counts(1) + smoothing

    return (target_freq * reference_length) / (reference_freq * target_length)


def glossex(term, target_model, reference_model=None, smoothing=1):
    return sum(math.log(weirdness(w, target_model, reference_model, smoothing))
               for w in term) / len(term)


# PERFORMANCE MEASURES
def recall_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(exp)


def precision_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(pred)


def precision_at_k(predicted_ranked: list, expected: set, k_values):
    for k in k_values:
        if k > len(predicted_ranked):
            k = len(predicted_ranked)
        predicted_at_k = set(predicted_ranked[:k])
        prec_at_k = len(predicted_at_k.intersection(expected)) / k
        print(f'P@{k:<6}', prec_at_k)
        if k == len(predicted_ranked):
            break


def f1_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def performance(predicted, expected):
    precision = precision_types(predicted, expected)
    recall = recall_types(predicted, expected)
    f1 = f1_measure(precision, recall)
    print('Precision:  ', round(precision, 3))
    print('Recall:     ', round(recall, 3))
    print('F1-measure: ', round(f1, 3))
