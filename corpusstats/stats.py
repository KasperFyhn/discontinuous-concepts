import math
import sys
from nltk import WordNetLemmatizer
from tqdm import tqdm
import datautils.annotations as anno
from corpusstats.ngramcounting import NgramCounter
import corpusstats.ngramcounting as ngc
import multiprocessing as mp

################################################################################
# STATISTICAL MEASURES
################################################################################


def contingency_table(ngram_a: tuple, ngram_b: tuple, counter: NgramCounter):
    a_b = counter.after(ngram_a).freq(ngram_b) + 1
    a_notb = counter.freq(ngram_a) - a_b + 1
    nota_b = counter.freq(ngram_b) - a_b + 1
    nota_notb = counter.total_counts(len(ngram_a) + len(ngram_b)) - a_b\
                - a_notb - nota_b + 1
    return {'a_b': a_b, 'a_notb': a_notb, 'nota_b': nota_b,
            'nota_notb': nota_notb}



# LOG-LIKELIHOOD
def calculate_ngram_log_likelihoods(ngram_pairs, counter: NgramCounter,
                                    smoothing=1):
    n_pairs = len(ngram_pairs)
    args_generator = ((pair, counter) for pair in ngram_pairs)
    ngram_lls = {}
    print('Calculating log-likelihood for n-gram pairs')
    for pair in tqdm(ngram_pairs, file=sys.stdout):
        ng1, ng2 = pair
        try:
            ngram_lls[pair] = ngram_log_likelihood(ng1, ng2, counter, smoothing)
        except:
            print(pair)
    return ngram_lls


def log_likelihood(p, k, n, log_base=math.e):
    """The binomial case of log-likelihood, calculated as in Dunning (1993)"""
    try:
        return k * math.log(p, log_base) + (n - k) * math.log(1 - p, log_base)
    except ValueError as e:
        print('Log-likelihood calculations go wrong if p=1 or p=0. '
              'Try smoothing perhaps?')
        raise e


def log_likelihood_ratio(k_1, k_2, n_1, n_2, smoothing=1):
    """The binomial case of the log-likelihood ratio (see Dunning 1993)."""
    k_1 += smoothing
    k_2 += smoothing
    n_1 += smoothing * 2
    n_2 += smoothing * 2
    p_1 = k_1 / n_1
    p_2 = k_2 / n_2
    p = (k_1 + k_2) / (n_1 + n_2)
    return 2 * (log_likelihood(p_1, k_1, n_1) + log_likelihood(p_2, k_2, n_2)
                - log_likelihood(p, k_1, n_1) - log_likelihood(p, k_2, n_2))


def ngram_log_likelihood(ngram1: tuple, ngram2: tuple,
                         counter: NgramCounter, smoothing=1):
    # k_1 = counter.after(ngram1).freq(ngram2)
    # k_2 = sum(counter.after(ng).freq(ngram2)
    #           for ng in counter.generate_ngrams(len(ngram1))
    #           if not ng == ngram1)
    # n_1 = counter.after(ngram1).total_counts(len(ngram2))
    # n_2 = sum(counter.after(ng).total_counts(len(ngram2))
    #           for ng in counter.generate_ngrams(len(ngram1))
    #           if not ng == ngram1)

    k_1 = counter.after(ngram1).freq(ngram2)
    k_2 = counter.freq(ngram2) - k_1
    n_1 = counter.freq(ngram1)
    n_2 = counter.total_counts(len(ngram1)) - n_1

    return log_likelihood_ratio(k_1, k_2, n_1, n_2, smoothing=smoothing)


# # see Dunning 1993
# A_B, A_notB, notA_B, notA_notB = 110, 2442, 111, 29114
# # (A, B), (A, not B), (A, B) + (not A, B), (A, not B) + (not A, not B)
# test1 = log_likelihood_ratio(A_B, A_notB, A_B + notA_B, A_notB + notA_notB, 0)
# # (A, B), (not A, B), (A, B) + (A, not B), (not A, B) + (not A, not B)
# test2 = log_likelihood_ratio(A_B, notA_B, A_B + A_notB, notA_B + notA_notB, 0)
# print(round(test1, 2) == round(test2, 2) == 270.72)


# MUTUAL INFORMATION
def calculate_mutual_information(ngram_pairs, counter: NgramCounter):
    mis = {}
    for pair in tqdm(ngram_pairs):
        ng1, ng2 = pair
        mis[pair] = mutual_information(ng1, ng2, counter)
    return mis


def mutual_information(ngram1: tuple, ngram2: tuple, counter: NgramCounter,
                       smoothing=1):
    table = contingency_table(ngram1, ngram2, counter)
    p_ngram1 = counter.freq(ngram1) / counter.total_counts(len(ngram1))
    p_ngram2 = counter.freq(ngram2) / counter.total_counts(len(ngram2))
    mi = 0
    for jc in table.values():
        # TODO: fix underflow or whatever's going wrong
        joint_prob = (jc + smoothing)\
                     / counter.total_counts(len(ngram1) + len(ngram2))
        try:
            mi += joint_prob * math.log(joint_prob / (p_ngram1 * p_ngram2))
        except:
            print('crap')
            return 0
    return mi


def calculate_pointwise_mutual_information(ngram_pairs, counter: NgramCounter):
    return {pair: pointwise_mutual_information(pair[0], pair[1], counter)
            for pair in ngram_pairs}


def pointwise_mutual_information(ngram1: tuple, ngram2: tuple,
                                 counter: NgramCounter):
    p_x = counter.freq(ngram1) / counter.total_counts(len(ngram1))
    p_y_given_x = counter.after(ngram1).freq(ngram2)
    p_x_and_y = p_x * p_y_given_x
    p_y = counter.freq(ngram2) / counter.total_counts(len(ngram2))
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
    lemmatizer = WordNetLemmatizer()
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
            if not c_tokens:
                # TODO: temporary fix due to some faulty gold concepts
                continue
            lemmaed_concept = tuple(
                lemmatizer.lemmatize(w.get_covered_text(), anno.POS_TAG_MAP[w])
                if anno.POS_TAG_MAP[w] in 'anvr'
                else lemmatizer.lemmatize(w.get_covered_text())
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
