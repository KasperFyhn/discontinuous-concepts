import math
import sys
from nltk import WordNetLemmatizer
import tqdm
import datautils.annotations as anno
from corpusstats.ngramcounting import NgramCounter
import corpusstats.ngramcounting as ngc


################################################################################
# STATISTICAL MEASURES
################################################################################

# LOG-LIKELIHOOD
from datautils import dataio


def log_likelihood(p, k, n, log_base=math.e):
    """The binomial case of log-likelihood, calculated as in Dunning (1993)"""
    return k * math.log(p, log_base) + (n - k) * math.log(1 - p, log_base)


def log_likelihood_ratio(k_1, k_2, n_1, n_2):
    """The binomial case of the log-likelihood ratio (see Dunning 1993)."""
    p_1 = k_1 / n_1
    p_2 = k_2 / n_2
    p = (k_1 + k_2) / (n_1 + n_2)
    return 2 * (log_likelihood(p_1, k_1, n_1) + log_likelihood(p_2, k_2, n_2)
                - log_likelihood(p, k_1, n_1) - log_likelihood(p, k_2, n_2))


def log_likelihood_ratio_ngrams(ngram1: tuple, ngram2: tuple,
                                counter: NgramCounter):
    k_1 = counter.after(ngram1).freq(ngram2)
    k_2 = sum(counter.after(ng).freq(ngram2)
              for ng in counter.generate_ngrams(len(ngram1))
              if not ng == ngram1)
    n_1 = counter.after(ngram1).total_counts(len(ngram2))
    n_2 = sum(counter.after(ng).total_counts(len(ngram2))
              for ng in counter.generate_ngrams(len(ngram1))
              if not ng == ngram1)

    return log_likelihood_ratio(k_1, k_2, n_1, n_2)

# see Dunning 1993
# A_B, A_notB, notA_B, notA_notB = 110, 2442, 111, 29114
# (A, B), (A, not B), (A, B) + (not A, B), (A, not B) + (not A, not B)
# test1 = log_likelihood_ratio(A_B, A_notB, A_B + notA_B, A_notB + notA_notB)
# (A, B), (not A, B), (A, B) + (A, not B), (not A, B) + (not A, not B)
# test2 = log_likelihood_ratio(A_B, notA_B, A_B + A_notB, notA_B + notA_notB)
# round(test1, 2) == round(test2, 2) == 270.72 >>> True
# ngs = {('A',): A_B + A_notB, ('B',): A_B + notA_B,
#        ('notA',): notA_B + notA_notB, ('notB',): A_notB + notA_notB,
#        ('A', 'B'): A_B, ('A', 'notB'): A_notB,
#        ('notA', 'B'): notA_B, ('notA', 'notB'): notA_notB}
# nc = NgramCounter()
# for ng, c in ngs.items():
#     nc.add_ngram(ng, count=c)
# print(nc.log_likelihood_ratio(('A',), ('B',)))


# MUTUAL INFORMATION


# C-VALUE
def calculate_c_values(candidate_terms: list, threshold: float,
                       counter: NgramCounter):
    """Return a dict of terms and their C-values, calculated as in Frantzi et
    al. (2000)."""
    # make sure that the candidate terms list is sorted
    print('Sorting candidate terms for C-value calculation ...')
    candidate_terms = sorted(candidate_terms, key=lambda x: len(x),
                             reverse=True)
    final_terms = {}
    nested_ngrams = {t: set() for t in candidate_terms}
    print('Calculating C-values')
    for t in tqdm.tqdm(candidate_terms):
        c = c_value(t, nested_ngrams, counter)
        if c >= threshold:
            final_terms[t] = c
            for ng in NgramCounter.make_ngrams(t, max_n=len(t)-1):
                if ng in nested_ngrams:
                    nested_ngrams[ng].add(t)
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
def calculate_tf_idf_values(candidate_terms: list, docs: list, n_docs=None):
    term_frequency = {t: 0 for t in candidate_terms}
    doc_frequency = {t: 0 for t in candidate_terms}
    if not n_docs:
        try:
            n_docs = len(docs)
        except Exception as e:
            print('Could not get number of docs for TF-IDF calculations, thus '
                  'making it impossible!')
            raise e
    print('Gathering numbers for TF-IDF calculations')
    for d in tqdm.tqdm(docs, total=n_docs, file=sys.stdout):
        doc_counter = ngc.count_ngrams_in_doc(d)
        for t in candidate_terms:
            count = doc_counter[t]
            term_frequency[t] += count
            if count:
                doc_frequency[t] += 1

    tf_idf_values = {t: tf_idf(term_frequency[t], doc_frequency[t], n_docs)
                     for t in candidate_terms}
    return tf_idf_values


def tf_idf(tf, df, n_docs):
    return tf * n_docs / (df + 1)


# PERFORMANCE MEASURES
def gold_standard_concepts(corpus):
    lemmatizer = WordNetLemmatizer()
    all_concepts = set()
    for doc in corpus:
        concepts = doc.get_annotations(anno.Concept)
        for c in concepts:
            # normalize to lemmaed version
            if isinstance(c, anno.DiscontinuousConcept):
                c_tokens = [t for span in c.spans
                            for t in doc.get_annotations_at(span, anno.Token)]
            else:
                c_tokens = [t for t in doc.get_annotations_at(c.span,
                                                              anno.Token)]
            if not c_tokens:
                print(c, doc)
                print(c.get_context())
                input()
            lemmaed_concept = tuple(
                lemmatizer.lemmatize(w.get_covered_text(), anno.POS_TAG_MAP[w])
                if anno.POS_TAG_MAP[w] in 'anvr'
                else lemmatizer.lemmatize(w.get_covered_text())
                for w in c_tokens
            )
            all_concepts.add(lemmaed_concept)
    return all_concepts


ngrams = ngc.main('GENIA')
genia = dataio.load_genia_corpus()
gold = gold_standard_concepts(genia)

