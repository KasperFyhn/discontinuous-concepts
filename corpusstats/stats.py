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
from typing import Union

################################################################################
# CONVENIENCE CLASSES
################################################################################


class NgramModel:
    """
    A wrapper of a Colibri Core n-gram model with its appropriate encoder
    and decoder.
    """

    def __init__(self, model: Union[cc.UnindexedPatternModel,
                                    cc.IndexedPatternModel],
                 encoder: cc.ClassEncoder, decoder: cc.ClassDecoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self._skip_counts = {}

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

    def freq(self, ngram, include_skipgrams=False):
        if isinstance(ngram, tuple):
            ngram = ' '.join(ngram)
        if isinstance(ngram, str):
            ngram = self.encoder.buildpattern(ngram)
        if not include_skipgrams:
            return self.model.occurrencecount(ngram)
        else:
            skipgram_counts = sum(self.skipgrams_with(ngram).values())
            return self.model.occurrencecount(ngram) + skipgram_counts

    def prob(self, ngram, smoothing=1):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        return (self.freq(ngram) + smoothing)\
               / (self.total_counts(len(ngram)) + smoothing)

    def total_counts(self, of_length, skipgrams=False):
        if not skipgrams:
            return self.model.totaloccurrencesingroup(n=of_length)
        else:
            if of_length not in self._skip_counts:
                skip_counts = 0
                for p, c in self.model.filter(0, size=0,
                                              category=cc.Category.SKIPGRAM):
                    if len(p) - p.skipcount() == of_length:
                        skip_counts += c
                self._skip_counts[of_length] = skip_counts
            else:
                skip_counts = self._skip_counts[of_length]
            return self.model.totaloccurrencesingroup(n=of_length) + skip_counts

    @staticmethod
    def _skipgram_combinations(obl_left: list, skips_left: int, previous=None):
        if not previous:
            previous = []
        combos = []
        if obl_left:
            with_obl = previous + obl_left[:1]
            if len(obl_left) == 1 and not skips_left:
                combos.append(with_obl)
            else:
                for c in NgramModel._skipgram_combinations(
                        obl_left[1:], skips_left, with_obl):
                    combos.append(c)
        if skips_left:
            with_skip = previous + ['{*}']
            if skips_left == 1 and not obl_left:
                combos.append(with_skip)
            else:
                for c in NgramModel._skipgram_combinations(
                        obl_left, skips_left-1, with_skip):
                    combos.append(c)

        return combos

    def skipgrams_with(self, ngram, min_skips=None, max_size=None):
        if isinstance(ngram, cc.Pattern):
            ngram = ngram.tostring(self.decoder)
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        if not min_skips:
            min_skips = 1
        if not max_size:
            max_size = self.model.maxlength()
        max_skips = max_size - len(ngram)
        obligatory = list(ngram[1:-1])
        skipgrams = []
        for n_skips in range(min_skips, max_skips+1):
            skipgrams += [
                ngram[:1] + tuple(sg) + ngram[-1:] for sg
                in NgramModel._skipgram_combinations(obligatory, n_skips)
            ]
        skipgrams_in_model = {sg: self.freq(sg) for sg in skipgrams
                              if self.freq(sg) > 0}
        return skipgrams_in_model

    def contingency_table(self, ngram_a, ngram_b, smoothing=1, skipgrams=False):
        """
        :param ngram_a:
        :param ngram_b:
        :param smoothing:
        :param skipgrams:
        :return:
        """
        if isinstance(ngram_a, tuple):
            ngram_a = self.encoder.buildpattern(' '.join(ngram_a))
        elif isinstance(ngram_a, str):
            ngram_a = self.encoder.buildpattern(ngram_a)
        if isinstance(ngram_b, tuple):
            ngram_b = self.encoder.buildpattern(' '.join(ngram_b))
        elif isinstance(ngram_b, str):
            ngram_b = self.encoder.buildpattern(ngram_b)

        n = self.total_counts(1)  # TODO: re-evaluate this decision!
        a_b = self.freq(ngram_a + ngram_b, skipgrams) + smoothing
        a_not_b = self.freq(ngram_a, skipgrams) - a_b + smoothing * 2
        if a_not_b <= 0: a_not_b = smoothing
        not_a_b = self.freq(ngram_b, skipgrams) - a_b + smoothing * 2
        if not_a_b <= 0: not_a_b = smoothing
        not_a_not_b = n - a_not_b - not_a_b - a_b + smoothing * 4

        return ContingencyTable(a_b, a_not_b, not_a_b, not_a_not_b)


class IndexedNgramModel(NgramModel):

    def __init__(self, model: cc.IndexedPatternModel, encoder: cc.ClassEncoder,
                 decoder: cc.ClassDecoder):
        super().__init__(model, encoder, decoder)

    def left_neighbours(self, size):
        return self.model.getleftneighbours(size=size)

    def contingency_table(self, ngram_a, ngram_b, smoothing=1,
                          based_on_lower_order=True, same_order_threshold=3):
        """
        :param ngram_a:
        :param ngram_b:
        :param smoothing:
        :param based_on_lower_order:
        If True (default), counts are based on len(a)-grams except for (a, b);
        If False, all counts are based on len(a+b)-grams. NOTE: If set to false,
        it is MUCH slower!
        :param same_order_threshold:
        :return:
        """
        if isinstance(ngram_a, tuple):
            ngram_a = self.encoder.buildpattern(' '.join(ngram_a))
        elif isinstance(ngram_a, str):
            ngram_a = self.encoder.buildpattern(ngram_a)
        if isinstance(ngram_b, tuple):
            ngram_b = self.encoder.buildpattern(' '.join(ngram_b))
        elif isinstance(ngram_b, str):
            ngram_b = self.encoder.buildpattern(ngram_b)

        if not based_on_lower_order:
            base = len(ngram_a + ngram_b)
            n = self.model.totaloccurrencesingroup(n=base)
            a_b, a_not_b, not_a_b, not_a_not_b = (smoothing,) * 4
            split = len(ngram_a)
            cooc_ngram = ngram_a + ngram_b
            for ngram, count in self.iterate(n, threshold=same_order_threshold):
                if ngram == cooc_ngram: a_b += count
                elif ngram[:split] == ngram_a: a_not_b += count
                elif ngram[split:] == ngram_b: not_a_b += count
                else: not_a_not_b += count

        else:  # default
            base = len(ngram_a)
            n = self.model.totaloccurrencesingroup(n=base)
            a_b = self.model.occurrencecount(ngram_a + ngram_b) + smoothing
            a_not_b = self.model.occurrencecount(ngram_a) - a_b + smoothing * 2
            not_a_b = self.model.occurrencecount(ngram_b) - a_b + smoothing * 2
            not_a_not_b = n - a_not_b - not_a_b - a_b + smoothing * 4

        return ContingencyTable(a_b, a_not_b, not_a_b, not_a_not_b)


class ContingencyTable:

    def __init__(self, a_b, a_not_b, not_a_b, not_a_not_b):
        for count in (a_b, a_not_b, not_a_b, not_a_not_b):
            if count <= 0:
                raise ValueError("ContingencyTables can't have negative counts")
        self.a_b = a_b
        self.a_not_b = a_not_b
        self.not_a_b = not_a_b
        self.not_a_not_b = not_a_not_b

    def iterate(self):
        return [(self.a_b, self.marginal_a(), self.marginal_b()),
                (self.not_a_b, self.marginal_not_a(), self.marginal_b()),
                (self.a_not_b, self.marginal_a(), self.marginal_not_b()),
                (self.not_a_not_b, self.marginal_not_a(), self.marginal_not_b())
                ]

    def n(self):
        return self.a_b + self.a_not_b + self.not_a_b + self.not_a_not_b

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
def ngram_log_likelihood_ratio(ngram1, ngram2, model, smoothing=1):
    table = model.contingency_table(ngram1, ngram2, smoothing)
    return log_likelihood_ratio(table)


def log_likelihood_ratio(contingency_table):
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


def ngram_pointwise_mutual_information(ngram_a, ngram_b, model, smoothing=1,
                                       extra_count=0):
    p_x_and_y = (model.freq(ngram_a + ngram_b) + extra_count + smoothing)\
                / model.total_counts(len(ngram_a))
    p_x = (model.freq(ngram_a) + smoothing)\
          / model.total_counts(len(ngram_a))
    p_y = (model.freq(ngram_b) + smoothing)\
          / model.total_counts(len(ngram_b))
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
            for ng in NgramCounter.make_ngrams(t, max_n=len(t)-1):
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
def calculate_tf_idf_values(candidate_terms, docs, counter, n_docs=None):
    term_frequency = {t: counter.freq(t) for t in candidate_terms}
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
    print('Retrieving gold standard concepts ...', end=' ', flush=True)
    lemmatize = WordNetLemmatizer().lemmatize
    all_concepts = set()
    skipped = set()
    for doc in corpus:
        concepts = doc.get_annotations(anno.Concept)
        for c in concepts:
            if allow_discontinuous and isinstance(c, anno.DiscontinuousConcept):
                c_tokens = [t for span in c.spans
                            for t in doc.get_annotations_at(span, anno.Token)]
            elif isinstance(c, anno.DiscontinuousConcept):
                continue  # skip DiscontinuousConcept if not allowed
            else:
                c_tokens = [t for t in doc.get_annotations_at(c.span,
                                                              anno.Token)]
            # concept span does not equal token span, e.g. if only
            # part of a token constitutes a concept
            if len(c_tokens) == 0 or \
                not (c.span[0] == c_tokens[0].span[0]
                     and c.span[1] == c_tokens[-1].span[1]):
                skipped.add(c.get_covered_text())
                continue
            #  normalize to lemmaed version
            lemmaed_concept = tuple(
                lemmatize(w.get_covered_text().lower().replace(' ', '_'),
                          pos=w.mapped_pos()) if w.mapped_pos() in 'anvr'
                else lemmatize(w.get_covered_text().lower().replace(' ', '_'))
                for w in c_tokens
            )
            all_concepts.add(lemmaed_concept)
    print(f'Skipped {len(skipped)} concepts not bounded at tokens boundaries.')
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


if __name__ == '__main__':
    ngram_model = NgramModel.load_model('genia', '_skip_min1')

