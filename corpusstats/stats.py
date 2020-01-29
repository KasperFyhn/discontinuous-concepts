import timeit
from datautils import dataio, annotations as anno
from collections import Counter, defaultdict
import nltk
import math
import tqdm


class NgramCounter(Counter):

    def __init__(self, ngrams=None):
        super().__init__()
        self.next = defaultdict(NgramCounter)
        if ngrams:
            for ngram in ngrams:
                self.add_ngram(ngram)

    @staticmethod
    def make_ngrams(tokens, min_n=1, max_n=5):
        """Make n-grams from a list of tokens."""
        n_grams = []
        for n in range(min_n, max_n + 1):
            n_grams += list(nltk.ngrams(tokens, n))
        return n_grams

    def after(self, n_gram: tuple):
        """Return the NgramCounter after the given n-gram."""
        if isinstance(n_gram, str) or isinstance(n_gram, int):
            return self.next[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self.next[n_gram[0]]
        else:
            return self.next[n_gram[0]].after(n_gram[1:])

    def add_ngram(self, n_gram: tuple, count=1):
        """Add count(s) of the n-gram."""
        if len(n_gram) == 1:
            self[n_gram[0]] += count
        else:
            self.after(n_gram[0]).add_ngram(n_gram[1:], count=count)

    def freq(self, n_gram: tuple):
        """Return the count of the n-gram."""
        # str or int cases are from an earlier, more dynamic approach;
        # should probably just be deleted
        if isinstance(n_gram, str) or isinstance(n_gram, int):
            return self[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self[n_gram[0]]
        return self.after(n_gram[0]).freq(n_gram[1:])

    def generate_ngrams(self, of_length=None, prev=None):
        """Generate n-grams from the NgramCounter's trie-like structure.
        Optional arguments are a specified length and a specified
        start-sequence which should only be used from its corresponding
        sub-counter, e.g. counter.after(bigram).generate_ngrams(prev=bigram)."""
        if not prev:
            prev = []
        elif isinstance(prev, str):
            prev = [prev]
        for word in self.keys():
            n_gram = list(prev) + [word]
            if not of_length or len(n_gram) == of_length:
                yield tuple(n_gram)
            if not of_length or len(n_gram) < of_length:
                yield from self.after(word).generate_ngrams(of_length=of_length,
                                                            prev=n_gram)

    def total_counts(self, of_length):
        """Sum of counts of n-grams of length n. NOTE: Counted on the go, so it
        is a bit slow."""

        def count_ngrams(counter, prev):
            correct_length = len(prev) == of_length - 1
            for word, count in counter.items():
                n_gram = prev + [word]
                if correct_length:
                    yield count
                else:
                    yield from count_ngrams(counter.after(word), n_gram)

        return sum(count_ngrams(self, []))

    def most_common(self, of_length, n=None, prev=None):
        ngrams = self.generate_ngrams(of_length=of_length, prev=prev)
        return sorted(((ngram, self.freq(ngram)) for ngram in ngrams),
                      key=lambda pair: pair[1], reverse=True)[:n]

    def update_recursively(self, another_counter):
        """"Trickle-down" update this counter with another NgramCounter object.
        Use this instead of regular update."""
        super().update(another_counter)
        for key, next_counter in another_counter.next.items():
            self.next[key].update(next_counter)

    def log_likelihood_ratio(self, ngram1: tuple, ngram2: tuple):
        k_1 = self.after(ngram1).freq(ngram2)
        k_2 = sum(self.after(ng).freq(ngram2)
                  for ng in self.generate_ngrams(len(ngram1))
                  if not ng == ngram1)
        n_1 = self.after(ngram1).total_counts(len(ngram2))
        n_2 = sum(self.after(ng).total_counts(len(ngram2))
                  for ng in self.generate_ngrams(len(ngram1))
                  if not ng == ngram1)

        return log_likelihood_ratio(k_1, k_2, n_1, n_2)

    def save_to_file(self, path):
        """Save this NgramCounter to a file, stored in a regular dict
        structure."""
        print('Saving NgramCounter object to', path, '...')
        dictionary = {}
        for ngram in self.generate_ngrams():
            dictionary[ngram] = self.freq(ngram)
        with open(path, 'w+') as out:
            print(dictionary, file=out)

    @classmethod
    def from_file(cls, path):
        """Load an NgramCounter from a file where n-grams and their counts are
        stored in a regular dict structure."""
        print('Loading NgramCounter object from', path, '...')
        with open(path) as in_file:
            dictionary = eval(in_file.read())
        counter = cls()
        for ngram, count in dictionary.items():
            counter.add_ngram(ngram, count)
        return counter


################################################################################
# STATISTICAL MEASURES
################################################################################

# LOG-LIKELIHOOD
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
    nested_ngrams = defaultdict(list)
    print('Calculating C-values')
    for t in tqdm.tqdm(candidate_terms):
        c = c_value(t, nested_ngrams, counter)
        if c >= threshold:
            final_terms[t] = c
            for ng in NgramCounter.make_ngrams(t, max_n=len(t)-1):
                nested_ngrams[ng].append(t)
    return final_terms


def c_value(term: tuple, nested_ngrams: dict, counter: NgramCounter):
    """Calculate C-value as in Frantzi et al. (2000)."""
    if term not in nested_ngrams:
        return math.log2(len(term)) * counter.freq(term)
    else:
        nested_in = nested_ngrams[term]
        return math.log2(len(term)) * (counter.freq(term) - sum(
            rectified_freq(s, nested_ngrams, counter) for s in nested_in
        ) / len(nested_in))


def rectified_freq(ngram: tuple, nested_ngrams: dict, counter: NgramCounter):
    """Return the frequency of the ngram occurring as non-nested."""
    if ngram not in nested_ngrams:
        return counter.freq(ngram)
    else:
        return counter.freq(ngram) - sum(
            rectified_freq(sg, nested_ngrams, counter)
            for sg in nested_ngrams[ngram])




