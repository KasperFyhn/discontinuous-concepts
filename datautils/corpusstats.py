import timeit
from tqdm import tqdm
import multiprocessing as mp
from datautils import dataio, annotations as anno
from collections import Counter, defaultdict
import nltk

WORD_TO_ID = {}
ID_TO_WORD = {}


class NgramCounter(Counter):

    def __init__(self):
        super().__init__()
        self.next = defaultdict(NgramCounter)

    @staticmethod
    def ngram_to_ids(ngram):
        return tuple(WORD_TO_ID[w] for w in ngram)

    @staticmethod
    def ids_to_ngram(ids):
        return tuple(ID_TO_WORD[w] for w in ids)

    @staticmethod
    def make_ngrams(tokens, min_n=1, max_n=5):
        """Make n-grams from a list of tokens."""
        n_grams = []
        for n in range(min_n, max_n + 1):
            n_grams += list(nltk.ngrams(tokens, n))
        return n_grams

    @classmethod
    def from_ngrams(cls, n_grams):
        """Make an NgramCounter object from a list or generator of n-grams."""
        counter = cls()
        for ngram in n_grams:
            counter.add_ngram(ngram)
        return counter

    @classmethod
    def from_token_lists(cls, token_lists, n_docs=None, min_n=1, max_n=5):
        """Make an NgramCounter object from a list of token lists."""
        if not n_docs:
            try:
                n_docs = len(token_lists)
            except TypeError:
                print('Number of documents is not known; set to 0.')
                n_docs = 0

        counter = cls()
        for i, tokens in enumerate(token_lists):
            print(f'Counting n-grams in docs: {i+1} of {n_docs}', end='\r')
            for ngram in cls.make_ngrams(tokens, min_n, max_n):
                counter.add_ngram(ngram)
        print()
        return counter

    @classmethod
    def from_documents(cls, docs, n_docs=None, min_n=1, max_n=5):
        """Make an NgramCounter object from a list of documents."""

        token_lists = [
            [t.get_covered_text() for t in doc.get_annotations(anno.Token)]
            for doc in docs
                       ]
        return cls.from_token_lists(token_lists, n_docs, min_n, max_n)

    def generate_ngrams(self, of_length=None, prev=None):
        """Generate n-grams from the NgramCounter's trie-like structure.
        Optional arguments are a specified length and a specified
        start-sequence which should only be used from its corresponding
        sub-counter, e.g. counter.after(bigram).generate_ngrams(prev=bigram)."""
        if not prev:
            prev = []
        for word in self.keys():
            n_gram = prev + [word]
            if not of_length or len(n_gram) == of_length:
                yield tuple(n_gram)
            if not of_length or len(n_gram) < of_length:
                yield from self.after(word).generate_ngrams(of_length=of_length,
                                                            prev=n_gram)

    def total_counts(self):
        """Total sum of counts. NOTE: Counted on the go, so it is a bit slow."""
        return sum(self.values())\
               + sum(c.total_counts() for c in self.next.values())

    def counts_of_length_n(self, n):
        """Sum of counts of n-grams of length n. NOTE: Counted on the go, so it
        is a bit slow."""

        def count_ngrams(counter, prev):
            correct_length = len(prev) == n - 1
            for word, count in counter.items():
                n_gram = prev + [word]
                if correct_length:
                    yield count
                else:
                    yield from count_ngrams(counter.after(word), n_gram)

        return sum(count_ngrams(self, []))

    def after(self, n_gram):
        """Return the NgramCounter after the given n-gram."""
        if isinstance(n_gram, str):
            return self.next[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self.next[n_gram[0]]
        else:
            return self.next[n_gram[0]].after(n_gram[1:])

    def add_ngram(self, n_gram, count=1):
        """Add count(s) of the n-gram."""
        # n_gram = NgramCounter.ngram_to_ids(n_gram)
        if len(n_gram) == 1:
            self[n_gram[0]] += count
        else:
            self.after(n_gram[0]).add_ngram(n_gram[1:], count=count)

    def freq(self, n_gram):
        """Return the count of the n-gram."""
        if isinstance(n_gram, str):
            return self[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self[n_gram[0]]
        return self.after(n_gram[0]).freq(n_gram[1:])

    def update_recursively(self, another_counter):
        """"Trickle-down" update this counter with another NgramCounter object.
        Use this instead of regular update."""
        super().update(another_counter)
        for key, next_counter in another_counter.next.items():
            self.next[key].update(next_counter)

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


class CorpusStats:

    def __init__(self, documents, n_docs=None, max_n_gram=5):
        print('Initializing CorpusStats object ...')
        self.ngram_counter = NgramCounter.from_documents(
            documents, n_docs=n_docs, max_n=max_n_gram
        )
        self._total_ngram_freqs = {
            n: self.ngram_counter.counts_of_length_n(n)
            for n in range(1, max_n_gram + 1)
        }

    def raw_freq(self, n_gram):
        return self.ngram_counter.freq(n_gram)

    def freq(self, n_gram):
        total = self._total_ngram_freqs[len(n_gram)]
        return self.raw_freq(n_gram) / total


if __name__ == '__main__':
    corpus = dataio.load_genia_corpus()
    start = timeit.default_timer()
    test = NgramCounter.from_documents(corpus)
    end = timeit.default_timer()
    print('Time elapsed:', end - start)
