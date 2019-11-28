from tqdm import tqdm
import multiprocessing as mp
from datautils import dataio, annotations as anno
from collections import Counter, defaultdict
import nltk


class NgramCounter(Counter):

    def __init__(self):
        super().__init__()
        self.next = defaultdict(NgramCounter)

    @staticmethod
    def _counter(doc):
        tokens = [t.get_covered_text() for t in doc.get_annotations(anno.Token)]
        n_grams = get_n_grams(tokens)
        c = Counter(n_grams)
        return c

    @classmethod
    def from_n_grams(cls, n_grams):
        counter = cls()
        for ngram in n_grams:
            counter.add_ngram(ngram)
        return counter

    @classmethod
    def from_documents(cls, docs, n_docs=None, multithreading=True):
        if not n_docs:
            try:
                n_docs = len(docs)
            except TypeError:
                print('Number of documents is not known; set to 0.')
                n_docs = 0

        counter = cls()
        if multithreading:
            print('Counting n-grams in texts ...')
            for c in tqdm(
                    (c for c in mp.Pool().imap_unordered(cls._counter, docs)),
                    total=n_docs):
                for ngram, ngram_count in c.items():
                    counter.add_ngram(ngram, ngram_count)
        else:
            for i, text in enumerate(docs):
                print(f'Counting n-grams in docs: {i+1} of {n_docs}', end='\r')
                counter.update_recursively(
                    NgramCounter.from_n_grams(get_n_grams(text))
                )
            print()
        return counter

    def n_grams_of_length(self, n):

        def _generate_ngrams(counter, prev):
            for word in counter.keys():
                n_gram = prev + [word]
                if len(n_gram) == n:
                    n_gram = tuple(n_gram)
                    for _ in range(self.freq(n_gram)):
                        yield tuple(n_gram)
                else:
                    yield from _generate_ngrams(counter.after(word), n_gram)

        return _generate_ngrams(self, [])

    def sum_all(self):
        return sum(self.values())\
               + sum(c.sum_all() for c in self.next.values())

    def after(self, n_gram):
        if isinstance(n_gram, str):
            return self.next[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self.next[n_gram[0]]
        else:
            return self.next[n_gram[0]].after(n_gram[1:])

    def add_ngram(self, n_gram, count=1):
        if len(n_gram) == 1:
            self[n_gram[0]] += count
        else:
            self.after(n_gram[0]).add_ngram(n_gram[1:], count=count)

    def freq(self, n_gram):
        if isinstance(n_gram, str):
            return self[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self[n_gram[0]]
        return self.after(n_gram[0]).freq(n_gram[1:])

    def update_recursively(self, another_counter):
        super().update(another_counter)
        for key, next_counter in another_counter.next.items():
            self.next[key].update(next_counter)


class CorpusStats:

    def __init__(self, documents, multithreading=True, n_docs=None,
                 max_n_gram=5):
        print('Initializing CorpusStats object ...')
        self.ngram_counter = NgramCounter.from_documents(
            documents, n_docs=n_docs, multithreading=multithreading
        )
        self._total_ngram_freqs = {
            n: sum(1 for _ in self.ngram_counter.n_grams_of_length(n))
            for n in range(1, max_n_gram + 5)
        }

    def raw_freq(self, n_gram):
        return self.ngram_counter.freq(n_gram)

    def freq(self, n_gram):
        total = self._total_ngram_freqs[len(n_gram)]
        return self.raw_freq(n_gram) / total


def get_n_grams(tokens, max_n=5):
    n_grams = []
    for n in range(1, max_n + 1):
        n_grams += list(nltk.ngrams(tokens, n))
    return n_grams


corpus = dataio.load_genia_corpus()
test = CorpusStats(corpus)
