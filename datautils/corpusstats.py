import timeit

from tqdm import tqdm
import multiprocessing as mp
from datautils import dataio
from collections import Counter, defaultdict
import nltk


class CorpusStats:

    def __init__(self):
        pass


class NgramCounter(Counter):

    def __init__(self):
        super().__init__()
        self.next = defaultdict(NgramCounter)

    @classmethod
    def from_n_grams(cls, n_grams):
        counter = cls()
        for ngram in n_grams:
            counter.add_ngram(ngram)
        return counter

    def after(self, n_gram):
        if isinstance(n_gram, str):
            n_gram = n_gram.split()
        if len(n_gram) == 1:
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
            n_gram = n_gram.split()
        if len(n_gram) == 1:
            return self[n_gram[0]]
        else:
            return self.after(n_gram[0]).freq(n_gram[1:])

    def update_recursively(self, another_counter):
        super().update(another_counter)
        for key, next_counter in another_counter.next.items():
            self.next[key].update(next_counter)


def get_n_grams(text, max_n=5):

    if isinstance(text, str):
        text = text.split()
    n_grams = []
    for n in range(1, max_n + 1):
        n_grams += list(nltk.ngrams(text, n))

    return n_grams


def _counter(text):
    n_grams = get_n_grams(text)
    c = Counter(n_grams)
    return c


def ngram_counter_of_corpus(texts, multithreading=False, n_texts=None):

    master_counter = NgramCounter()
    if not n_texts:
        n_texts = len(texts)

    if multithreading:
        print('Counting n-grams in texts ...')
        for c in tqdm((c for c in mp.Pool().imap_unordered(_counter, texts)),
                      total=n_texts):
            for ngram, ngram_count in c.items():
                master_counter.add_ngram(ngram, ngram_count)

        return master_counter
    else:
        for i, text in enumerate(texts):
            print(f'Counting n-grams in texts: {i+1} of {n_texts}', end='\r')
            master_counter.update_recursively(
                NgramCounter.from_n_grams(get_n_grams(text))
            )
        print()

    return master_counter


corpus = dataio.load_genia_corpus()
test = ngram_counter_of_corpus((d.get_text() for d in corpus),
                               multithreading=True, n_texts=len(corpus))
disc_concepts = Counter(c.get_concept()
                        for doc in corpus
                        for c in doc.get_annotations('DiscontinuousConcept'))

for dc, count in disc_concepts.most_common(20):
    print(f'{dc:25}{count:4}{test.freq(dc):4}')

