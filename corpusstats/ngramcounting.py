import multiprocessing as mp
import os
import re
import sys
import nltk
from nltk.stem import WordNetLemmatizer
import tqdm
from datautils import dataio, annotations
import timeit
import psutil
from collections import Counter, defaultdict

###############################################################################
# This script creates n-gram count data from a given corpus. If done from
# another script, simply call main().
###############################################################################

# run configurations
FREQUENCY_THRESHOLD = 1
MIN_N = 1
MAX_N = 5

# resource management
N_CORE = os.cpu_count()
MAX_RAM_USAGE_PERCENT = 90

# for corpus loading
_LOAD_DOC_FUNCTIONS = {'craft': dataio.load_craft_document,
                       'genia': dataio.load_genia_document,
                       'pmc': dataio.load_pmc_document}
_CORPUS_IDS = {'craft': dataio.craft_corpus_ids,
               'genia': dataio.genia_corpus_ids,
               'pmc': dataio.pmc_corpus_ids}

# filtering
POS_TAG_MAP = annotations.POS_TAG_MAP
LEMMA = WordNetLemmatizer().lemmatize
FILTERS = {
    'UNSILO': re.compile('([na]|(ng)|(vn))+n'),
    'simple': re.compile('[an]+n'),
    'liberal': re.compile('[^x]*n')
}
FILTER = FILTERS['UNSILO']


def main(corpus_name):
    """Main function of the module which returns an NgramCounter object of
    n-grams in the given corpus.
    A bit unconventionally, configuration parameters are made as "constant"
    fields which can be changed either in the script or on the go because they
    are accessed globally from subprocesses. These are:
    FREQUENCY_THRESHOLD, MIN_N, MAX_N, FILTER=None (choose from FILTERS)."""
    print('Initializing ...')
    LEMMA('test')  # to initialize the lemmatizer
    start = timeit.default_timer()
    corpus_loader = _CORPUS_IDS[corpus_name.lower()]
    corpus = corpus_loader()

    # prepare queues for use between concurrent processes
    path_queue = mp.Queue(maxsize=N_CORE)
    counter_queue = mp.Queue(maxsize=10)
    final_counter_queue = mp.Queue()

    # initialize master counter process
    counter_process = mp.Process(
        target=_counter, args=(counter_queue, final_counter_queue, len(corpus))
    )
    counter_process.start()

    # initialize pool of "process" workers
    print('Counting n-grams in documents')
    doc_loader = _LOAD_DOC_FUNCTIONS[corpus_name.lower()]
    pool = mp.Pool(initializer=_process, initargs=(path_queue, counter_queue,
                                                   doc_loader))
    for doc in tqdm.tqdm(corpus, file=sys.stdout):  # give paths to workers
        path_queue.put(doc)
    for _ in range(N_CORE):  # tell workers we're done
        path_queue.put(None)
    pool.close()
    pool.join()

    # retrieve the master counter object and turn it into an NgramCounter
    master_counter = final_counter_queue.get()
    counter_process.join()
    print('Converting master Counter to NgramCounter')
    ngram_counter = NgramCounter()
    n_types = len(master_counter)
    master_counter = (kv for kv in master_counter.items())
    for ngram, cnt in tqdm.tqdm(master_counter, total=n_types, file=sys.stdout):
        ngram_counter.add_ngram(ngram, cnt)

    end = timeit.default_timer()
    secs = int(end - start)
    mins = secs // 60
    secs = secs % 60
    hours = mins // 60
    mins = mins % 60
    print(f'Time elapsed: {hours}h {mins}m {secs}s')

    return ngram_counter


class NgramCounter(Counter):
    """A counter object of n-grams stored as a trie-like structure and with
    convenience methods for making n-gram statistics.
    NOTE: Not safe for pickling!"""

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

    def add_ngram(self, n_gram: tuple, count=1):
        """Add count(s) of the n-gram."""
        if len(n_gram) == 1:
            self[n_gram[0]] += count
        else:
            self.after(n_gram[0]).add_ngram(n_gram[1:], count=count)

    def after(self, n_gram: tuple):
        """Return the NgramCounter after the given n-gram."""
        if isinstance(n_gram, str) or isinstance(n_gram, int):
            return self.next[n_gram]
        elif isinstance(n_gram, tuple) and len(n_gram) == 1:
            return self.next[n_gram[0]]
        else:
            return self.next[n_gram[0]].after(n_gram[1:])

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
        if not of_length or len(prev) < of_length:
            for next_word in self.next.keys():
                n_gram = list(prev) + [next_word]
                yield from self.after(next_word).generate_ngrams(
                    of_length=of_length, prev=n_gram
                )

    def total_counts(self, of_length):
        """Sum of counts of n-grams of length n. NOTE: Counted on the go, so it
        is a bit slow."""
        return sum(self.freq(ng) for ng in self.generate_ngrams(of_length))

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


def count_ngrams_in_doc(d, pos_tag_filter=FILTER, min_n=MIN_N, max_n=MAX_N):
    """Return a regular Counter object of n-grams in document d, filtered using
    a pos tag filter."""
    all_ngrams = []
    # loop over sentences and make n-grams within them
    for sentence in d.get_annotations('Sentence'):
        # get tokens with pos tags from the sentence span and normalize them
        tokens = [(t.get_covered_text().lower(), POS_TAG_MAP[t.pos])
                  for t in d.get_annotations_at(sentence.span, 'Token')]
        lemmaed_tokens = [(LEMMA(t[0]), t[1]) if t[1] not in 'anvr'
                          else (LEMMA(t[0], t[1]), t[1]) for t in tokens]

        # make n-grams (1-5)
        ngrams = NgramCounter.make_ngrams(lemmaed_tokens, min_n, max_n)
        for ng_with_pos in ngrams:
            if FILTER:
                pos_tag_sequence = ''.join(w[1] for w in ng_with_pos)
                if not re.fullmatch(pos_tag_filter, pos_tag_sequence):
                    continue
            ngram = tuple(w[0] for w in ng_with_pos)
            all_ngrams.append(ngram)

    # count the n-grams and pass it on to the master counter
    c = Counter(all_ngrams)
    return c


def _process(in_queue, out_queue, doc_loader):
    """Processing of a single document:
    1) get doc id from queue and open
    2) loop over sentences and tokens and create n-grams
    3) count n-grams in a counter object and pass it on to master counter

    There are multiple workers of this kind."""

    while True:  # keep going until reaching the "stop sign", i.e. a None object
        id_ = in_queue.get()  # get next doc id
        if not id_:  # a None was drawn from the queue
            # means that there are no more docs: send signal to master counter
            # that no more docs will come from this worker
            out_queue.put(None)
            break

        # load the doc, count and pass on
        d = doc_loader(id_)
        c = count_ngrams_in_doc(d)
        out_queue.put(c)


def _counter(in_queue, out_queue, n_docs):
    """Master counter process:
    1) retrieve Counter objects from the queue, passed on by 'process' workers
    2) update master Counter object with single doc Counter object
    3) when all docs have been counted and added together, pass the master
       Counter object on

    There is only one process doing this."""

    master = Counter()  # master counter object
    counted = 0  # tracker of counted docs
    working_workers = N_CORE  # active workers
    while True:  # keep going until reaching the "stop sign", i.e. a None object
        next_counter = in_queue.get()  # get next counter
        if not next_counter:  # a None was drawn from the queue
            working_workers -= 1  # notify that a worker has sent a "stop sign"
            if working_workers == 0:  # when all workers are done
                break
        else:  # an actual counter was drawn
            master.update(next_counter)
            counted += 1
            min_f = 2
            # n-gram types are insanely many which clogs up memory; the ones
            # with very few occurrences are not that interesting, though, and
            # can be filtered away (at least, that's a working assumption)
            while psutil.virtual_memory().percent > MAX_RAM_USAGE_PERCENT:
                print()
                print('Running out of memory. Filtering out most infrequent '
                      'n-grams to clear out space.')
                n_items_before = len(master)
                master = Counter(
                    {key: value for key, value in tqdm.tqdm(master.items(),
                                                            file=sys.stdout)
                     if value > min_f})
                n_items_after = len(master)
                print(f'Cleared out {n_items_before - n_items_after} n-grams '
                        'with a frequency lower than', min_f)
                min_f += 1  # if not enough, clear out even more

    if not counted == n_docs:
        print(f'Updated with {counted} of {n_docs} documents only!')
    print('Filtering out infrequent n-grams')
    master = Counter({key: value for key, value in tqdm.tqdm(master.items(),
                                                             file=sys.stdout)
                      if value >= FREQUENCY_THRESHOLD})
    out_queue.put(master)  # pass on master counter


if __name__ == '__main__':
    final_counter = main('genia')