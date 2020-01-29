import multiprocessing as mp
import os
import tqdm
from datautils import dataio
import timeit
import psutil
from collections import Counter
from corpusstats.stats import NgramCounter, calculate_c_values


###############################################################################
# This script creates n-gram count data from a given corpus.
###############################################################################

FREQUENCY_THRESHOLD = 1
MIN_N = 1
MAX_N = 5

N_CORE = os.cpu_count()
MAX_RAM_USAGE_PERCENT = 90

_LOAD_DOC_FUNCTIONS = {'craft': dataio.load_craft_document,
                       'genia': dataio.load_genia_document,
                       'pmc': dataio.load_pmc_document}
_CORPUS_IDS = {'craft': dataio.craft_corpus_ids,
               'genia': dataio.genia_corpus_ids,
               'pmc': dataio.pmc_corpus_ids}


def count(corpus_name):
    """Main function of the module which returns an NgramCounter object of """

    print('Initializing ...')
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
    for doc in tqdm.tqdm(corpus):  # give paths to workers
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
    for ngram, cnt in tqdm.tqdm(master_counter, total=n_types):
        ngram_counter.add_ngram(ngram, cnt)

    end = timeit.default_timer()
    secs = int(end - start)
    mins = secs // 60
    secs = secs % 60
    hours = mins // 60
    mins = mins % 60
    print(f'Time elapsed: {hours}h {mins}m {secs}s')

    return ngram_counter


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

        # load the doc and the corresponding annotations
        d = doc_loader(id_)

        ngrams = []
        # loop over sentences and make n-grams within them
        for sentence in d.get_annotations('Sentence'):
            # get tokens from the sentence span and normalize them
            tokens = [t.get_covered_text().lower() + '/' + t.pos
                      for t in d.get_annotations_at(sentence.span,
                                                    annotation_type='Token')]
            # make n-grams (1-5)
            ngrams += NgramCounter.make_ngrams(tokens, min_n=MIN_N, max_n=MAX_N)

        # count the n-grams and pass it on to the master counter
        c = Counter(ngrams)
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
                    {key: value for key, value in tqdm.tqdm(master.items())
                     if value > min_f})
                n_items_after = len(master)
                print(f'Cleared out {n_items_before - n_items_after} n-grams '
                        'with a frequency lower than', min_f)
                min_f += 1  # if not enough, clear out even more

    if not counted == n_docs:
        print(f'Updated with {counted} of {n_docs} documents only!')
    print('Filtering out infrequent n-grams')
    master = Counter({key: value for key, value in tqdm.tqdm(master.items())
                      if value >= FREQUENCY_THRESHOLD})
    out_queue.put(master)  # pass on master counter


if __name__ == '__main__':
    final_counter = count('genia')
    c_values = calculate_c_values((ng for ng in final_counter.generate_ngrams()
                                   if ng[-1][-2:] == 'NN'
                                   and len(ng) > 1), 5, final_counter)
