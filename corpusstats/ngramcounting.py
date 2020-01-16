import multiprocessing as mp
import os
import tqdm
from datautils import dataio
import timeit
import nltk
from collections import Counter


N_CORE = os.cpu_count()


def process(in_queue, out_queue):
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
        d = dataio.load_pmc_document(id_)

        # loop over sentences and make n-grams within them
        for sentence in d.get_annotations('Sentence'):
            # get tokens from the sentence span and normalize them
            tokens = [t.get_covered_text().lower()
                      for t in d.get_annotations_at(sentence.span,
                                                    annotation_type='Token')]
            # make n-grams (1-5)
            ngrams = []
            for n in range(1, 6):
                ngrams += list(nltk.ngrams(tokens, n))

        # count the n-grams and pass it on to the master counter
        c = Counter(ngrams)
        out_queue.put(c)


def counter(in_queue, out_queue, n_docs):
    """Master counter process:
    1) retrieve Counter objects from the queue, passed on 'process' workers
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
                print()
                print(f'Updated with doc {counted} of {n_docs}!')
                print('Finished counting!')
                out_queue.put(master)  # pass on master counter
                break

        else:  # an actual counter was drawn
            master.update(next_counter)
            counted += 1
            # print(f'Updated with doc {counted} of {n_docs}!', end='\r')


if __name__ == '__main__':
    start = timeit.default_timer()
    corpus = dataio.pmc_corpus_ids()

    # prepare queues for use between concurrent processes
    path_queue = mp.Queue(maxsize=N_CORE)
    counter_queue = mp.Queue(maxsize=10)
    final_counter_queue = mp.Queue()

    # initialize master counter process
    counter_process = mp.Process(
        target=counter, args=(counter_queue, final_counter_queue, len(corpus))
    )
    counter_process.start()

    # intialize pool of "process" workers
    pool = mp.Pool(initializer=process, initargs=(path_queue, counter_queue))
    for doc in tqdm.tqdm(corpus):  # give paths to workers
        path_queue.put(doc)
    for _ in range(N_CORE):  # tell workers we're done
        path_queue.put(None)
    pool.close()
    pool.join()

    # retrieve the master counter object
    master_counter = final_counter_queue.get()

    end = timeit.default_timer()
    print('Time elapsed:', round((end - start) / 60, 2), 'min')