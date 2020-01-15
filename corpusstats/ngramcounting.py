import multiprocessing as mp
import os
import tqdm
from datautils import dataio
import timeit
import nltk
from collections import Counter


N_CORE = os.cpu_count()


def process(in_queue, out_queue):
    while True:
        id_ = in_queue.get()
        if not id_:
            out_queue.put(None)
            break
        d = dataio.load_pmc_document(id_)
        sentences = (s for s in d.get_annotations('Sentence'))
        for sentence in sentences:
            tokens = [t.get_covered_text().lower()
                      for t in d.get_annotations_at(sentence.span,
                                                    annotation_type='Token')]
            ngrams = []
            for n in range(1, 6):
                ngrams += list(nltk.ngrams(tokens, n))

        c = Counter(ngrams)
        out_queue.put(c)


def update(in_queue, out_queue, n_docs):

    master = Counter()
    counted = 0
    working_workers = N_CORE
    while True:
        next_counter = in_queue.get()
        if not next_counter:
            working_workers -= 1
            if working_workers == 0:
                print()
                print(f'Updated with doc {counted} of {n_docs}!')
                print('Finished counting!')
                out_queue.put(master)
                break
        else:
            master.update(next_counter)
            counted += 1
        # print(f'Updated with doc {counted} of {n_docs}!', end='\r')


start = timeit.default_timer()

test_pmc_corpus = dataio.pmc_corpus_ids()

path_queue = mp.Queue(maxsize=N_CORE)
counter_queue = mp.Queue(maxsize=10)
final_counter_queue = mp.Queue()

counter_process = mp.Process(target=update, args=(counter_queue,
                                                  final_counter_queue,
                                                  len(test_pmc_corpus)))
counter_process.start()

pool = mp.Pool(initializer=process, initargs=(path_queue, counter_queue))
for doc in tqdm.tqdm(test_pmc_corpus):  # give paths to workers
    path_queue.put(doc)
for _ in range(N_CORE):  # tell workers we're done
    path_queue.put(None)
pool.close()
pool.join()

master_counter = final_counter_queue.get()
end = timeit.default_timer()

print('Time elapsed:', round((end - start) / 60, 2), 'min')