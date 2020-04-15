from datautils import dataio as dio
from stats import ngramcounting as ngc
from tqdm import tqdm
import os
import sys
import multiprocessing as mp

N_CORE = os.cpu_count() - 1

concepts = {concept for concept in dio.load_mesh_terms() if len(concept) > 1}

out_file = 'test'


def n_concepts_in_doc(doc_id):
    doc = dio.load_pmc_document(doc_id)
    ngrams = {ngram for s in doc.get_annotations('Sentence')
              for ngram in ngc.make_ngrams([t.lemma() for t in s.get_tokens()],
                                           min_n=2, max_n=5)}
    global concepts
    return len(set.intersection(ngrams, concepts))


def concept_checker(in_queue, out_queue):
    while True:  # keep going until reaching the "stop sign", i.e. a None object
        doc_id = in_queue.get()  # get next doc id
        if not doc_id:  # a None was drawn from the queue
            # means that there are no more docs: send signal to master counter
            # that no more docs will come from this worker
            out_queue.put(None)
            break
        else:
            out_queue.put((doc_id, n_concepts_in_doc(doc_id)))


def merger(out_file_name, in_queue):
    working_workers = N_CORE  # active workers
    with open(out_file_name, 'w+') as out:
        while True:  # keep going until reaching the "stop sign", a None object
            next_string = in_queue.get()  # get next id
            if not next_string:  # a None was drawn from the queue
                working_workers -= 1  # notify that a worker has sent stop sign
                if working_workers == 0:  # when all workers are done
                    break
            else:  # an actual id
                print(next_string, file=out)


if __name__ == '__main__':
    # prepare queues
    print('Initializing ...')
    id_queue = mp.Queue(maxsize=N_CORE)
    verified_queue = mp.Queue(maxsize=10)

    # set up child processes and go
    merger_process = mp.Process(target=merger, args=(out_file, verified_queue))
    merger_process.start()
    with mp.Pool(initializer=concept_checker,
                 initargs=(id_queue, verified_queue)) as pool:
        for id_ in tqdm(dio.pmc_corpus_ids(), desc='Running over documents'):
            id_queue.put(id_)  # give ids to workers
        for _ in range(N_CORE):  # tell workers we're done
            id_queue.put(None)
        merger_process.join()
