import colibricore
from datautils import dataio as dio, annotations as anno
from tqdm import tqdm
from nltk import WordNetLemmatizer
from corpusstats import ngramcounting
import os
import multiprocessing as mp

DATA_FOLDER = os.path.dirname(__file__) + '/ngramdata/'
LEMMA = WordNetLemmatizer().lemmatize
N_CORE = os.cpu_count()


def tokenized_text_from_doc(doc: anno.Document):
    string_builder = ''
    for sent in doc.get_annotations('Sentence'):
        tokenized_sent = ' '.join(
            LEMMA(t.get_covered_text().lower().replace(' ', '_'),
                  pos=t.mapped_pos()) if t.mapped_pos() in 'anvr'
            else LEMMA(t.get_covered_text().lower().replace(' ', '_'))
            for t in doc.get_annotations_at(sent.span, 'Token')
        )
        string_builder += tokenized_sent + '\n'
    return string_builder.strip()


def _text_tokenizer(in_queue, out_queue, doc_loader):
    while True:  # keep going until reaching the "stop sign", i.e. a None object
        id_ = in_queue.get()  # get next doc id
        if not id_:  # a None was drawn from the queue
            # means that there are no more docs: send signal to master counter
            # that no more docs will come from this worker
            out_queue.put(None)
            break

        # load the doc, count and pass on
        d = doc_loader(id_)
        tokenized_text = tokenized_text_from_doc(d)
        out_queue.put(tokenized_text)


def _string_merger(out_file_name, in_queue):
    out_file = open(out_file_name, 'w+')
    working_workers = N_CORE  # active workers
    while True:  # keep going until reaching the "stop sign", i.e. a None object
        next_string = in_queue.get()  # get next counter
        if not next_string:  # a None was drawn from the queue
            working_workers -= 1  # notify that a worker has sent a "stop sign"
            if working_workers == 0:  # when all workers are done
                break
        else:  # an actual string
            print(next_string, file=out_file)
    out_file.close()


def encode_corpus(name, corpus_ids, doc_loader):

    # prepare filenames
    tokenized_corpus_file = DATA_FOLDER + name + '_tokenized.txt'
    class_file = DATA_FOLDER + name + '.colibri.cls'
    encoded_corpus_file = DATA_FOLDER + name + '.colibri.dat'

    # prepare queues
    path_queue = mp.Queue(maxsize=N_CORE)
    string_queue = mp.Queue(maxsize=10)

    # set up child processes and go
    writer_process = mp.Process(target=_string_merger,
                                args=(tokenized_corpus_file, string_queue))
    writer_process.start()
    pool = mp.Pool(initializer=_text_tokenizer,
                   initargs=(path_queue, string_queue, doc_loader))
    for doc in tqdm(corpus_ids, desc='Creating raw text corpus file'):
        path_queue.put(doc)  # give paths to workers
    for _ in range(N_CORE):  # tell workers we're done
        path_queue.put(None)
    pool.close()
    pool.join()
    writer_process.join()

    print('Making colibri class file ...')
    encoder = colibricore.ClassEncoder()
    encoder.build(tokenized_corpus_file)
    encoder.save(class_file)

    print('Encoding corpus ...')

    encoder.encodefile(tokenized_corpus_file, encoded_corpus_file)


def make_colibri_model(name, model_spec_name='', model_options=None):
    encoded_corpus_file = DATA_FOLDER + name + '.colibri.dat'
    model_file = DATA_FOLDER + name + model_spec_name + '.colibri.patternmodel'

    if not model_options:
        model_options = colibricore.PatternModelOptions(mintokens=0,
                                                        maxlength=5)
    model = colibricore.UnindexedPatternModel()
    model.train(encoded_corpus_file, model_options)
    model.write(model_file)

    return model


def load_colibri_model(name, model_spec_name=''):
    model_file = DATA_FOLDER + name + model_spec_name + '.colibri.patternmodel'
    class_file = DATA_FOLDER + name + '.colibri.cls'

    model = colibricore.UnindexedPatternModel(model_file)
    encoder = colibricore.ClassEncoder(class_file)
    decoder = colibricore.ClassDecoder(class_file)

    return model, encoder, decoder



