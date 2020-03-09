import re
from collections import Counter

import colibricore as cc
import nltk

from datautils import dataio as dio, annotations as anno
from tqdm import tqdm
from nltk import WordNetLemmatizer
from typing import Union
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
    encoder = cc.ClassEncoder()
    encoder.build(tokenized_corpus_file)
    encoder.save(class_file)

    print('Encoding corpus ...')

    encoder.encodefile(tokenized_corpus_file, encoded_corpus_file)


def make_colibri_model(name, model_spec_name='', mintokens=0, maxlength=5,
                       skipgrams=False):
    encoded_corpus_file = DATA_FOLDER + name + '.colibri.dat'
    model_file = DATA_FOLDER + name + model_spec_name + '.colibri.patternmodel'

    model_options = cc.PatternModelOptions(mintokens=mintokens,
                                           maxlength=maxlength,
                                           doskipgrams_exhaustive=skipgrams)
    model = cc.UnindexedPatternModel()
    model.train(encoded_corpus_file, model_options)
    model.write(model_file)

    return model


def load_colibri_model(name, model_spec_name=''):
    model_file = DATA_FOLDER + name + model_spec_name + '.colibri.patternmodel'
    class_file = DATA_FOLDER + name + '.colibri.cls'

    model = cc.UnindexedPatternModel(model_file)
    encoder = cc.ClassEncoder(class_file)
    decoder = cc.ClassDecoder(class_file)

    return model, encoder, decoder


################################################################################
# CONVENIENCE CLASSES
################################################################################


class NgramModel:
    """
    A wrapper of a Colibri Core n-gram model with its appropriate encoder
    and decoder.
    """

    def __init__(self, model: Union[cc.UnindexedPatternModel,
                                    cc.IndexedPatternModel],
                 encoder: cc.ClassEncoder, decoder: cc.ClassDecoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self._skip_counts = {}

    @classmethod
    def load_model(cls, name, model_spec_name=''):
        model, encoder, decoder = load_colibri_model(name, model_spec_name)
        return cls(model, encoder, decoder)

    def decode_pattern(self, pattern):
        return tuple(pattern.tostring(self.decoder).split())

    def iterate(self, n, threshold=0, encoded_patterns=False):
        if encoded_patterns:
            return self.model.filter(threshold, size=n)
        else:
            return ((p.tostring(self.decoder), c)
                    for p, c in self.model.filter(threshold, size=n))

    def freq(self, ngram, include_skipgrams=False):
        if isinstance(ngram, tuple):
            ngram = ' '.join(ngram)
        if isinstance(ngram, str):
            ngram = self.encoder.buildpattern(ngram)
        if not include_skipgrams:
            return self.model.occurrencecount(ngram)
        else:
            skipgram_counts = sum(self.skipgrams_with(ngram).values())
            return self.model.occurrencecount(ngram) + skipgram_counts

    def prob(self, ngram, smoothing=1):
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        return (self.freq(ngram) + smoothing) \
               / (self.total_counts(len(ngram)) + smoothing)

    def total_counts(self, of_length, skipgrams=False):
        if not skipgrams:
            return self.model.totaloccurrencesingroup(n=of_length)
        else:
            if of_length not in self._skip_counts:
                skip_counts = 0
                for p, c in self.model.filter(0, size=0,
                                              category=cc.Category.SKIPGRAM):
                    if len(p) - p.skipcount() == of_length:
                        skip_counts += c
                self._skip_counts[of_length] = skip_counts
            else:
                skip_counts = self._skip_counts[of_length]
            return self.model.totaloccurrencesingroup(n=of_length) + skip_counts

    @staticmethod
    def _skipgram_combinations(obl_left: list, skips_left: int, previous=None):
        if not previous:
            previous = []
        combos = []
        if obl_left:
            with_obl = previous + obl_left[:1]
            if len(obl_left) == 1 and not skips_left:
                combos.append(with_obl)
            else:
                for c in NgramModel._skipgram_combinations(
                        obl_left[1:], skips_left, with_obl):
                    combos.append(c)
        if skips_left:
            with_skip = previous + ['{*}']
            if skips_left == 1 and not obl_left:
                combos.append(with_skip)
            else:
                for c in NgramModel._skipgram_combinations(
                        obl_left, skips_left - 1, with_skip):
                    combos.append(c)

        return combos

    def skipgrams_with(self, ngram, min_skips=None, max_size=None):
        if isinstance(ngram, cc.Pattern):
            ngram = ngram.tostring(self.decoder)
        if isinstance(ngram, str):
            ngram = tuple(ngram.split())
        if not min_skips:
            min_skips = 1
        if not max_size:
            max_size = self.model.maxlength()
        max_skips = max_size - len(ngram)
        obligatory = list(ngram[1:-1])
        skipgrams = []
        for n_skips in range(min_skips, max_skips+1):
            skipgrams += [
                ngram[:1] + tuple(sg) + ngram[-1:] for sg
                in NgramModel._skipgram_combinations(obligatory, n_skips)
            ]
        skipgrams_in_model = {sg: self.freq(sg) for sg in skipgrams
                              if self.freq(sg) > 0}
        return skipgrams_in_model

    def contingency_table(self, ngram_a, ngram_b, smoothing=1, skipgrams=False):
        """
        :param ngram_a:
        :param ngram_b:
        :param smoothing:
        :param skipgrams:
        :return:
        """
        if isinstance(ngram_a, tuple):
            ngram_a = self.encoder.buildpattern(' '.join(ngram_a))
        elif isinstance(ngram_a, str):
            ngram_a = self.encoder.buildpattern(ngram_a)
        if isinstance(ngram_b, tuple):
            ngram_b = self.encoder.buildpattern(' '.join(ngram_b))
        elif isinstance(ngram_b, str):
            ngram_b = self.encoder.buildpattern(ngram_b)

        n = self.total_counts(1)  # TODO: re-evaluate this decision!
        a_b = self.freq(ngram_a + ngram_b, skipgrams) + smoothing
        a_not_b = self.freq(ngram_a, skipgrams) - a_b + smoothing * 2
        if a_not_b <= 0: a_not_b = smoothing
        not_a_b = self.freq(ngram_b, skipgrams) - a_b + smoothing * 2
        if not_a_b <= 0: not_a_b = smoothing
        not_a_not_b = n - a_not_b - not_a_b - a_b + smoothing * 4

        return ContingencyTable(a_b, a_not_b, not_a_b, not_a_not_b)


class IndexedNgramModel(NgramModel):

    def __init__(self, model: cc.IndexedPatternModel, encoder: cc.ClassEncoder,
                 decoder: cc.ClassDecoder):
        super().__init__(model, encoder, decoder)

    def left_neighbours(self, size):
        return self.model.getleftneighbours(size=size)

    def contingency_table(self, ngram_a, ngram_b, smoothing=1,
                          based_on_lower_order=True, same_order_threshold=3):
        """
        :param ngram_a:
        :param ngram_b:
        :param smoothing:
        :param based_on_lower_order:
        If True (default), counts are based on len(a)-grams except for (a, b);
        If False, all counts are based on len(a+b)-grams. NOTE: If set to false,
        it is MUCH slower!
        :param same_order_threshold:
        :return:
        """
        if isinstance(ngram_a, tuple):
            ngram_a = self.encoder.buildpattern(' '.join(ngram_a))
        elif isinstance(ngram_a, str):
            ngram_a = self.encoder.buildpattern(ngram_a)
        if isinstance(ngram_b, tuple):
            ngram_b = self.encoder.buildpattern(' '.join(ngram_b))
        elif isinstance(ngram_b, str):
            ngram_b = self.encoder.buildpattern(ngram_b)

        if not based_on_lower_order:
            base = len(ngram_a + ngram_b)
            n = self.model.totaloccurrencesingroup(n=base)
            a_b, a_not_b, not_a_b, not_a_not_b = (smoothing,) * 4
            split = len(ngram_a)
            cooc_ngram = ngram_a + ngram_b
            for ngram, count in self.iterate(n, threshold=same_order_threshold):
                if ngram == cooc_ngram: a_b += count
                elif ngram[:split] == ngram_a: a_not_b += count
                elif ngram[split:] == ngram_b: not_a_b += count
                else: not_a_not_b += count

        else:  # default
            base = len(ngram_a)
            n = self.model.totaloccurrencesingroup(n=base)
            a_b = self.model.occurrencecount(ngram_a + ngram_b) + smoothing
            a_not_b = self.model.occurrencecount(ngram_a) - a_b + smoothing * 2
            not_a_b = self.model.occurrencecount(ngram_b) - a_b + smoothing * 2
            not_a_not_b = n - a_not_b - not_a_b - a_b + smoothing * 4

        return ContingencyTable(a_b, a_not_b, not_a_b, not_a_not_b)


class ContingencyTable:

    def __init__(self, a_b, a_not_b, not_a_b, not_a_not_b):
        for count in (a_b, a_not_b, not_a_b, not_a_not_b):
            if count <= 0:
                raise ValueError("ContingencyTables can't have negative counts")
        self.a_b = a_b
        self.a_not_b = a_not_b
        self.not_a_b = not_a_b
        self.not_a_not_b = not_a_not_b

    def iterate(self):
        return [(self.a_b, self.marginal_a(), self.marginal_b()),
                (self.not_a_b, self.marginal_not_a(), self.marginal_b()),
                (self.a_not_b, self.marginal_a(), self.marginal_not_b()),
                (self.not_a_not_b, self.marginal_not_a(), self.marginal_not_b())
                ]

    def n(self):
        return self.a_b + self.a_not_b + self.not_a_b + self.not_a_not_b

    def marginal_a(self):
        return self.a_b + self.a_not_b

    def marginal_not_a(self):
        return self.not_a_b + self.not_a_not_b

    def marginal_b(self):
        return self.a_b + self.not_a_b

    def marginal_not_b(self):
        return self.a_not_b + self.not_a_not_b


def make_ngrams(tokens, min_n=1, max_n=5):
    """Make n-grams from a list of tokens."""
    n_grams = []
    for n in range(min_n, max_n + 1):
        n_grams += list(nltk.ngrams(tokens, n))
    return n_grams


def count_ngrams_in_doc(d, pos_tag_filter=None, min_n=1, max_n=5):
    """Return a regular Counter object of n-grams in document d, filtered using
    a pos tag filter."""

    all_ngrams = []
    # loop over sentences and make n-grams within them
    for sentence in d.get_annotations('Sentence'):
        # get tokens with pos tags from the sentence span and normalize them
        lemmaed_tokens = [(t.lemma(), t.mapped_pos())
                          for t in d.get_annotations_at(sentence.span, 'Token')]

        # make n-grams
        ngrams = make_ngrams(lemmaed_tokens, min_n, max_n)
        for ng_with_pos in ngrams:
            if pos_tag_filter and len(ng_with_pos) > 1:
                pos_tag_sequence = ''.join(w[1] for w in ng_with_pos)
                # test if sequence is allowed
                if not re.fullmatch(pos_tag_filter, pos_tag_sequence):
                    continue  # just discard it
                else:
                    # create actual n-gram
                    ngram = tuple(w[0] for w in ng_with_pos)
                    all_ngrams.append(ngram)

    # count the n-grams and return
    c = Counter(all_ngrams)
    return c
