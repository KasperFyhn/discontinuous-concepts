import re
from collections import defaultdict

import nltk
import requests
import os
from pathlib import Path
from datautils import annotations as anno, dataio
from stats.ngramcounting import make_ngrams, NgramModel
from stats import conceptstats
import subprocess
from tqdm import tqdm
import multiprocessing as mp


################################################################################
# BASIC NLP ANNOTATION
################################################################################

class CoreNlpServer:

    CORE_NLP_PATH = str(Path.home()) + '/stanford-corenlp-full-2018-02-27'

    def __init__(self, segmentation=True, pos_tagging=True,
                 constituency_parsing=False, dependency_parsing=False,
                 show_output=False):

        annotation_types = []
        if segmentation: annotation_types.append('tokenize,ssplit')
        if pos_tagging: annotation_types.append('pos')
        if constituency_parsing: annotation_types.append('parse')
        if dependency_parsing: annotation_types.append('depparse')
        self.anno_types = ','.join(annotation_types)
        self._server = self._start_server(show_output)

    def _start_server(self, show_output=False):
        command = ['java', '-mx8g', '-cp', "*",
                   'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
                   '-preload ' + self.anno_types + ' -timeout 15000']
        os.chdir(self.CORE_NLP_PATH)
        print('Opening CoreNLP annotator server. It may still be running '
              'after termination if shut-down is not stated explicitly.')
        server_process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL
                                          if not show_output else None)
        while True:  # try to establish connection
            try:
                nltk.CoreNLPParser(tagtype='pos').api_call('test')
                break  # connection works
            except requests.exceptions.ConnectionError:
                continue
        return server_process

    def stop_server(self):
        self._server.terminate()
        print('CoreNLP server shut down successfully.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()

    def annotate_doc(self, doc):
        return CoreNlpServer._annotate_doc(doc, self.anno_types)

    def annotate_batch(self, docs, n_docs=None):
        if not n_docs:
            n_docs = len(docs)
        args_list = [(doc, self.anno_types) for doc in docs]
        annotated_docs = []
        with mp.Pool() as pool:
            for doc in tqdm(pool.imap_unordered(
                    CoreNlpServer._annotate_parallel, args_list),
                    total=n_docs, desc='Annotating batch'):
                annotated_docs.append(doc)

        return annotated_docs

    @staticmethod
    def _annotate_doc(doc, annotation_types):
        if ',parse' in annotation_types:
            annotation_types = annotation_types.replace(',parse', '')
            parse = True
        else:
            parse = False

        try:
            parser = nltk.CoreNLPParser(tagtype='pos')
            doc_text = doc.get_text().replace('%', '%25')
            # the reason for replacing percentage signs is that it screws up
            # spans in the text
            # TODO: replacing percentage signs is sort of a hacky fix;
            #  make it more robust!
            annotations = parser.api_call(
                doc_text, properties={
                    'annotators': annotation_types,
                    'ssplit.newlineIsSentenceBreak': 'two'}, timeout=15000
            )
        except requests.exceptions.HTTPError as e:
            # some docs cause such an error; if so, give up and return
            print('\rHTTPError for:', doc.id)
            return doc

        # loop over sentences and resolve the spans
        # TODO: find out  how to do it for const and dep parsing
        sentences = annotations['sentences']
        for sentence in sentences:
            # make sentence annotation
            tokens = sentence['tokens']
            sentence_begin = tokens[0]['characterOffsetBegin']
            sentence_end = tokens[-1]['characterOffsetEnd']
            sent_anno = anno.Sentence(doc, (sentence_begin, sentence_end))
            doc.add_annotation(sent_anno)

            if parse:
                try:
                    list(parser.parse([t['word'] for t in tokens]))
                except Exception as e:
                    print(type(e), 'Parsing went wrong:',
                          [t['word'] for t in tokens])
            # loop over tokens to make token annotations
            for token in tokens:
                token_begin = token['characterOffsetBegin']
                token_end = token['characterOffsetEnd']
                pos_tag = token['pos']
                token_anno = anno.Token(doc, (token_begin, token_end), pos_tag)
                doc.add_annotation(token_anno)

        return doc

    @staticmethod
    def _annotate_parallel(args_tuple):
        doc, anno_types = args_tuple
        return CoreNlpServer._annotate_doc(doc, anno_types)


################################################################################
# CONCEPT EXTRACTION
################################################################################

class CandidateConcept(anno.Concept):

    def __init__(self, document, tokens):
        super().__init__(document, (tokens[0].span[0], tokens[-1].span[-1]))
        self.covered_tokens = tokens

    def accept(self):
        concept = anno.Concept(self.document, self.span)
        self.document.add_annotation(concept)


class CandidateDiscConcept(anno.DiscontinuousConcept):

    def __init__(self, document, token_chunks):
        spans = [(tokens[0].span[0], tokens[-1].span[-1])
                      for tokens in token_chunks]
        super().__init__(document, spans)
        self.covered_tokens = [t for tokens in token_chunks for t in tokens]

    def accept(self):
        concept = anno.DiscontinuousConcept(self.document, self.spans)
        self.document.add_annotation(concept)


class AbstractCandidateExtractor:

    def __init__(self):
        self.doc_index = defaultdict(list)
        self.concept_index = defaultdict(list)
        self.all_candidates = []

    def extract_candidates(self, doc):
        candidates = self._extract_candidates(doc)
        for c in candidates:
            self.add(c)

    def add(self, concept):
        self.doc_index[concept.document].append(concept)
        self.concept_index[concept.normalized_concept()].append(concept)
        self.all_candidates.append(concept)

    def _extract_candidates(self, doc):
        pass

    def accept_candidates(self, accepted_candidates):
        for concept in accepted_candidates:
            for instance in self.concept_index[concept]:
                instance.accept()

    def update(self, other):
        for c in other.all_candidates:
            self.add(c)


class CandidateExtractor(AbstractCandidateExtractor):

    class FILTERS:
        unsilo = re.compile(r'([na]|(ns)|(vn))+n')
        simple = re.compile(r'[an]+n')
        liberal = re.compile(r'[navrdsp]*n')

    def __init__(self, pos_tag_filter=None, min_n=1, max_n=5):
        super().__init__()
        self.pos_filter = pos_tag_filter
        self.min_n = min_n
        self.max_n = max_n

    def _extract_candidates(self, doc):
        pos_filter = self.pos_filter
        candidates = []
        for sentence in doc.get_annotations(anno.Sentence):
            tokens = doc.get_annotations_at(sentence.span, anno.Token)
            ngrams = make_ngrams(tokens, self.min_n,  self.max_n)
            for ngram_tokens in ngrams:
                pos_sequence = ''.join(t.mapped_pos() for t in ngram_tokens)
                if re.fullmatch(pos_filter, pos_sequence):
                    candidates.append(CandidateConcept(doc, ngram_tokens))
        return candidates

    def candidate_types(self):
        return set(self.concept_index.keys())


class HypernymCandidateExtractor(CandidateExtractor):

    def __init__(self, pos_tag_filter=None, min_n=3, max_n=5, max_k=None):
        super().__init__(pos_tag_filter, min_n, max_n)
        if not max_k:
            self.max_k = max_n - 2
        else:
            self.max_k = min(max_k, max_n - 2)

    def _extract_candidates(self, doc):
        basic_candidates = super()._extract_candidates(doc)
        dc_candidates = []
        for candidate in basic_candidates:
            c_tokens = candidate.get_tokens()
            if len(c_tokens) < 3:
                continue

            obligatory_token = c_tokens[-1]
            min_n = min(self.min_n, 2)

            skipgrams = [sg for n in range(min_n, self.max_n)
                         for sg in nltk.skipgrams(c_tokens, n, self.max_k)
                         if sg[-1] == obligatory_token]
            for sg in skipgrams:
                chunks = [[sg[0]]]
                for token in sg[1:]:
                    if token.span[0] - chunks[-1][-1].span[-1] > 1:  # gap
                        chunks.append([])
                    chunks[-1].append(token)

                if len(chunks) < 2:
                    continue
                else:
                    dc = CandidateDiscConcept(doc, chunks)
                    dc_candidates.append(dc)

        return dc_candidates


class CoordCandidateExtractor(AbstractCandidateExtractor):

    class FILTERS:
        coord_unsilo = (
            re.compile(r'(?:[navs]+,?)+c[navs]n+'),
            re.compile(r'(?:([navs]+),?)(?:[navs]+,?)*c[navs]+?(n+)'))
        coord_simple = (re.compile(r'(?:[an]+,?)+cnn+'),
                        re.compile(r'(?:([an]+),?)(?:[an]+,?)*cn(n+)'))
        coord_liberal = (
            re.compile(r'(?:[navrds]+,?)+c[navrds]n+'),
            re.compile(r'(?:([navrds]+),?)(?:[navrds]+,?)*c[navrds](n+)')
        )

    def __init__(self, pos_tag_filters, max_n=5):
        super().__init__()
        self.pos_filters = pos_tag_filters
        self.min_n = 3
        self.max_n = max_n

    def _extract_candidates(self, doc):
        candidates = []
        for sentence in doc.get_annotations(anno.Sentence):
            tokens = doc.get_annotations_at(sentence.span, anno.Token)
            for f in self.pos_filters:
                sent_candidates = self._apply_filter(f, tokens, doc)
                candidates += sent_candidates

        return candidates

    @staticmethod
    def _apply_filter(filter_tuple, tokens, doc):
        super_pattern = filter_tuple[0]
        concept_pattern = filter_tuple[1]

        candidates = []

        full_pos_seq = ''.join(t.mapped_pos() for t in tokens)
        super_match = re.finditer('.*?(' + super_pattern.pattern + ').*',
                                  full_pos_seq)
        super_match_strings = [tokens[m.start(1):m.end(1)] for m in super_match]

        for ngram in (ngram for sm in super_match_strings
                      for ngram in make_ngrams(sm, max_n=len(sm))):
            pos_seq = ''.join(t.mapped_pos() for t in ngram)
            match = re.fullmatch(super_pattern, pos_seq)
            if match:
                concept_match = re.fullmatch(concept_pattern, pos_seq)
                dc_token_chunks = []
                for group_number in range(1, len(concept_match.groups()) + 1):
                    start = concept_match.start(group_number)
                    end = concept_match.end(group_number)
                    dc_token_chunks.append(ngram[start:end])
                dc = CandidateDiscConcept(doc, dc_token_chunks)
                candidates.append(dc)

        return candidates


class AbstractCandidateRanker:

    def __init__(self, candidate_extractor):
        self.candidates = candidate_extractor
        self._ranked = []

    def rank(self):
        pass

    def filter_at_value(self, value):
        return [c for c, v in self._ranked if v > value]

    def keep_proportion(self, proportion: float):
        cutoff = int(len(self._ranked) * proportion)
        return [c for c, v in self._ranked[:cutoff]]

    def keep_n_highest(self, n: int):
        return self._ranked[:n]


class CValueRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor, c_value_threshold, ngram_model,
                 include_skipgrams):
        super().__init__(candidate_extractor)
        self._c_threshold = c_value_threshold
        self._ngram_model = ngram_model
        self._skipgrams = include_skipgrams
        self._c_values = None

    def rank(self):
        c_values = conceptstats.calculate_c_values(
            self.candidates.candidate_types(), self._c_threshold,
            self._ngram_model, self._skipgrams
        )
        self._c_values = c_values
        self._ranked = sorted(c_values.items(), reverse=True,
                              key=lambda x: x[1])

    def c_value(self, term):
        if self._c_values:
            if isinstance(term, str):
                term = tuple(term.split())
            return self._c_values[term]


class VotingCandidateRanker(AbstractCandidateRanker):

    pass


class OntologyMatcher:

    def __init__(self, candidate_extractor):
        self.candidates = candidate_extractor
        self._verified = set()

    def verify_candidates(self, *args):
        pass

    def verified(self):
        return self._verified


class MeshMatcher(OntologyMatcher):

    def __init__(self, candidate_extractor):
        super().__init__(candidate_extractor)
        self._mesh_terms = dataio.load_mesh_terms()

    def verify_candidates(self, *args):
        for ct in tqdm(self.candidates.candidate_types(),
                       desc='Matching candidates against MeSH'):
            if ct in self._mesh_terms:
                self._verified.add(ct)


