import re
from collections import defaultdict, Counter
import itertools

import math
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
        except requests.exceptions.HTTPError:
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
# CANDIDATE EXTRACTION
################################################################################

class CandidateConcept(anno.Concept):

    def __init__(self, document, tokens):
        super().__init__(document, (tokens[0].span[0], tokens[-1].span[-1]))
        self.covered_tokens = tokens

    def to_concept(self):
        return anno.Concept(self.document, self.span)

    def accept(self):
        self.document.add_annotation(self.to_concept())


class CandidateDiscConcept(anno.DiscontinuousConcept):

    def __init__(self, document, token_chunks):
        spans = [(tokens[0].span[0], tokens[-1].span[-1])
                 for tokens in token_chunks]
        super().__init__(document, spans)
        self.covered_tokens = [t for tokens in token_chunks for t in tokens]

    def to_concept(self):
        return anno.DiscontinuousConcept(self.document, self.spans)

    def accept(self):
        self.document.add_annotation(self.to_concept())


class ExtractionFilters:
    UNSILO = 'unsilo'
    SIMPLE = 'simple'
    LIBERAL = 'liberal'


class AbstractCandidateExtractor:

    def __init__(self):
        self.doc_index = defaultdict(set)
        self.concept_index = defaultdict(set)
        self.types = set()
        self.all_candidates = set()
        self.extracted_concepts = set()

    def extract_candidates(self, doc):
        candidates = self._extract_candidates(doc)
        for c in candidates:
            self.add(c)

    def candidate_types(self):
        return self.types

    def add(self, concept):
        self.doc_index[concept.document].add(concept)
        self.concept_index[concept.normalized_concept()].add(concept)
        self.types.add(concept.normalized_concept())
        self.all_candidates.add(concept)
        self.extracted_concepts.add(concept.to_concept())

    def _extract_candidates(self, doc):
        pass

    def accept_candidates(self, accepted_candidates):
        for concept in accepted_candidates:
            for instance in self.concept_index[concept]:
                instance.accept()

    def update(self, other, only_existing=False):
        for c in other.all_candidates:
            if only_existing:
                if c.normalized_concept() in self.types:
                    self.add(c)
            else:
                self.add(c)

    def term_frequencies(self):
        return Counter({key: len(sample)
                        for key, sample in self.concept_index.items()})

    def doc_frequencies(self):
        counter = Counter()
        for doc, sample in self.doc_index.items():
            present_concepts = {c.normalized_concept() for c in sample}
            for c in present_concepts:
                counter[c] += 1
        return counter


class CandidateExtractor(AbstractCandidateExtractor):

    FILTERS = {
        ExtractionFilters.UNSILO: re.compile(r'([na]|(ns)|(vn))+n'),
        ExtractionFilters.SIMPLE: re.compile(r'[an]+n'),
        ExtractionFilters.LIBERAL: re.compile(r'[navrds]*n')
    }

    def __init__(self, pos_tag_filter=None, min_n=1, max_n=5):
        super().__init__()
        self.pos_filter = self.FILTERS[pos_tag_filter]
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


class HypernymCandidateExtractor(CandidateExtractor):

    def __init__(self, pos_tag_filter, model, *extractors, min_n=3, max_n=5,
                 max_k=None, freq_threshold=1, pmi_threshold=1):
        super().__init__(pos_tag_filter, min_n, max_n)
        self.model = model
        self.extractors = extractors
        if not max_k:
            self.max_k = max_n - 2
        else:
            self.max_k = min(max_k, max_n - 2)
        self.freq_threshold = freq_threshold
        self.pmi_threshold = pmi_threshold

    def _extract_candidates(self, doc):
        basic_candidates = [c for extractor in self.extractors
                            for c in extractor.doc_index[doc]]
        dc_candidates = []
        for candidate in basic_candidates:
            c_tokens = candidate.get_tokens()
            if len(c_tokens) < 3:
                continue

            obligatory_token = c_tokens[-1]
            max_n = min(self.max_n, len(c_tokens))

            skipgrams = [sg for n in range(2, max_n)
                         for sg in nltk.skipgrams(c_tokens, n, self.max_k)
                         if sg[-1] == obligatory_token]
            for sg in skipgrams:
                chunks = [[sg[0]]]
                for token in sg[1:]:
                    if token.span[0] - chunks[-1][-1].span[-1] > 1:  # gap
                        chunks.append([])
                    chunks[-1].append(token)
                # check if all gaps have a bridge with high enough association
                bridges_accepted = True
                for i in range(len(chunks) - 1):
                    chunk = chunks[i]
                    next_chunk = chunks[i+1]
                    left_of_gap, right_of_gap = chunk[-1], next_chunk[0]
                    bridge_pmi = conceptstats.ngram_pmi(left_of_gap.lemma(),
                                                        right_of_gap.lemma(),
                                                        self.model)
                    if bridge_pmi < self.pmi_threshold:
                        bridges_accepted = False  # not high enough association

                if len(chunks) < 2 or not bridges_accepted:
                    continue
                else:
                    dc = CandidateDiscConcept(doc, chunks)
                    if re.match(self.pos_filter, dc.pos_sequence()) \
                            and self.model[dc.normalized_concept()] \
                            > self.freq_threshold:
                        dc_candidates.append(dc)

        return dc_candidates


class CoordCandidateExtractor(AbstractCandidateExtractor):
    FILTERS = {
        ExtractionFilters.UNSILO:
            (re.compile(r'(([navs]+,?)+c[navs])+[navs]*'),
             re.compile(r'[navs]+n')),
        ExtractionFilters.SIMPLE: (re.compile(r'(([an]+,?)+c[an])+[an]*'),
                                   re.compile(r'[an]+n')),
        ExtractionFilters.LIBERAL:
            (re.compile(r'(([navrds]+,?)+c[navrds])+[navrds]*'),
             re.compile(r'[navrds]+n'))
    }

    def __init__(self, pos_tag_filter, model, max_n=5, freq_threshold=1,
                 pmi_threshold=1):
        super().__init__()
        self.pos_filter = self.FILTERS[pos_tag_filter]
        self.model = model
        self.max_n = max_n
        self.pmi_threshold = pmi_threshold
        self.freq_threshold = freq_threshold

    def _extract_candidates(self, doc):
        candidates = []
        for sentence in doc.get_annotations(anno.Sentence):
            tokens = doc.get_annotations_at(sentence.span, anno.Token)
            sent_candidates = self._apply_filter(
                self.pos_filter[0], self.pos_filter[1], tokens, doc, self.model,
                self.freq_threshold, self.pmi_threshold)
            candidates += sent_candidates

        return candidates

    @staticmethod
    def _apply_filter(super_pattern, concept_pattern, tokens, doc, model,
                      freq_threshold, pmi_threshold):

        candidates = []

        full_pos_seq = ''.join(t.mapped_pos() for t in tokens)
        super_match = [m for m in re.finditer(super_pattern.pattern,
                                              full_pos_seq)]
        super_match_chunks = [tokens[m.start(0):m.end(0)] for m in super_match]
        supergrams = [sg for smc in super_match_chunks
                      for n in range(4, len(smc) + 1)
                      for sg in nltk.ngrams(smc, n)
                      if len(sg) > 3
                      and re.match(super_pattern, ''.join(t.mapped_pos()
                                                          for t in sg))]

        for supergram in supergrams:
            pos_seq = ''.join(t.mapped_pos() for t in supergram)

            edges = []
            # loop over the sequence to find all coordination pairs
            # this is to determine which gaps to bridge
            for cc_group in re.finditer('(.)(?:,[^,]+)*,?c(.)', pos_seq):
                # we need the indices of the tokens around the coordination
                # but forget extra enumerated words for now
                # e.g. we want B and D in A B, C and D E
                first_index = cc_group.start(1)
                last_index = cc_group.start(2)

                # determine gap from token 1 to somewhere after the last, if any
                # e.g. given A B C c D E F, both E and F will be candidates to
                # make a bridge to from B
                first_token = supergram[first_index]
                if first_token.pos == 'NNS':
                    # plural word, probably not a modifier, but a head which is
                    # coordinated with the whole thing on the right, e.g.
                    # JJ NNS CC NN NN
                    right_edge = None
                else:
                    # get indices of all tokens after the last word
                    # but only up until another coordination
                    after_last = []
                    hit_cc = False
                    for i in range(last_index+1, len(pos_seq)):
                        if pos_seq[i] == 'c':
                            # another coordination was encountered
                            hit_cc = True
                            break
                        else:
                            after_last.append(i)  # is a candidate

                    # each candidate is a potential edge, but also None
                    # decide based on high PMI between candidates and first.
                    # if not any tokens after the last in the coordination,
                    # there is no edge to take care of, e.g. A B c C, and
                    # None will win.
                    if hit_cc:
                        potential_edges = {}
                    else:
                        potential_edges = {None: pmi_threshold}
                    for index in after_last:
                        token = supergram[index]
                        bridge = (first_token.lemma(), token.lemma())
                        freq = model[bridge]
                        if freq == 0:
                            pmi = -math.inf
                        else:
                            pmi = conceptstats.ngram_pmi(bridge[0], bridge[1],
                                                         model)
                        potential_edges[index] = pmi
                        if pmi > pmi_threshold and freq >= freq_threshold:
                            potential_edges = {index: pmi}
                            break

                    right_edge = max(potential_edges,
                                     key=lambda x: potential_edges[x])

                # determine gap from last token to somewhere before first
                # more or less the same as previous, just reversed. Also, pay
                # attention to indices
                last_token = supergram[last_index]
                # get indices of all tokens before the first word
                # but only down until another coordination
                before_first = []
                hit_cc = False
                for i in range(first_index - 1, -1, -1):
                    if pos_seq[i] == 'c':
                        # another coordination was encountered
                        hit_cc = True
                        break
                    else:
                        before_first.append(i)

                if hit_cc:
                    potential_edges = {}
                else:
                    potential_edges = {None: pmi_threshold}
                for index in before_first:
                    token = supergram[index]
                    bridge = (token.lemma(), last_token.lemma())
                    freq = model[bridge]
                    if freq == 0:
                        pmi = -math.inf
                    else:
                        pmi = conceptstats.ngram_pmi(bridge[0], bridge[1],
                                                     model)
                    potential_edges[index] = pmi
                    if pmi > pmi_threshold and freq >= freq_threshold:
                        potential_edges = {index: pmi}
                        break

                left_edge = max(potential_edges,
                                key=lambda x: potential_edges[x])
                # + 1 to ensure that the index is at the token AFTER the edge
                if type(left_edge) == int:
                    left_edge += 1

                edges.append((left_edge, right_edge))

            # make common and shared element chunks based on the edges
            # candidates are then made from the Cartesian product of elements
            prev_right = 0
            elements = []
            for left_edge, right_edge in edges:
                if left_edge:
                    shared = supergram[prev_right:left_edge]
                    if shared:  # shared material before the coordination
                        elements.append([shared])
                alternatives = [[]]
                # loop over tokens in the coordination and chop up the chunks
                # based on commas and coordination (more can be added)
                for token in supergram[left_edge:right_edge]:
                    if token.mapped_pos() in ',c':
                        if alternatives[-1]:  # wrap up the Tokens and make new
                            alternatives.append([])
                    else:  # add it to the current
                        alternatives[-1].append(token)
                if not alternatives[-1]:
                    # sometimes, due to punctuation, this last one can be empty
                    # if so, remove it
                    alternatives.pop()
                elements.append(alternatives)
                prev_right = right_edge

            if prev_right:  # shared material after the last coordination
                shared = supergram[prev_right:]
                if shared:  # shared material before the coordination
                    elements.append([shared])

            for chunks in itertools.product(*elements):
                # first, merge adjacent spans
                chunks = [tuple(chunk) for chunk in chunks]
                merged_chunks = [chunks.pop(0)]
                for chunk in chunks:
                    if chunk[0].span[0] - merged_chunks[-1][-1].span[1] < 2:
                        merged_chunks[-1] += chunk
                    else:
                        merged_chunks.append(chunk)

                # then see if they are merged into one; if so, move on
                if len(merged_chunks) == 1:  # not discontinuous
                    continue
                else:
                    dc = CandidateDiscConcept(doc, merged_chunks)
                    # post filtering step: allowed POS-sequence?
                    if re.match(concept_pattern, dc.pos_sequence()):
                        candidates.append(dc)

        return candidates


class CoordCandidateExtractor2(CoordCandidateExtractor):

    @staticmethod
    def _apply_filter(super_pattern, concept_pattern, tokens, doc, model,
                      freq_threshold, pmi_threshold):

        candidates = []

        full_pos_seq = ''.join(t.mapped_pos() for t in tokens)
        super_match = [m for m in re.finditer(super_pattern.pattern,
                                              full_pos_seq)]
        super_match_chunks = [tokens[m.start(0):m.end(0)] for m in super_match]
        supergrams = [sg for smc in super_match_chunks
                      for n in range(4, len(smc) + 1)
                      for sg in nltk.ngrams(smc, n)
                      if len(sg) > 3
                      and re.match(super_pattern, ''.join(t.mapped_pos()
                                                          for t in sg))]

        for supergram in super_match_chunks:
            pos_seq = ''.join(t.mapped_pos() for t in supergram)

            bridges = defaultdict(set)

            # loop over the sequence to find all coordination pairs
            # this is to make all possible bridges
            for cc_group in re.finditer('(.)((?:,[^,]+)*),?c(.)', pos_seq):
                # we need the indices of the tokens around the coordination
                # but forget extra enumerated words for now
                # e.g. we want B and D in A B, C and D E
                first_index = cc_group.start(1)
                enumerations = [i for i in range(cc_group.start(2),
                                                 cc_group.end(2))
                                if not pos_seq[i] == 'c']
                last_index = cc_group.start(3)

                # make potential bridges from first token to after the last
                first_token = supergram[first_index]
                if first_token.pos == 'NNS':
                    # plural word, probably not a modifier, but a head which is
                    # coordinated with the whole thing on the right, e.g.
                    # JJ NNS CC NN NN
                    continue
                else:
                    # get indices of all tokens after the last word
                    # except coordinations
                    after_last = [i for i in range(last_index + 1, len(pos_seq))
                                  if not pos_seq[i] == 'c']
                    
                # and indices of all tokens before the first word (except cc's)
                before_first = [i for i in range(first_index - 1, -1, -1)
                                if not pos_seq[i] == 'c']

                potential_bridges = []
                # from the first token to anything after the last
                potential_bridges += list(itertools.product([first_index],
                                                            after_last))
                # from the last token to anything before the first
                potential_bridges += list(itertools.product(before_first,
                                                            [last_index]))
                # from anything before first to after last
                potential_bridges += list(itertools.product(before_first,
                                                            after_last))
                # from before first to enumerated elements
                potential_bridges += list(itertools.product(before_first,
                                                            enumerations))
                # from enumerated elements to after last
                potential_bridges += list(itertools.product(before_first,
                                                            enumerations))

                for index1, index2 in potential_bridges:
                    token1 = supergram[index1]
                    token2 = supergram[index2]
                    bridge = (token1.lemma(), token2.lemma())
                    freq = model[bridge]
                    if freq == 0:
                        pmi = -math.inf
                    else:
                        pmi = conceptstats.ngram_pmi(bridge[0], bridge[1],
                                                     model)
                    if pmi > pmi_threshold and freq >= freq_threshold:
                        bridges[index1].add(index2)







################################################################################
# CANDIDATE SCORING AND RANKING
################################################################################


class AbstractCandidateRanker:

    def __init__(self, candidate_extractor: AbstractCandidateExtractor):
        self.candidate_extractor = candidate_extractor
        self._values = None
        self._calculate_values()
        self._ranked = []
        self._rank_candidates()
        self._ranks = {term: i + 1 for i, term in enumerate(self._ranked)}

    def __contains__(self, item):
        return item in self._values.keys()

    def value(self, term):
        if isinstance(term, str):
            term = tuple(term.split())
        return self._values[term]

    def rank(self, term):
        if isinstance(term, str):
            term = tuple(term.split())
        return self._ranks[term]

    def __getitem__(self, item):
        return self._ranked[item - 1]

    def _calculate_values(self):
        pass

    def _rank_candidates(self):
        self._ranked = sorted(self._values.keys(), reverse=True,
                              key=lambda x: self._values[x])

    def filter_at_value(self, value):
        return_list = []
        for c in self._ranked:
            if self.value(c) > value:
                return_list.append(c)
            else:
                break
        return return_list

    def keep_proportion(self, proportion: float):
        cutoff = int(len(self._ranked) * proportion)
        return [c for c in self._ranked[:cutoff]]

    def keep_n_highest(self, n: int):
        return self._ranked[:n]


class CValueRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor,
                 c_value_threshold, term_counter=None):
        print('Calculating C-values')
        self._c_threshold = c_value_threshold
        if not term_counter:
            term_counter = candidate_extractor.term_frequencies()
        self._term_counter = term_counter
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = conceptstats.calculate_c_values(
            list(self.candidate_extractor.candidate_types()), self._c_threshold,
            self.candidate_extractor.term_frequencies()
        )


class RectifiedFreqRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor,
                 term_counter=None):
        print('Calculating Rectified Frequencies')
        if not term_counter:
            term_counter = candidate_extractor.term_frequencies()
        self._term_counter = term_counter
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = conceptstats.calculate_rectified_freqs(
            self.candidate_extractor.candidate_types(), self._term_counter
        )


class TfIdfRanker(AbstractCandidateRanker):
    def __init__(self, candidate_extractor: AbstractCandidateExtractor,
                 term_counter=None, doc_counter=None, n_docs=None):
        print('Calculating TF-IDF values')
        if not term_counter:
            term_counter = candidate_extractor.term_frequencies()
        self._term_counter = term_counter
        if not doc_counter:
            doc_counter = candidate_extractor.doc_frequencies()
        self._doc_counter = doc_counter
        if not n_docs:
            n_docs = len(candidate_extractor.doc_index)
        self._n_docs = n_docs
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = conceptstats.calculate_tf_idf_values(
            self.candidate_extractor.candidate_types(), self._term_counter,
            self._doc_counter, self._n_docs
        )


class GlossexRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor, model):
        print('Calculating Glossex values')
        self._ngram_model = model
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = {tc: conceptstats.glossex(tc, self._ngram_model)
                        for tc in self.candidate_extractor.candidate_types()}


class PmiNlRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor, model):
        print('Calculating length normalized PMI values')
        self._ngram_model = model
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = {tc: conceptstats.length_normalized_pmi(
            tc, self._ngram_model)
            for tc in self.candidate_extractor.candidate_types()}


class TermCoherenceRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor, model):
        print('Calculating Term Coherence values')
        self._ngram_model = model
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = {tc: conceptstats.term_coherence(tc, self._ngram_model)
                        for tc in self.candidate_extractor.candidate_types()}


class VotingRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor,
                 *rankers, weights=None):
        print('Calculating votes between rankers')
        self.candidate_extractor = candidate_extractor
        self._rankers = rankers
        if not weights:
            weights = [1] * len(rankers)
        self._weights = {rankers[i]: weights[i] for i in range(len(rankers))}
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = {tc: sum(1 / r.rank(tc) for r in self._rankers)
                        for tc in self.candidate_extractor.candidate_types()}


################################################################################
# ONTOLOGY MATCHING/VERIFICATION
################################################################################

class Matcher:

    def __init__(self, candidate_extractor, verifiers):
        self.candidates = candidate_extractor
        self._verifiers = verifiers
        self._verified = set()
        self.verify_candidates()

    def __contains__(self, item):
        return item in self._verified

    def verify_candidates(self):
        for ct in self.candidates.candidate_types():
            if ct in self._verifiers:
                self._verified.add(ct)

    def verified(self):
        return self._verified

    def value(self, term):
        if isinstance(term, str):
            term = tuple(term.split())
        return term in self._verified


class MeshMatcher(Matcher):

    def __init__(self, candidate_extractor):
        super().__init__(candidate_extractor, dataio.load_mesh_terms())


class GoldMatcher(Matcher):

    def __init__(self, candidate_extractor, gold_concepts):
        super().__init__(candidate_extractor, gold_concepts)


class ExtractionMatcher(Matcher):

    def __init__(self, candidate_extractor):
        super().__init__(candidate_extractor,
                         set(candidate_extractor.concept_index.keys()))


################################################################################
# METRICS AND FILTERING
################################################################################


class Metrics:

    C_VALUE = CValueRanker.__name__
    TF_IDF = TfIdfRanker.__name__
    RECT_FREQ = RectifiedFreqRanker.__name__
    GLOSSEX = GlossexRanker.__name__
    PMI_NL = PmiNlRanker.__name__
    TERM_COHERENCE = TermCoherenceRanker.__name__
    VOTER = VotingRanker.__name__
    MESH_MATCHER = MeshMatcher.__name__
    GOLD = GoldMatcher.__name__

    def __init__(self, *rankers):
        self.rankers = list(rankers)

    def add(self, *rankers):
        for ranker in rankers:
            self.rankers.append(ranker)

    def __getitem__(self, item):
        metrics = {}
        for ranker in self.rankers:
            if isinstance(ranker, Matcher):
                metrics[type(ranker).__name__] = ranker.value(item)
            elif item in ranker:
                metrics[type(ranker).__name__] = ranker.value(item)
        return metrics

    def inspect(self, concepts, extractors=None, doc_dict=None, char_window=20):
        for concept in concepts:
            print(concept.get_context(char_window))
            print(self[concept.normalized_concept()])
            if extractors:
                for ext in extractors:
                    print(f'Extracted by {type(ext).__name__}:',
                          concept in ext.extracted_concepts)
            if doc_dict:
                if concept.document.id in doc_dict:
                    doc = doc_dict[concept.document.id]
                    print(doc.get_annotations_at(concept.span, anno.Token))
            print()


class ConceptFilter:

    class METHODS:
        ALL = all
        ANY = any
        MORE_THAN = lambda x: sum(1 for b in x if b)\
                              > sum(1 for b in x if not b)
        AT_LEAST = lambda n: lambda x: sum(1 for b in x if b) >= n

    def __init__(self, *filters, filtering_method=METHODS.ALL):
        self.filters = filters
        self.filtering_method = filtering_method

    def apply(self, candidates):
        passed_candidates = []
        for c in candidates:
            passed_filters = [f(c) for f in self.filters]
            if self.filtering_method(passed_filters):
                passed_candidates.append(c)
        return passed_candidates
