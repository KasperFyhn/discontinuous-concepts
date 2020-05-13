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


def make_bridges(potential_bridges, token_sequence, model,
                 bridge_strength_threshold, freq_threshold, freq_factor):
    bridges = defaultdict(set)
    for index1, index2 in potential_bridges:
        token1 = token_sequence[index1]
        token2 = token_sequence[index2]
        bridge = (token1.lemma(), token2.lemma())
        freq = model[bridge]
        if freq == 0:
            continue
        else:
            bridge_strength = conceptstats.ngram_pmi(
                bridge[0], bridge[1], model) \
                              + math.log10(freq) * freq_factor
        if bridge_strength > bridge_strength_threshold\
                and freq >= freq_threshold:
            bridges[index1].add(index2)
    return bridges


def trie_combos(prefix, bridges, pos_seq):
    last = prefix[-1]
    combos = []
    for b in bridges[last]:
        combos += trie_combos(prefix + [b], bridges, pos_seq)
    if last + 1 >= len(pos_seq) or pos_seq[last + 1] in 'c,':
        combos += [prefix]
    else:
        combos += trie_combos(prefix + [last + 1], bridges, pos_seq)
    return combos


class HypernymCandidateExtractor(AbstractCandidateExtractor):

    def __init__(self, pos_tag_filter, model, *extractors, freq_threshold=1,
                 bridge_strength_threshold=1, freq_factor=1):
        super().__init__()
        self.pos_filter = CandidateExtractor.FILTERS[pos_tag_filter]
        self.model = model
        self.extractors = extractors
        self.min_freq = freq_threshold
        self.min_bridge_strength = bridge_strength_threshold
        self.freq_factor = freq_factor

    def _extract_candidates(self, doc):
        basic_candidates = [c for extractor in self.extractors
                            for c in extractor.doc_index[doc]]
        dc_candidates = set()
        for candidate in basic_candidates:
            c_tokens = candidate.get_tokens()
            if len(c_tokens) < 3:
                continue

            potential_bridges = [(i, j) for i in range(len(c_tokens)-2)
                                 for j in range(i+2, len(c_tokens))]
            bridges = make_bridges(potential_bridges, c_tokens, self.model,
                                   self.min_bridge_strength, self.min_freq,
                                   self.freq_factor)
            combinations = trie_combos(
                [0], bridges, ''.join(t.mapped_pos() for t in c_tokens))
            skipgrams = [[c_tokens[i] for i in combo] for combo in combinations]
            for sg in skipgrams:
                chunks = [[sg[0]]]
                for token in sg[1:]:
                    if token.span[0] - chunks[-1][-1].span[-1] > 1:  # gap
                        chunks.append([])
                    chunks[-1].append(token)

                if len(chunks) < 2:  # not discontinuous
                    continue
                else:
                    dc = CandidateDiscConcept(doc, chunks)
                    if re.match(self.pos_filter, dc.pos_sequence()):
                        dc_candidates.add(dc)

        # remove dc's in which a bridge goes from a chunk of words across an
        # identical chunk of words, e.g. "human T lymphocytes and T cell lines"
        allowed_candidates = {dc for dc in dc_candidates
                              if not dc.contains_illegal_bridges()}
        return allowed_candidates


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
                 pmi_threshold=1, freq_factor=1):
        super().__init__()
        self.pos_filter = self.FILTERS[pos_tag_filter]
        self.model = model
        self.max_n = max_n
        self.pmi_threshold = pmi_threshold
        self.freq_factor = freq_factor
        self.freq_threshold = freq_threshold

    def _extract_candidates(self, doc):
        candidates = set()
        for sentence in doc.get_annotations(anno.Sentence):
            tokens = doc.get_annotations_at(sentence.span, anno.Token)
            if len(tokens) > 30:
                continue
            sent_candidates = self._apply_filter(
                self.pos_filter[0], self.pos_filter[1], tokens, doc, self.model,
                self.freq_threshold, self.pmi_threshold, self.freq_factor)
            candidates.update(sent_candidates)

        # remove dc's in which a bridge goes from a chunk of words across an
        # identical chunk of words, e.g. "T lymphocytes and T cell lines"
        allowed_candidates = {dc for dc in candidates
                              if not dc.contains_illegal_bridges()}
        return allowed_candidates

    @staticmethod
    def _apply_filter(super_pattern, concept_pattern, sequence, doc, model,
                      freq_threshold, pmi_threshold, freq_factor):

        candidates = []

        full_pos_seq = ''.join(t.mapped_pos() for t in sequence)
        super_match = [m for m in re.finditer(super_pattern.pattern,
                                              full_pos_seq)]
        super_match_chunks = [sequence[m.start(0):m.end(0)]
                              for m in super_match
                              if m.end(0) - m.start(0) < 15]

        for super_match_chunk in super_match_chunks:
            pos_seq = ''.join(t.mapped_pos() for t in super_match_chunk)

            potential_bridges = set()

            # loop over the sequence to find all coordination pairs
            # this is to make all possible bridges
            for cc_group in re.finditer('(.)((?:,[^,]+)*),?c(.)', pos_seq):
                # we need the indices of the coordinated/enumerated tokens
                first_index = cc_group.start(1)
                enumerations = [i for i in range(cc_group.start(2),
                                                 cc_group.end(2))
                                if not pos_seq[i] == ',']
                last_index = cc_group.start(3)

                # get indices of all tokens after the last word
                after_last = []
                for i in range(last_index + 1, len(pos_seq)):
                    if pos_seq[i] == 'c':  # another coordination
                        after_last.append(i+1)
                        break
                    else:
                        after_last.append(i)  # is a candidate
                    
                # and indices of all tokens before the first word
                before_first = []
                for i in range(first_index - 1, -1, -1):
                    if pos_seq[i] == 'c':  # another coordination
                        before_first.append(i-1)
                        break
                    else:
                        before_first.append(i)

                # from the first token to anything after the last
                first_token = super_match_chunk[first_index]
                if not first_token.pos == 'NNS':
                    # if it's a plural word, it's probably not a modifier, but a
                    # head which is coordinated with the whole thing on the
                    # right, e.g. JJ NNS CC NN NN
                    potential_bridges.update(set(itertools.product(
                        [first_index], after_last)))
                # from the last token to anything before the first
                potential_bridges.update(set(itertools.product(before_first,
                                                               [last_index])))

                # from before first to enumerated elements
                potential_bridges.update(set(itertools.product(before_first,
                                                               enumerations)))
                # from enumerated elements to after last
                potential_bridges.update(set(itertools.product(enumerations,
                                                               after_last)))

            bridges = make_bridges(potential_bridges, super_match_chunk, model,
                                   pmi_threshold, freq_threshold, freq_factor)

            subgrams = [sg for n in range(4, len(super_match_chunk) + 1)
                        for sg in nltk.ngrams(super_match_chunk, n)
                        if len(sg) > 3
                        and re.fullmatch(super_pattern, ''.join(t.mapped_pos()
                                                                for t in sg))]
            for subgram in subgrams:
                start = super_match_chunk.index(subgram[0])
                end = super_match_chunk.index(subgram[-1])

                combinations = trie_combos([start], bridges, pos_seq[:end+1])
                skip_sequences = [[super_match_chunk[i] for i in combo]
                                  for combo in combinations]

                for sequence in skip_sequences:
                    # first, merge adjacent spans
                    merged_chunks = [[sequence.pop(0)]]
                    for token in sequence:
                        if token.span[0] - merged_chunks[-1][-1].span[1] < 2:
                            merged_chunks[-1].append(token)
                        else:
                            merged_chunks.append([token])

                    # then see if they are merged into one; if so, move on
                    if len(merged_chunks) < 2:  # not discontinuous
                        continue
                    else:
                        dc = CandidateDiscConcept(doc, merged_chunks)
                        # post filtering step: allowed POS-sequence?
                        if re.fullmatch(concept_pattern, dc.pos_sequence()):
                            candidates.append(dc)

        return candidates


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
                 c_value_threshold, term_counter=None, consider_dcs=False):
        print('Calculating C-values')
        self._c_threshold = c_value_threshold
        if not term_counter:
            term_counter = candidate_extractor.term_frequencies()
        self._term_counter = term_counter
        self.consider_dcs = consider_dcs
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = conceptstats.calculate_c_values(
            list(self.candidate_extractor.candidate_types()), self._c_threshold,
            self.candidate_extractor.term_frequencies(),
            skipgrams=self.consider_dcs
        )


class RectifiedFreqRanker(AbstractCandidateRanker):

    def __init__(self, candidate_extractor: AbstractCandidateExtractor,
                 term_counter=None, consider_dcs=False):
        print('Calculating Rectified Frequencies')
        if not term_counter:
            term_counter = candidate_extractor.term_frequencies()
        self._term_counter = term_counter
        self.consider_dcs = consider_dcs
        super().__init__(candidate_extractor)

    def _calculate_values(self):
        self._values = conceptstats.calculate_rectified_freqs(
            self.candidate_extractor.candidate_types(), self._term_counter,
            skipgrams=self.consider_dcs
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
        self._values = {tc: sum(1 / r.rank(tc) * self._weights[r]
                                for r in self._rankers)
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
        self.rankers = {type(ranker).__name__: ranker for ranker in rankers}

    def add(self, *rankers):
        for ranker in rankers:
            self.rankers[type(ranker).__name__] = ranker

    def __getitem__(self, item):
        metrics = {}
        for ranker_name, ranker in self.rankers.items():
            if isinstance(ranker, Matcher):
                metrics[ranker_name] = ranker.value(item)
            elif item in ranker:
                metrics[ranker_name] = ranker.value(item)
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

