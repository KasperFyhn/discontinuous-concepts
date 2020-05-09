from pipeline import components as cm
from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import components as cm
from pipeline.evaluation import CorpusReport, TypesReport,\
    gold_standard_concepts
from tqdm import tqdm


class Configuration:

    def __init__(self, freq_threshold=2, c_value_threshold=2, max_n=5,
                 skipgrams=False, bridge_strength_threshold=0, freq_factor=1,
                 extraction_filter=cm.ExtractionFilters.SIMPLE,
                 consider_dcs_in_ranking=False,
                 hypernym_dcs=True, coord_dcs=True):
        self.freq_threshold = freq_threshold
        self.c_value_threshold = c_value_threshold
        self.max_n = max_n
        self.skipgrams = skipgrams
        self.bridge_strength_threshold = bridge_strength_threshold
        self.freq_factor = freq_factor
        self.coord_dcs = coord_dcs
        self.hypernym_dcs = hypernym_dcs
        self.consider_dcs_in_ranking = consider_dcs_in_ranking
        self.extraction_filter = extraction_filter


class Pipeline:

    def __init__(self, configuration: Configuration, *args):
        self.config = configuration

    def run(self, *args):
        pass


class PreProcessingPipeline(Pipeline):

    def run(self, docs):
        print('--- Running pre-processing pipeline ---')

        # first, annotate documents
        with cm.CoreNlpServer() as server:
            docs = server.annotate_batch(docs)

        # second, make n-gram model
        colibri_model_name = 'pipeline_model'
        doc_dict = {doc.id: doc for doc in docs}
        ngramcounting.encode_corpus(colibri_model_name, list(doc_dict.keys()),
                                    lambda x: doc_dict[x])
        ngramcounting.make_colibri_model(colibri_model_name,
                                         mintokens=self.config.freq_threshold,
                                         maxlength=self.config.max_n,
                                         skipgrams=self.config.skipgrams)
        ngram_model = conceptstats.NgramModel.load_model(colibri_model_name)

        return docs, ngram_model


class ConceptExtractionPipeline(Pipeline):

    def run(self, docs, ngram_model: ngramcounting.NgramModel):
        print('--- Running concept extraction pipeline ---')

        # set up extractors
        extractor = cm.CandidateExtractor(self.config.extraction_filter,
                                          max_n=self.config.max_n)
        dc_extractors = []
        if self.config.coord_dcs:
            coord_extractor = cm.CoordCandidateExtractor2(
                self.config.extraction_filter, ngram_model, self.config.max_n,
                self.config.bridge_strength_threshold, self.config.freq_factor
            )
            dc_extractors.append(coord_extractor)

        if self.config.hypernym_dcs:
            hypernym_extractor = cm.HypernymCandidateExtractor(
                self.config.extraction_filter, ngram_model, extractor,
                *dc_extractors, max_n=self.config.max_n,
                pmi_threshold=self.config.bridge_strength_threshold,
                freq_factor=self.config.freq_factor
            )
            dc_extractors.append(hypernym_extractor)

        # then run extraction on all docs
        for doc in tqdm(docs, desc='Extracting candidates'):
            extractor.extract_candidates(doc)
            for dc_extractor in dc_extractors:
                dc_extractor.extract_candidates(doc)

        n_dcs = len(set.union(*[ext.all_candidates for ext in dc_extractors]))
        print(f'Extracted {len(extractor.all_candidates)} continuous '
              f'candidates and {n_dcs} discontinuous candidates.')

        print('Scoring, ranking and filtering concepts')
        if self.config.consider_dcs_in_ranking:
            for dc_extractor in dc_extractors:
                extractor.update(dc_extractor, only_existing=True)

        metrics = cm.Metrics()
        c_value = cm.CValueRanker(
            extractor, self.config.c_value_threshold,
            consider_dcs=self.config.consider_dcs_in_ranking)
        rect_freq = cm.RectifiedFreqRanker(
            extractor, consider_dcs=self.config.consider_dcs_in_ranking)
        tf_idf = cm.TfIdfRanker(extractor)
        glossex = cm.GlossexRanker(extractor, ngram_model)
        pmi_nl = cm.PmiNlRanker(extractor, ngram_model)
        term_coherence = cm.TermCoherenceRanker(extractor, ngram_model)
        voter = cm.VotingRanker(extractor, rect_freq, c_value, tf_idf, glossex,
                                pmi_nl, term_coherence)
        metrics.add(c_value, rect_freq, tf_idf, glossex, pmi_nl, term_coherence,
                    voter)

        if not self.config.consider_dcs_in_ranking:
            for dc_extractor in dc_extractors:
                extractor.update(dc_extractor)

        return extractor, metrics

