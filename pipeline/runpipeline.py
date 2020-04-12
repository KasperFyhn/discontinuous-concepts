from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import components as cm
from pipeline.evaluation import CorpusReport, TypesReport,\
    gold_standard_concepts
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'genia'
RUN_VERSION = '1'

SKIPGRAMS = False
C_VALUE_THRESHOLD = 1.5
FREQ_THRESHOLD = 3
MAX_N = 5
GAP_PMI_THRESHOLD = .1

METRICS = cm.Metrics()
FILTER = cm.ConceptFilter(
    lambda c: METRICS[c][cm.Metrics.C_VALUE] >= C_VALUE_THRESHOLD,
    lambda c: METRICS[c][cm.Metrics.RECT_FREQ] >= FREQ_THRESHOLD,
    lambda c: METRICS[c][cm.Metrics.GLOSSEX] >= 1.5,
    # lambda c: METRICS[c][cm.Metrics.PMI_NL] >= 2,
    # lambda c: METRICS[c][cm.Metrics.TF_IDF] >= 100,
    filtering_method=cm.ConceptFilter.METHODS.ALL
)

print('STEP 0: LOAD GOLD DOCUMENTS AND CONCEPTS')
print('Loading gold docs: ', end='')
gold_docs = dio.load_corpus(CORPUS)
for doc in tqdm(gold_docs, desc='Removing unigram Concepts'):
    unigram_concepts = [c for c in doc.get_annotations(anno.Concept)
                        if len(c) == 1]
    for c in unigram_concepts:
        doc.remove_annotation(c)
gold_concepts = gold_standard_concepts(gold_docs)
gold_counter = conceptstats.count_concepts(gold_docs)


print('\nSTEP 1: ANNOTATE DOCUMENTS')
docs = dio.load_corpus(CORPUS, only_text=True)
with cm.CoreNlpServer() as server:
    docs = server.annotate_batch(docs)


print('\nSTEP 2: MAKE N-GRAM MODEL')
colibri_model_name = CORPUS + 'v' + RUN_VERSION
spec_name = '_std'
doc_dict = {doc.id: doc for doc in docs}
ngramcounting.encode_corpus(colibri_model_name, list(doc_dict.keys()),
                            lambda x: doc_dict[x])
ngramcounting.make_colibri_model(colibri_model_name, spec_name,
                                 mintokens=FREQ_THRESHOLD, maxlength=MAX_N,
                                 skipgrams=SKIPGRAMS)
ngram_model = conceptstats.NgramModel.load_model(colibri_model_name, spec_name)


print('\nSTEP 3: EXTRACT CANDIDATE CONCEPTS')
extractor = cm.CandidateExtractor(
    pos_tag_filter=cm.ExtractionFilters.SIMPLE, max_n=MAX_N
)
coord_extractor = cm.CoordCandidateExtractor(
    cm.ExtractionFilters.SIMPLE, ngram_model,
    max_n=MAX_N, pmi_threshold=GAP_PMI_THRESHOLD
)
hypernym_extractor = cm.HypernymCandidateExtractor(
    cm.ExtractionFilters.SIMPLE, ngram_model, extractor,
    coord_extractor, max_n=MAX_N, pmi_threshold=GAP_PMI_THRESHOLD
)
for doc in tqdm(docs, desc='Extracting candidates'):
    extractor.extract_candidates(doc)
    coord_extractor.extract_candidates(doc)
    hypernym_extractor.extract_candidates(doc)

n_dcs = len(set.union(coord_extractor.all_candidates,
                      hypernym_extractor.all_candidates))
print(f'Extracted {len(extractor.all_candidates)} continuous candidates and '
      f'{n_dcs} discontinuous candidates.')


print('\nSTEP 4: SCORE, RANK AND FILTER CANDIDATE CONCEPTS')
c_value = cm.CValueRanker(extractor, C_VALUE_THRESHOLD)
rect_freq = cm.RectifiedFreqRanker(extractor)
tf_idf = cm.TfIdfRanker(extractor)
glossex = cm.GlossexRanker(extractor, ngram_model)
pmi_nl = cm.PmiNlRanker(extractor, ngram_model)
term_coherence = cm.TermCoherenceRanker(extractor, ngram_model)
voter = cm.VotingRanker(extractor, c_value, tf_idf, glossex, pmi_nl,
                        term_coherence)
METRICS.add(c_value, rect_freq, tf_idf, glossex, pmi_nl, term_coherence, voter)

final = voter.keep_proportion(1)  # keep all, but ranked
final = FILTER.apply(final)  # then filter

extractor.update(coord_extractor)
extractor.update(hypernym_extractor)

mesh_matcher = cm.MeshMatcher(extractor)

extractor.accept_candidates(set(final).union(mesh_matcher.verified()))

print('\nSTEP 5: EVALUATE')
corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
corpus_report.performance_summary()
actual_errors = corpus_report.error_analysis(
    gold_concepts, mesh_matcher.verified(), MAX_N,
    cm.CandidateExtractor.FILTERS[cm.ExtractionFilters.SIMPLE], gold_counter,
    FREQ_THRESHOLD
)
print()
dc_corpus_report = CorpusReport(anno.DiscontinuousConcept, docs, gold_docs)
dc_corpus_report.performance_summary()
dc_actual_errors = dc_corpus_report.error_analysis(
    gold_concepts, mesh_matcher.verified(), MAX_N,
    cm.CandidateExtractor.FILTERS[cm.ExtractionFilters.SIMPLE], gold_counter,
    FREQ_THRESHOLD
)
print()
types_report = TypesReport(final, gold_concepts)
types_report.performance_summary()
types_actual_errors = types_report.error_analysis(mesh_matcher.verified(),
                                                  MAX_N, gold_counter,
                                                  FREQ_THRESHOLD)

METRICS.add(mesh_matcher, cm.GoldMatcher(extractor, gold_concepts))

