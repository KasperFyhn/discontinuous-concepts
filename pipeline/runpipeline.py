from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import annotator as an
from pipeline.evaluation import CorpusReport, TypesReport,\
    gold_standard_concepts
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'genia'
RUN_VERSION = '1'

SKIPGRAMS = False
C_VALUE_THRESHOLD = 2
FREQ_THRESHOLD = 2
MAX_N = 7

METRICS = an.Metrics()
FILTER = an.ConceptFilter(
    lambda c: METRICS[c][an.Metrics.C_VALUE] > C_VALUE_THRESHOLD,
    lambda c: METRICS[c][an.Metrics.RECT_FREQ] >= FREQ_THRESHOLD,
    # lambda c: METRICS[c][an.Metrics.PMI_NL] > .5,
    filtering_method=an.ConceptFilter.METHODS.ALL
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
with an.CoreNlpServer() as server:
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
extractor = an.CandidateExtractor(
    pos_tag_filter=an.CandidateExtractor.FILTERS.simple, max_n=MAX_N
)
coord_extractor = an.CoordCandidateExtractor(
    pos_tag_filters=[an.CoordCandidateExtractor.FILTERS.simple],
    max_n=MAX_N
)
hypernym_extractor = an.HypernymCandidateExtractor(
    an.HypernymCandidateExtractor.FILTERS.simple, max_n=MAX_N
)
for doc in tqdm(docs, desc='Extracting candidates'):
    extractor.extract_candidates(doc)
    coord_extractor.extract_candidates(doc)
    hypernym_extractor.extract_candidates(doc)

print(f'Extracted {len(extractor.all_candidates)} continuous candidates and '
      f'{len(coord_extractor.all_candidates+hypernym_extractor.all_candidates)}'
      f' discontinuous candidates.')


print('\nSTEP 4: SCORE, RANK AND FILTER CANDIDATE CONCEPTS')
mesh_matcher = an.MeshMatcher(extractor)
mesh_matcher.verify_candidates()

c_value = an.CValueRanker(extractor, C_VALUE_THRESHOLD)
rect_freq = an.RectifiedFreqRanker(extractor)
tf_idf = an.TfIdfRanker(extractor)
glossex = an.GlossexRanker(extractor, ngram_model)
pmi_nl = an.PmiNlRanker(extractor, ngram_model)
term_coherence = an.TermCoherenceRanker(extractor, ngram_model)
voter = an.VotingRanker(extractor, c_value, tf_idf, glossex, pmi_nl,
                        term_coherence)
METRICS.add(c_value, rect_freq, tf_idf, glossex, pmi_nl, term_coherence, voter)

final = c_value.keep_proportion(1)  # keep all, but ranked
final = FILTER.apply(final)  # then filter

extractor.update(coord_extractor)
#extractor.update(hypernym_extractor)

extractor.accept_candidates(set(final).union(mesh_matcher.verified()))

print('\nSTEP 5: EVALUATE')
corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
corpus_report.performance_summary()
corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(), MAX_N,
                             an.CandidateExtractor.FILTERS.simple,
                             gold_counter, FREQ_THRESHOLD)
print()
dc_corpus_report = CorpusReport(anno.DiscontinuousConcept, docs, gold_docs)
dc_corpus_report.performance_summary()
dc_corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(), MAX_N,
                                an.CandidateExtractor.FILTERS.simple,
                                gold_counter, FREQ_THRESHOLD)
print()
types_report = TypesReport(final, gold_concepts)
types_report.performance_summary()
types_report.error_analysis(mesh_matcher.verified(), MAX_N, gold_counter,
                            FREQ_THRESHOLD)


