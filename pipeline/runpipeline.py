from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import annotator
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
with annotator.CoreNlpServer() as server:
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
extractor = annotator.CandidateExtractor(
    pos_tag_filter=annotator.CandidateExtractor.FILTERS.simple, max_n=MAX_N
)
coord_extractor = annotator.CoordCandidateExtractor(
    pos_tag_filters=[annotator.CoordCandidateExtractor.FILTERS.simple],
    max_n=MAX_N
)
hypernym_extractor = annotator.HypernymCandidateExtractor(
    annotator.HypernymCandidateExtractor.FILTERS.simple, max_n=MAX_N
)
for doc in tqdm(docs, desc='Extracting candidates'):
    extractor.extract_candidates(doc)
    coord_extractor.extract_candidates(doc)
    hypernym_extractor.extract_candidates(doc)

print(f'Extracted {len(extractor.all_candidates)} continuous candidates and '
      f'{len(coord_extractor.all_candidates+hypernym_extractor.all_candidates)}'
      f' discontinuous candidates.')


print('\nSTEP 4: SCORE, RANK AND FILTER CANDIDATE CONCEPTS')
c_value = annotator.CValueRanker(extractor, C_VALUE_THRESHOLD)
rect_freq = annotator.RectifiedFreqRanker(extractor)
tf_idf = annotator.TfIdfRanker(extractor)
glossex = annotator.GlossexRanker(extractor, ngram_model)
pmi_nl = annotator.PmiNlRanker(extractor, ngram_model)
term_coherence = annotator.TermCoherenceRanker(extractor, ngram_model)
voter = annotator.VotingRanker(extractor, c_value, tf_idf, glossex, pmi_nl,
                               term_coherence)
final = voter.keep_proportion(.8)
mesh_matcher = annotator.MeshMatcher(extractor)
mesh_matcher.verify_candidates()

extractor.update(coord_extractor)
extractor.update(hypernym_extractor)

final = [c for c in final if rect_freq.value(c) >= FREQ_THRESHOLD]
extractor.accept_candidates(set(final).union(mesh_matcher.verified()))


print('\nSTEP 5: EVALUATE')
corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
corpus_report.performance_summary()
corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(), MAX_N,
                             annotator.CandidateExtractor.FILTERS.simple,
                             gold_counter, FREQ_THRESHOLD)
print()
dc_corpus_report = CorpusReport(anno.DiscontinuousConcept, docs, gold_docs)
dc_corpus_report.performance_summary()
dc_corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(), MAX_N,
                                annotator.CandidateExtractor.FILTERS.simple,
                                gold_counter, FREQ_THRESHOLD)
print()
types_report = TypesReport(final, gold_concepts)
types_report.performance_summary()
types_report.error_analysis(mesh_matcher.verified(), MAX_N, gold_counter,
                            FREQ_THRESHOLD)

metrics = annotator.Metrics(c_value, tf_idf, rect_freq, glossex, pmi_nl,
                            term_coherence)
