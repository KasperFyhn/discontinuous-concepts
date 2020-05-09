from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import components as cm, buildpipeline
from pipeline.evaluation import CorpusReport, TypesReport,\
    gold_standard_concepts
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'genia'
background_model = None
#  ngramcounting.NgramModel.load_model('genia_pmc_20k', '_l7_min10')
configs = buildpipeline.Configuration(freq_threshold=4,
                                      consider_dcs_in_ranking=True)

print('STEP 1: LOAD GOLD DOCUMENTS AND CONCEPTS')
print('Loading gold docs: ', end='')
gold_docs = dio.load_corpus(CORPUS)
for doc in tqdm(gold_docs, desc='Removing unigram Concepts'):
    unigram_concepts = [c for c in doc.get_annotations(anno.Concept)
                        if len(c) == 1]
    for c in unigram_concepts:
        doc.remove_annotation(c)
gold_concepts = gold_standard_concepts(gold_docs)
gold_counter = conceptstats.count_concepts(gold_docs)

docs = dio.load_corpus(CORPUS, only_text=True)

print('STEP 2: Run pre-processing and concept extraction pipeline')
preprocessing = buildpipeline.PreProcessingPipeline(configs)
docs, ngram_model = preprocessing.run(docs)
if background_model:
    ngram_model = ngramcounting.AggregateNgramModel(ngram_model,
                                                    background_model)

concept_extraction = buildpipeline.ConceptExtractionPipeline(configs)
extractor, metrics = concept_extraction.run(docs, ngram_model)

final = metrics.rankers[cm.Metrics.VOTER].keep_proportion(1)
concept_filter = cm.ConceptFilter(
    lambda c: metrics[c][cm.Metrics.C_VALUE] >= configs.c_value_threshold,
    lambda c: metrics[c][cm.Metrics.RECT_FREQ] >= configs.freq_threshold,
    lambda c: metrics[c][cm.Metrics.GLOSSEX] >= 1.5,
    # lambda c: METRICS[c][cm.Metrics.PMI_NL] >= 2,
    # lambda c: METRICS[c][cm.Metrics.TF_IDF] >= 100,
    filtering_method=cm.ConceptFilter.METHODS.ALL
)
final = concept_filter.apply(final)
mesh_matcher = cm.MeshMatcher(extractor)
extractor.accept_candidates(set(final).union(mesh_matcher.verified()))

print('\nSTEP 3: EVALUATE')
corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
corpus_report.performance_summary()
actual_errors = corpus_report.error_analysis(
    gold_concepts, mesh_matcher.verified(), configs.max_n, '.+n', gold_counter,
    configs.freq_threshold
)
print()
dc_corpus_report = CorpusReport(anno.DiscontinuousConcept, docs, gold_docs)
dc_corpus_report.performance_summary()
dc_actual_errors = dc_corpus_report.error_analysis(
    gold_concepts, mesh_matcher.verified(), configs.max_n, '.+n', gold_counter,
    configs.freq_threshold
)
print()
types_report = TypesReport(final, gold_concepts)
types_report.performance_summary()
types_actual_errors = types_report.error_analysis(mesh_matcher.verified(),
                                                  configs.max_n, gold_counter,
                                                  configs.freq_threshold)

metrics.add(mesh_matcher, cm.GoldMatcher(extractor, gold_concepts))

