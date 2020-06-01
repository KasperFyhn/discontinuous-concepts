from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import components as cm, buildpipeline
from pipeline.evaluation import CorpusReport, TypesReport,\
    gold_standard_concepts
from tqdm import tqdm

BLACKLIST = {'figure', 'fig.', 'table', 'various', 'other', 'most', 'many',
             'current'}

# RUN CONFIGURATIONS
CORPUS = 'acl'
run_with_test_docs = False
background_model = ngramcounting.NgramModel.load_model('aclarc', '_l7_min10')
configs = buildpipeline.Configuration(
    freq_threshold=2,
    extraction_filter=cm.ExtractionFilters.SIMPLE,
    coord_dcs=True, hypernym_dcs=True,
    consider_dcs_in_ranking=False
)

print('STEP 1: LOAD GOLD DOCUMENTS AND CONCEPTS')
print('Loading gold docs: ', end='')
gold_docs = dio.load_corpus(CORPUS)
if run_with_test_docs:
    if CORPUS == 'genia':
        test_gold_docs = dio.load_genia_corpus(
            path=dio.PATH_TO_GENIA + '/test docs/')
    else:
        test_gold_docs = dio.load_craft_corpus(
            path=dio.PATH_TO_CRAFT + '/test docs/')
    gold_docs += test_gold_docs
for doc in tqdm(gold_docs, desc='Removing unigram Concepts'):
    unigram_concepts = [c for c in doc.get_annotations(anno.Concept)
                        if len(c) == 1]
    for c in unigram_concepts:
        doc.remove_annotation(c)
gold_concepts = gold_standard_concepts(gold_docs)
gold_counter = conceptstats.count_concepts(gold_docs)

docs = dio.load_corpus(CORPUS, only_text=True)
if run_with_test_docs:
    if run_with_test_docs:
        if CORPUS == 'genia':
            test_docs = dio.load_genia_corpus(
                path=dio.PATH_TO_GENIA + '/test docs/', text_only=True)
        else:
            test_docs = dio.load_craft_corpus(
                path=dio.PATH_TO_CRAFT + '/test docs/', text_only=True)
        docs += test_docs

print('STEP 2: Run pre-processing and concept extraction pipeline')
preprocessing = buildpipeline.PreProcessingPipeline(configs)
docs, ngram_model = preprocessing.run(docs)
if background_model:
    ngram_model = ngramcounting.AggregateNgramModel(ngram_model,
                                                    background_model)

concept_extraction = buildpipeline.ConceptExtractionPipeline(configs)
extractor, metrics = concept_extraction.run(docs, ngram_model)
mesh_matcher = cm.MeshMatcher(extractor)

all_corpus_reports = []
all_dc_reports = []
all_types_reports = []

if run_with_test_docs:
    gold_docs = test_gold_docs
    test_docs_ids = {doc.id for doc in test_docs}
    docs = [d for d in docs if d.id in test_docs_ids]

for threshold in range(1, 2, 2):
    for doc in tqdm(docs, desc='Cleaning documents'):
        concepts = [c for c in doc.get_annotations(anno.Concept)]
        for c in concepts:
            doc.remove_annotation(c)

    final = metrics.rankers[cm.Metrics.VOTER].keep_proportion(1)
    concept_filter = cm.ConceptFilter(
        lambda c: metrics[c][cm.Metrics.C_VALUE] >= configs.c_value_threshold,
        lambda c: metrics[c][cm.Metrics.RECT_FREQ] >= threshold,
        lambda c: metrics[c][cm.Metrics.GLOSSEX] >= 1.5,
        lambda c: len(set(c).intersection(BLACKLIST)) < 1,
        # lambda c: METRICS[c][cm.Metrics.PMI_NL] >= 2,
        # lambda c: METRICS[c][cm.Metrics.TF_IDF] >= 100,
        filtering_method=cm.ConceptFilter.METHODS.ALL
    )
    final = concept_filter.apply(final)

    extractor.accept_candidates(set(final).union(mesh_matcher.verified()))

    print('\nSTEP 3: EVALUATE')
    print('THRESHOLD =', threshold)
    corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
    corpus_report.performance_summary()
    corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(),
                                 configs.max_n, '.+n', gold_counter,
                                 configs.freq_threshold)
    all_corpus_reports.append(corpus_report)
    print()
    dc_corpus_report = CorpusReport(anno.DiscontinuousConcept, docs, gold_docs)
    dc_corpus_report.performance_summary()
    dc_corpus_report.error_analysis(gold_concepts, mesh_matcher.verified(),
                                    configs.max_n, '.+n', gold_counter,
                                    configs.freq_threshold)
    all_dc_reports.append(dc_corpus_report)
    print()
    types_report = TypesReport(final, gold_concepts)
    types_report.performance_summary()
    types_report.error_analysis(mesh_matcher.verified(), configs.max_n,
                                gold_counter, configs.freq_threshold)
    all_types_reports.append(types_report)

    metrics.add(mesh_matcher, cm.GoldMatcher(extractor, gold_concepts))
