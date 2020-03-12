from datautils import dataio as dio, annotations as anno
from stats import ngramcounting, conceptstats
from pipeline import annotator
from pipeline.evaluation import CorpusReport
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'acl'
RUN_VERSION = '1'

SKIPGRAMS = False
C_VALUE_THRESHOLD = 1
FREQ_THRESHOLD = 0
MAX_N = 5

USE_PMC_CORPUS = False

print('STEP 1: ANNOTATE DOCUMENTS')
if CORPUS.lower() == 'genia':
    docs = dio.load_genia_corpus(text_only=True)
elif CORPUS.lower() == 'acl':
    docs = dio.load_acl_corpus(text_only=True)
else:
    docs = dio.load_craft_corpus(text_only=True)
with annotator.CoreNlpServer() as server:
    docs = server.annotate_batch(docs)


print('\nSTEP 2: MAKE/LOAD N-GRAM MODEL')
if USE_PMC_CORPUS:
    ngram_model = conceptstats.NgramModel.load_model('PMC', '_min20')
else:
    colibri_model_name = CORPUS + 'v' + RUN_VERSION
    spec_name = '_std'
    doc_dict = {doc.id: doc for doc in docs}
    ngramcounting.encode_corpus(colibri_model_name, list(doc_dict.keys()),
                                lambda x: doc_dict[x])
    ngramcounting.make_colibri_model(colibri_model_name, spec_name,
                                     mintokens=FREQ_THRESHOLD, maxlength=MAX_N,
                                     skipgrams=SKIPGRAMS)
    ngram_model = conceptstats.NgramModel.load_model(colibri_model_name,
                                                     spec_name)


print('\nSTEP 3: EXTRACT CANDIDATE CONCEPTS')
extractor = annotator.CandidateExtractor(
    pos_tag_filter=annotator.CandidateExtractor.FILTERS.unsilo,
    max_n=MAX_N
)
dc_extractor = annotator.DiscCandidateExtractor(
    pos_tag_filters=[annotator.DiscCandidateExtractor.FILTERS.coord_unsilo],
    max_n=MAX_N
)
for doc in tqdm(docs, desc='Extracting candidates'):
    extractor.extract_candidates(doc)
    dc_extractor.extract_candidates(doc)
print(f'Extracted {len(extractor.all_candidates)} continuous candidates and '
      f'{len(dc_extractor.all_candidates)} discontinuous candidates.')

extractor.update(dc_extractor)


print('\nSTEP 4: RANK, FILTER AND ADD CANDIDATE CONCEPTS')
ranker = annotator.CValueRanker(extractor, C_VALUE_THRESHOLD, ngram_model,
                                SKIPGRAMS)
ranker.rank()
final = ranker.filter_at_value(C_VALUE_THRESHOLD)
mesh_matcher = annotator.MeshMatcher(extractor)
mesh_matcher.verify_candidates()

extractor.accept_candidates(set(final).union(mesh_matcher.verified()))


print('\nSTEP 5: EVALUATE')
if CORPUS.lower() == 'genia':
    gold_docs = dio.load_genia_corpus()
elif CORPUS.lower() == 'acl':
    gold_docs = dio.load_acl_corpus()
else:
    gold_docs = dio.load_craft_corpus()

corpus_report = CorpusReport(anno.Concept, docs, gold_docs)
corpus_report.performance_summary()

gold_concepts = conceptstats.gold_standard_concepts(gold_docs)

conceptstats.performance(final, gold_concepts)
conceptstats.precision_at_k(final, gold_concepts, (100, 200, 500, 1000, 5000))

