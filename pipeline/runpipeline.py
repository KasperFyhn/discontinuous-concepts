import datautils.dataio as dio
from stats import ngramcounting, conceptstats
from pipeline.annotator import CoreNlpServer, SimpleCandidateConceptExtractor
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'craft'
RUN_VERSION = '1'

SKIPGRAMS = False
C_VALUE_THRESHOLD = 3


print('STEP 1: ANNOTATE DOCUMENTS')
if CORPUS.lower() == 'genia':
    docs = dio.load_genia_corpus(text_only=True)
else:
    docs = dio.load_craft_corpus(text_only=True)
with CoreNlpServer() as server:
    docs = server.annotate_batch(docs)


print('\nSTEP 2: MAKE N-GRAM MODEL')
colibri_model_name = CORPUS + 'v' + RUN_VERSION
spec_name = '_std'
doc_dict = {doc.id: doc for doc in docs}
ngramcounting.encode_corpus(colibri_model_name, list(doc_dict.keys()),
                            lambda x: doc_dict[x])
ngramcounting.make_colibri_model(colibri_model_name, spec_name)
ngram_model = conceptstats.NgramModel.load_model(colibri_model_name, spec_name)


print('\nSTEP 3: EXTRACT CANDIDATE CONCEPTS')
candidate_extractor = SimpleCandidateConceptExtractor(
    pos_tag_filter=SimpleCandidateConceptExtractor.FILTERS.unsilo
)
for doc in tqdm(docs, desc='Extracting candidates'):
    candidate_extractor.extract_candidates(doc)
n_candidates = len(candidate_extractor.all_candidates)
n_docs = len(docs)
candidate_terms = candidate_extractor.candidate_types()
print(f'Extracted {n_candidates} candidate concepts ({len(candidate_terms)} '
      f'types) from {n_docs} documents.')


print('\nSTEP 4: RANK AND FILTER CANDIDATE CONCEPTS')
c_values = conceptstats.calculate_c_values(candidate_terms, C_VALUE_THRESHOLD,
                                           ngram_model, skipgrams=SKIPGRAMS)
tf_idf_values = conceptstats.calculate_tf_idf_values(candidate_terms, docs,
                                                     ngram_model)
final = {c for c, v in c_values.items() if v > C_VALUE_THRESHOLD}
print(f'Filtered out {len(candidate_terms)} concept types, thus leaving '
      f'{len(final)} concept types in the final list.')


print('\nSTEP 5: EVALUATE')
if CORPUS.lower() == 'genia':
    gold_docs = dio.load_genia_corpus()
else:
    gold_docs = dio.load_craft_corpus()
gold_concepts = conceptstats.gold_standard_concepts(gold_docs)

conceptstats.performance(final, gold_concepts)
conceptstats.precision_at_k(sorted(final, reverse=True, key=lambda x: c_values[x]),
                            gold_concepts, (100, 200, 300, 500, 1000, 5000, 10000))

