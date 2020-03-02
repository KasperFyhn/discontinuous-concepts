import datautils.dataio as dio
from corpusstats import ngrammodel, stats
from pipeline.annotator import CoreNlpServer, SimpleCandidateConceptExtractor
from tqdm import tqdm

# RUN CONFIGURATIONS
CORPUS = 'craft'
RUN_VERSION = '1'

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
ngrammodel.encode_corpus(colibri_model_name, list(doc_dict.keys()),
                         lambda x: doc_dict[x])
ngrammodel.make_colibri_model(colibri_model_name, spec_name)
ngram_model = stats.NgramModel.load_model(colibri_model_name, spec_name)

print('\nSTEP 3: EXTRACT CANDIDATE CONCEPTS')
candidate_extractor = SimpleCandidateConceptExtractor(
    pos_tag_filter=SimpleCandidateConceptExtractor.FILTERS.simple
)
for doc in tqdm(docs, desc='Extracting candidates'):
    candidate_extractor.extract_candidates(doc)
n_candidates = len(candidate_extractor.all_candidates)
n_docs = len(docs)
print(f'Extracted {n_candidates} candidate concepts from {n_docs} documents.')
