from datautils import dataio, annotations as anno
from collections import Counter, defaultdict
from tqdm import tqdm
import multiprocessing as mp

# RUN CONFIGURATIONS
CORPUS = 'genia'

ONLY_DISCONTINUOUS = True

if CORPUS.lower() == 'craft':
    load_docs = dataio.load_craft_corpus
elif CORPUS.lower() == 'genia':
    load_docs = dataio.load_genia_corpus
docs = load_docs()
if CORPUS.lower() == 'genia':
    # not all docs in genia have Constituent annotations; if so, leave them out
    docs = [doc for doc in docs if doc.get_annotations(anno.Constituent)]


def _concept_constituents_from_doc(doc):
    structure_counter = Counter()
    structure_samples = defaultdict(set)
    concepts = doc.get_annotations(
        anno.DiscontinuousConcept if ONLY_DISCONTINUOUS else anno.Concept
    )
    for concept in concepts:
        try:
            const = doc.get_annotations_at(concept.span, anno.Constituent)[0]
            s = const.structure()
            structure_counter[s] += 1
            structure_samples[s].add(concept)
        except Exception as e:
            print(type(e), 'in', doc.id)
            print('Concept:', concept)
    return structure_counter, structure_samples


master_structures = Counter()
master_samples = defaultdict(set)
with mp.Pool() as p:
    for structure_counts, samples in tqdm(
            p.imap_unordered(_concept_constituents_from_doc, docs),
            desc='Retrieving Constituent annotations', total=len(docs)
    ):
        master_structures.update(structure_counts)
        for key, sample_set in samples.items():
            master_samples[key].update(sample_set)


