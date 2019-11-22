import dataio as dio
from collections import Counter

# RUN CONFIGURATIONS
CORPUS = 'CRAFT'

POS_TAG_MAP = {
    '-LRB-': '(', '-RRB-': ')', 'HYPH': '-', '``': 'Ø',
    'JJR': 'JJ', 'JJS': 'JJ',
    'NNP': 'NN', 'NNS': 'NN',
    'VBG': 'VB', 'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB'
}


def make_pos_tag_sequence(sequence):
    return ' '.join(POS_TAG_MAP[tag] if tag in POS_TAG_MAP else tag
                    for tag in sequence)


if CORPUS.lower() == 'craft':
    load_docs = dio.load_craft_corpus
elif CORPUS.lower() == 'genia':
    load_docs = dio.load_genia_corpus

docs = load_docs()

concepts_and_tokens = []

for i, doc in enumerate(docs):
    print(f'Processing document {i+1} of {len(docs)}', end='\r')
    disc_concepts = doc.get_annotations(dio.DiscontinuousConcept)

    for dc in disc_concepts:
        concept = dc.get_concept()
        full_span = dc.get_covered_text()
        concept_sequence = make_pos_tag_sequence(
            t.pos for span in dc.spans
            for t in doc.get_annotations_at(span, dio.Token)
        )
        full_sequence = make_pos_tag_sequence(
            t.pos for t in doc.get_annotations_at(dc.span, dio.Token)
        )
        concepts_and_tokens.append(
            (concept, concept_sequence, full_span, full_sequence)
        )

concept_pos_tag_seqs = Counter(c[1] for c in concepts_and_tokens)
full_pos_tag_seqs = Counter(c[3] for c in concepts_and_tokens)
both_pos_tag_seqs = Counter((c[1], c[3]) for c in concepts_and_tokens)
n_concepts = len(concepts_and_tokens)

print('\n\nMOST COMMON DISCONTINUOUS CONCEPT SEQUENCES PER SUPER SEQUENCE')
print(f'{"Concept sequence":20}{"Super sequence":22}{"Count":>5}'
      + f'{"Freq":>8}  Example')
for item in both_pos_tag_seqs.most_common(20):
    for concept in concepts_and_tokens:
        if item[0][0] == concept[1] and item[0][1] == concept[3]:
            example_super = concept[2]
            example = concept[0]
            break
    print(f'{str(item[0][0]):20}{str(item[0][1]):22}{item[1]:5}'
          + f'{item[1]/n_concepts:8.2}  {example_super} -> {example}')
