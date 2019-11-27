from datautils import dataio, annotations as anno
from collections import Counter

# RUN CONFIGURATIONS
CORPUS = 'genia'
COLLAPSE = True

# constants for use in make_pos_tag_sequence
POS_TAG_MAP = {
    '-LRB-': '(', '-RRB-': ')', 'HYPH': '-', '': 'Ø', '*': 'Pre-',
    'JJR': 'JJ', 'JJS': 'JJ',
    'NNP': 'NN', 'NNS': 'NN',
    'VBG': 'VB', 'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB'
}


def make_pos_tag_sequence(sequence):

    if COLLAPSE:
        short_sequence = []
        for tag in sequence:
            mapped_tag = POS_TAG_MAP[tag] if tag in POS_TAG_MAP else tag
            if short_sequence and mapped_tag == short_sequence[-1]:
                continue
            elif short_sequence and short_sequence[-1] == ','\
                    and mapped_tag == short_sequence[-2]:
                continue
            else:
                short_sequence.append(mapped_tag)
        return ' '.join(short_sequence)
    else:
        return ' '.join(POS_TAG_MAP[tag] if tag in POS_TAG_MAP else tag
                        for tag in sequence)


if CORPUS.lower() == 'craft':
    load_docs = dataio.load_craft_corpus
elif CORPUS.lower() == 'genia':
    load_docs = dataio.load_genia_corpus

docs = load_docs()

concepts_and_tokens = []

for doc in docs:
    disc_concepts = doc.get_annotations(anno.DiscontinuousConcept)

    for dc in disc_concepts:
        concept = dc.get_concept()
        full_span = dc.get_covered_text()
        concept_sequence = make_pos_tag_sequence(
            t.pos for span in dc.spans
            for t in doc.get_annotations_at(span, anno.Token)
        )
        full_sequence = make_pos_tag_sequence(
            t.pos for t in doc.get_annotations_at(dc.span, anno.Token)
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
