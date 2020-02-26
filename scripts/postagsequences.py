from datautils import dataio, annotations as anno
from collections import Counter, defaultdict
import re

# RUN CONFIGURATIONS
CORPUS = 'genia'
COLLAPSE = True
CROSS_COUNTING = ['concept', 'skip', 'super']




SUPER_SEQ_COLLAPSERS = {
    re.compile('n*(n,?)+cnn+'),
    re.compile('a*(a,?)+caa*n+')
}
CONCEPT_SEQ_COLLAPSERS = {
    re.compile('(a|n)+n')
}
COUNTER_TYPES = {'concept', 'super', 'skip'}


def make_sequence(tokens, collapser=None):

    sequence = [t.mapped_pos() if not t.pos == '#' else '#'
                for t in tokens]
    sequence_str = ''.join(sequence)
    if not COLLAPSE or not collapser:
        return sequence_str

    for pos_pattern in collapser:
        if pos_pattern.match(sequence_str):
            return pos_pattern.pattern
    return sequence_str  # sequence did not match a pattern


if CORPUS.lower() == 'craft':
    load_docs = dataio.load_craft_corpus
elif CORPUS.lower() == 'genia':
    load_docs = dataio.load_genia_corpus
docs = load_docs()

# prepare counters and example lists
samples = {}
counters = {}
for ct in COUNTER_TYPES:
    samples[ct] = defaultdict(list)
    counters[ct] = Counter()

for doc in docs:
    for dc in doc.get_annotations(anno.DiscontinuousConcept):
        concept = dc.get_concept()
        full_span = dc.get_covered_text()

        concept_tokens = [t for span in dc.spans
                          for t in doc.get_annotations_at(span, anno.Token)]
        all_tokens = [t for t in doc.get_annotations_at(dc.span, anno.Token)]
        skip_tokens = [t if t in concept_tokens  # actual token
                       else anno.Token(doc, (-1, -1), '#')  # skipped token
                       for t in all_tokens]

        concept_sequence = make_sequence(concept_tokens, CONCEPT_SEQ_COLLAPSERS)
        samples['concept'][concept_sequence].append(concept)
        counters['concept'][concept_sequence] += 1

        super_sequence = make_sequence(all_tokens, SUPER_SEQ_COLLAPSERS)
        samples['super'][super_sequence].append(full_span)
        counters['super'][super_sequence] += 1

        skip_sequence = make_sequence(skip_tokens)
        samples['skip'][skip_sequence].append(' '.join(t.get_covered_text()
                                                       for t in skip_tokens))
        counters['skip'][skip_sequence] += 1



# concept_pos_tag_seqs = Counter(c[1] for c in concepts_and_tokens)
# full_pos_tag_seqs = Counter(c[3] for c in concepts_and_tokens)
# both_pos_tag_seqs = Counter((c[1], c[3]) for c in concepts_and_tokens)
# n_concepts = len(concepts_and_tokens)
#
# print('\n\nMOST COMMON DISCONTINUOUS CONCEPT SEQUENCES PER SUPER SEQUENCE')
# print(f'{"Concept sequence":20}{"Super sequence":22}{"Count":>5}'
#       + f'{"Freq":>8}  Example')
# for item, count in both_pos_tag_seqs.most_common(30):
#     for concept in concepts_and_tokens:
#         if item[0] == concept[1] and item[1] == concept[3]:
#             example_super = concept[2]
#             example = concept[0]
#             break
#     print(f'{str(item[0]):20}{str(item[1]):22}{count:5}'
#           + f'{count/n_concepts:8.2}  {example_super} -> {example}')
