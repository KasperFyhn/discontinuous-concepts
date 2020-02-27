from datautils import dataio, annotations as anno, datapaths as dp
from collections import Counter, defaultdict
import re
import os
import sys

# RUN CONFIGURATIONS
CORPUS = 'genia'
COLLAPSE = True
CROSS_COUNTING = ['concept', 'super', 'skip']  # order: concept, super, skip


SUPER_SEQ_COLLAPSERS = {
    re.compile('n*(n,?)+cnn+'),
    re.compile('a*(a,?)+caa*n+')
}
CONCEPT_SEQ_COLLAPSERS = {
    re.compile('(a|n)+n')
}
SAMPLE_CATEGORIES = {'concept', 'super', 'skip', 'cross-count'}


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
else:
    print('No valid corpus stated. Exiting ...')
    sys.exit(1)
docs = load_docs()

# prepare counters and example lists
samples_after_category = {}
for ct in SAMPLE_CATEGORIES:
    samples_after_category[ct] = defaultdict(list)

for doc in docs:
    for dc in doc.get_annotations(anno.DiscontinuousConcept):
        concept_tokens = dc.get_concept_tokens()
        all_tokens = dc.get_tokens()
        skip_tokens = [t if t in concept_tokens  # actual token
                       else anno.Token(doc, (-1, -1), '#')  # skipped token
                       for t in all_tokens]

        cross_count_type = []
        cross_count_example = []

        concept_sequence = make_sequence(concept_tokens, CONCEPT_SEQ_COLLAPSERS)
        concept_text = dc.get_concept()
        samples_after_category['concept'][concept_sequence].append(concept_text)
        if 'concept' in CROSS_COUNTING:
            cross_count_type.append(concept_sequence)
            cross_count_example.append(concept_text)

        super_sequence = make_sequence(all_tokens, SUPER_SEQ_COLLAPSERS)
        super_text = dc.get_covered_text()
        samples_after_category['super'][super_sequence].append(super_text)
        if 'super' in CROSS_COUNTING:
            cross_count_type.append(super_sequence)
            cross_count_example.append(super_text)

        skip_sequence = make_sequence(skip_tokens)
        skip_text = ' '.join(t.get_covered_text() if t.get_covered_text()
                             else '{*}' for t in skip_tokens)
        samples_after_category['skip'][skip_sequence].append(skip_text)
        if 'skip' in CROSS_COUNTING:
            cross_count_type.append(skip_sequence)
            cross_count_example.append(skip_text)

        samples_after_category['cross-count'][tuple(cross_count_type)].append(
            tuple(cross_count_example)
        )

counters = {}
for sample_category, samples in samples_after_category.items():
    counters[sample_category] = Counter({type_: len(samples[type_])
                                         for type_ in samples})


def category_summary(category, most_common=20):
    counter = counters[category]
    print(f'\nMOST COMMON {category.upper()} SEQUENCES')
    max_sample_type_length = max(len(str(k))
                                 for k, v in counter.most_common(most_common))
    extra_width = ' ' * (max_sample_type_length - len(str('Type')))
    print('Type' + extra_width, 'Count', 'Example', sep='\t')
    for sample_type, count in counter.most_common(most_common):
        example = samples_after_category[category][sample_type][0]
        extra_width = ' ' * (max_sample_type_length - len(str(sample_type)))
        print(str(sample_type) + extra_width, f'{count:5}', example, sep='\t')


def to_csv(category, directory=dp.PATH_TO_PAPER, most_common=None):
    # handle the data file
    data_folder = directory + '/pos-seq_data/'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if category == 'cross-count':
        outfile = data_folder + CORPUS + '_' + '+'.join(CROSS_COUNTING) + '.csv'
    else:
        outfile = data_folder + CORPUS + '_' + category + '.csv'
    with open(outfile, 'w+') as out:
        # print the data
        counter = counters[category]
        if category == 'cross-count':  # gives more columns
            type_headers = [type_ + '_type' for type_ in CROSS_COUNTING]
            example_headers = [type_ + '_example' for type_ in CROSS_COUNTING]
            print(*type_headers, 'Count', *example_headers, sep='\t', file=out)
            for sample_type, count in counter.most_common(most_common):
                example = samples_after_category[category][sample_type][0]
                print(*sample_type, count, *example, sep='\t', file=out)
        else:
            print('Type', 'Count', 'Example', sep='\t', file=out)
            for sample_type, count in counter.most_common(most_common):
                example = samples_after_category[category][sample_type][0]
                print(sample_type, count, example, sep='\t', file=out)


category_summary('cross-count')
