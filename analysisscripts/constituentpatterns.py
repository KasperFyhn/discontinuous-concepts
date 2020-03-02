from datautils import dataio, annotations as anno, datapaths as dp
from collections import Counter, defaultdict
from tqdm import tqdm
import multiprocessing as mp
import os
import sys

# RUN CONFIGURATIONS
CORPUS = 'genia'

ONLY_DISCONTINUOUS = True

SAMPLE_CATEGORIES = {'concept', 'super', 'cross-count'}
CROSS_COUNTING = ['concept', 'super']

if CORPUS.lower() == 'craft':
    load_docs = dataio.load_craft_corpus
elif CORPUS.lower() == 'genia':
    load_docs = dataio.load_genia_corpus
else:
    print('No valid corpus stated. Exiting ...')
    sys.exit(1)
docs = load_docs()
if CORPUS.lower() == 'genia':
    # not all docs in genia get Constituent annotations; if so, leave them out
    docs = [doc for doc in docs if doc.get_annotations(anno.Constituent)]


def _concept_constituents_from_doc(doc):
    super_samples = defaultdict(list)
    concept_samples = defaultdict(list)
    cross_count_samples = defaultdict(list)
    concepts = doc.get_annotations(
        anno.DiscontinuousConcept if ONLY_DISCONTINUOUS else anno.Concept
    )
    for concept in concepts:

        const = doc.get_annotations_at(concept.span, anno.Constituent)[0]
        const_structure = const.structure()
        const_text = const.get_covered_text()
        super_samples[const_structure].append(const_text)

        if isinstance(concept, anno.DiscontinuousConcept):
            concept_tokens = concept.get_concept_tokens()
        else:
            concept_tokens = concept.get_tokens()
        skip_structure = _skipped_structure(const, concept_tokens)

        if isinstance(concept, anno.DiscontinuousConcept):
            concept_text = concept.get_concept()
        else:
            concept_text = concept.get_covered_text()
        concept_samples[skip_structure].append(concept_text)

        cross_count_samples[(const_structure, skip_structure)].append(
            (const_text, concept_text)
        )

    return {'super': super_samples, 'concept': concept_samples,
            'cross-count': cross_count_samples}


def _skipped_structure(const, allowed_tokens):
    return '(' + const.label + ' ' + ' '.join(
        _skipped_structure(c, allowed_tokens) if isinstance(c, anno.Constituent)
        else '#' + c.mapped_pos() if c not in allowed_tokens
        else c.mapped_pos()
        for c in const.constituents
    ) + ')'


def category_summary(category, most_common=20):
    counter = counters[category]
    print(f'\nMOST COMMON {category.upper()} CONSTITUENT STRUCTURES')
    max_sample_type_length = max(len(str(k))
                                 for k, v in counter.most_common(most_common))
    extra_width = ' ' * (max_sample_type_length - len(str('Type')))
    print('Type' + extra_width, 'Count', 'Example', sep='\t')
    for type_, count in counter.most_common(most_common):
        example = samples_after_category[category][type_][0]
        extra_width = ' ' * (max_sample_type_length - len(str(type_)))
        print(str(type_) + extra_width, f'{count:5}', example, sep='\t')


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
            for type_, count in counter.most_common(most_common):
                example = samples_after_category[category][type_][0]
                print(*type_, count, *example, sep='\t', file=out)
        else:
            print('Type', 'Count', 'Example', sep='\t', file=out)
            for type_, count in counter.most_common(most_common):
                example = samples_after_category[category][type_][0]
                print(type_, count, example, sep='\t', file=out)


counters = {}
samples_after_category = {}
for sample_category in SAMPLE_CATEGORIES:
    counters[sample_category] = Counter()
    samples_after_category[sample_category] = defaultdict(list)

with mp.Pool() as p:
    for samples in tqdm(
            p.imap_unordered(_concept_constituents_from_doc, docs),
            desc='Retrieving Constituent annotations', total=len(docs)):
        for sample_category, sample_dict in samples.items():
            for sample_type, sample in sample_dict.items():
                counters[sample_category][sample_type] += len(sample)
                samples_after_category[sample_category][sample_type] += sample

category_summary('super', None)
