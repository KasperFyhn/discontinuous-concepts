import glob
import itertools
import re
from bs4 import BeautifulSoup
import bs4
import os
from nltk.corpus import brown
import multiprocessing as mp
from tqdm import tqdm
from datautils.annotations import *
from datautils.datapaths import PATH_TO_CRAFT, PATH_TO_GENIA, PATH_TO_PMC,\
    PATH_TO_MESH, PATH_TO_ACL, PATH_TO_ACLARC


def load_craft_document(doc_id, folder_path=PATH_TO_CRAFT, only_text=False):
    """Loads in the CRAFT document with the given ID and returns it as a
    Document object with annotations.py."""

    if type(doc_id) == tuple:
        doc_id, folder_path = doc_id

    path = os.path.join(folder_path, 'txt', doc_id + '.txt')
    doc_id = os.path.basename(path)[:-4]
    with open(path) as craft_file:
        doc = Document(doc_id, craft_file.read())

    if only_text:
        return doc

    # add sentence and token annotations; sents are recognized by their labels
    # make soup from POS XML
    path_to_xml = os.path.join(PATH_TO_CRAFT, 'part-of-speech', doc_id + '.xml')
    raw_xml = open(path_to_xml).read()
    pos_bs = BeautifulSoup(raw_xml, 'xml')

    # just run over all annotations.py and get the relevant info
    for tag in pos_bs.find_all('annotation'):
        span = (int(tag.span['start']), int(tag.span['end']))
        pos_tag = tag.find('class')['label']
        if pos_tag == 'sentence':
            sentence = Sentence(doc, span)
            doc.add_annotation(sentence)
        else:
            token = Token(doc, span, pos_tag)
            doc.add_annotation(token)

    # make beautiful soup from concept XML and add concept annotations
    path_to_xml = os.path.join(PATH_TO_CRAFT, 'concepts', doc_id + '.xml')
    raw_xml = open(path_to_xml).read()
    concept_bs = BeautifulSoup(raw_xml, 'xml')

    # just run over all annotations and get the relevant info
    for tag in concept_bs.find_all('annotation'):
        spans = tag.find_all('span')

        # if the concept is discontinuous, handle it differently: has more spans
        if len(spans) > 1:
            spans = [(int(span['start']), int(span['end']))
                     for span in spans]
            label = tag.find('class')['label']
            concept = DiscontinuousConcept(doc, spans, label)
        else:
            span = (int(tag.span['start']), int(tag.span['end']))
            label = tag.find('class')['label']
            concept = Concept(doc, span, label)

        doc.add_annotation(concept)

    # parse the appropriate .tree file
    path_to_tree = os.path.join(PATH_TO_CRAFT, 'treebank', doc_id + '.tree')
    raw_tree = open(path_to_tree).read()
    section_trees = raw_tree.split('\n')
    tokens_stack = doc.get_annotations(Token)
    for tree, sentence in zip(section_trees, doc.get_annotations(Sentence)):

        def get_constituents(sub_tree, stack):

            if re.match(r'\s?\(\S* [^(]*\)\s?', sub_tree):
                if '-NONE-' in sub_tree:
                    empty_span = (tokens_stack[0].span[0],
                                  tokens_stack[0].span[0])
                    return Token(doc, empty_span, '-NONE-')
                else:
                    return tokens_stack.pop(0)

            else:
                open_parentheses = re.finditer(r'^\s?\((\S*) .*', sub_tree)
                open_paren = next(open_parentheses)
                next_sub_trees = []
                stack.append(open_paren)
                # find the closing parentheses
                balance = -1
                last_sub_start = 0
                for i, c in enumerate(sub_tree):
                    if c == '(':
                        balance += 1
                        if balance == 1:
                            last_sub_start = i
                    elif c == ')':
                        balance -= 1
                        if balance == 0:
                            next_sub_trees.append((last_sub_start, i+1))
                    if balance < 0:
                        break

                sub_cons = []
                for nst in next_sub_trees:
                    start = nst[0]
                    end = nst[1]
                    sub_cons += [get_constituents(sub_tree[start:end], stack)]

                const = Constituent(doc, sub_cons, stack.pop().groups()[0])
                doc.add_annotation(const)
                return const

        get_constituents(tree, [])  # they are added to the doc in the function

    return doc


def load_craft_corpus(path=PATH_TO_CRAFT, text_only=False, as_generator=False):
    
    ids = craft_corpus_ids(path=path)
    if path != PATH_TO_CRAFT:
        ids = [(id_, path) for id_ in ids]

    if text_only:
        print('Loading CRAFT corpus without annotations ...')
        return [load_craft_document(doc_id, only_text=True) for doc_id in ids]

    if as_generator:
        return (load_craft_document(doc_id) for doc_id in ids)

    print('Loading CRAFT corpus ...')
    loaded_docs = []
    with mp.Pool() as pool:
        for doc in tqdm(pool.imap_unordered(load_craft_document, ids),
                        total=len(ids)):
            loaded_docs.append(doc)

    return loaded_docs


def craft_corpus_ids(path=PATH_TO_CRAFT):
    return [os.path.basename(name[:-4]) for name in glob.glob(path + 'txt/*')]


with open(PATH_TO_GENIA + 'MEDLINE-to-PMID') as map_file:
    _MEDLINE_TO_PMID = eval(map_file.read())
with open(PATH_TO_GENIA + 'constituent-quarantine') as quarantine_file:
    _CONST_QUARANTINE = {str(q) for q in eval(quarantine_file.read())}


def _flatten(li):
    return sum(([x] if not isinstance(x, list)
                else _flatten(x) for x in li), [])


def _resolve_child(tag, offset, doc):
    """Handles children tags met in a sentence."""

    # three types can be met: whitespace, w or cons
    if not tag.name:  # whitespace
        return len(tag)  # not an actual child

    elif tag.name == 'w':  # token
        # make span based on offset
        token_length = len(tag.get_text())
        token_span = (offset, offset + token_length)
        pos = tag['c']
        token = Token(doc, token_span, pos)
        doc.add_annotation(token)

        for c in tag.children:
            offset += _resolve_child(c, offset, doc)

        return token_length

    elif tag.name == 'cons':  # concept

        if tag.has_attr('sem'):
            label = tag['sem']
        else:
            label = ''

        if label:
            concept_spans = _resolve_cons_spans(tag, offset)
            # make concepts out of the retrieved spans
            for concept_span in concept_spans:
                if isinstance(concept_span, list):  # discontinuous concept
                    # first, merge adjacent spans
                    merged_spans = [concept_span.pop(0)]
                    for s in concept_span:
                        if s[0] - merged_spans[-1][1] < 2:
                            merged_spans[-1] = (merged_spans[-1][0], s[1])
                        else:
                            merged_spans.append(s)
                    if len(merged_spans) == 1:  # might not be discontinuous
                        concept = Concept(doc, merged_spans[0], label)
                    else:
                        concept = DiscontinuousConcept(doc, merged_spans, label)
                else:
                    concept = Concept(doc, concept_span, label)
                doc.add_annotation(concept)

        for c in tag.children:
            offset += _resolve_child(c, offset, doc)

        return len(tag.get_text())


def _resolve_cons_spans(cons_tag: bs4.Tag, offset):
    """Resolves the span(s) of a given <cons> tag by running
    recursively through its descendants. Due to complex
    constructions, the span options are flattened in the end."""

    try:
        label_ = cons_tag['sem']
    except:
        label_ = ''

    if 'AND' in label_ or 'OR' in label_:  # coordinated concepts!
        children_tags = [c for c in cons_tag.children]
        coord_words_indexes = set()
        prev_cons_index = None
        need_second = False
        for i, child in enumerate(children_tags):
            if not child.name:
                continue
            elif (child.name == 'w' and child['c'] == 'CC'
                  and child.string not in {'both', 'neither'})\
                    or child.get_text() in ',/':
                coord_words_indexes.add(prev_cons_index)
                need_second = True
            elif child.name == 'cons':
                prev_cons_index = i
                if need_second:
                    coord_words_indexes.add(prev_cons_index)
                    need_second = False

        common_before = []
        coordinated_words = []
        common_after = []

        for i, t in enumerate(children_tags):
            if not t.name:  # whitespace
                offset += len(t)
                continue
            elif i < min(coord_words_indexes):
                common_before.append(_resolve_cons_spans(t, offset))
            elif i in coord_words_indexes:
                coordinated_words += _resolve_cons_spans(t, offset)
            elif i > max(coord_words_indexes):
                common_after.append(_resolve_cons_spans(t, offset))
            # update offset
            offset += len(t.get_text())

        return [_flatten(list(option))
                for option in itertools.product(*common_before,
                                                coordinated_words,
                                                *common_after)
                ]

    else:  # single concept
        # make the span based on offsets and length of concept
        return [(offset, offset + len(cons_tag.get_text()))]


def load_genia_document(doc_id, folder_path=PATH_TO_GENIA, only_text=False):
    """Loads in the GENIA document with the given ID and returns it as a
    Document object with annotations.py."""

    if type(doc_id) == tuple:
        doc_id, folder_path = doc_id

    # create xml soup
    xml_path = os.path.join(folder_path, 'pos+concepts', doc_id + '.xml')
    xml_file = open(xml_path)
    soup = BeautifulSoup(xml_file.read(), 'xml')

    # doc text is made up of retrievable strings between tags in title+abstract
    doc_text = ''.join(soup.title.strings) + ''.join(soup.abstract.strings)
    # some double line breaks cause trouble in the offsets: correct to single
    # some stray spaces at the beginning or end of lines do the same
    doc_text = doc_text.strip().replace('\n\n', '\n').replace('\n ', '\n')
    doc = Document(doc_id, doc_text)

    if only_text:
        return doc

    offset = 0  # keeps track of how far in we are in the doc text
    # loop over sentences, extract the sentence and tokens + concepts within
    for sent in soup.find_all('sentence'):

        raw_sent = sent.get_text()  # also strings within tags
        sent_span = (offset, offset + len(raw_sent))
        sentence = Sentence(doc, sent_span)
        doc.add_annotation(sentence)

        for child in sent.children:
            offset += _resolve_child(child, offset, doc)

        # update before moving on to the next; remember the line break
        offset += 1

    # if the doc cannot get Constituent annotations, return at this point
    if str(doc_id) in _CONST_QUARANTINE:
        return doc

    # treebank annotations are found elsewhere. Handle these similar to before
    # create xml soup
    pmid = _MEDLINE_TO_PMID[doc_id]
    xml_path = os.path.join(PATH_TO_GENIA, 'treebank', pmid + '.xml')
    xml_file = open(xml_path)
    soup = BeautifulSoup(xml_file.read(), 'xml')

    tokens_stack = doc.get_annotations('Token')
    tokens_stack.reverse()

    for sentence in soup.find_all('sentence'):

        def get_constituents(sub_tree, stack):
            if sub_tree.name == 'tok':
                if not tokens_stack:
                    t = [Token(doc, (0, 0), '-NONE-')]
                else:
                    t = [tokens_stack.pop()]

                sub_tree_str = sub_tree.string.replace(' ', '')
                token_str = t[0].get_covered_text().replace(' ', '')
                # handle prefixes, other split tokens and missing tokens
                while sub_tree_str != token_str:
                    if sub_tree_str == '.':
                        for token in reversed(t):
                            tokens_stack.append(token)
                        empty_span = (t[0].span[0],
                                      t[0].span[0])
                        return [Token(doc, empty_span, '-NONE-')]

                    elif sub_tree_str in token_str:
                        if not token_str.endswith(sub_tree_str):
                            for token in reversed(t):
                                tokens_stack.append(token)
                            empty_span = (t[0].span[0],
                                          t[0].span[0])
                            return [Token(doc, empty_span, '-NONE-')]
                        else:
                            return t
                    elif token_str in sub_tree_str:
                        t.append(tokens_stack.pop())
                        token_str += t[-1].get_covered_text().replace(' ', '')
                    else:
                        print('Erroneous tokens:', sub_tree_str, token_str)
                        raise ValueError('Cannot create Constituent annotations'
                                         ' for' + str(doc_id))
                return t
            else:
                try:
                    stack.append(sub_tree['cat'])
                except KeyError:
                    stack.append(sub_tree.name)

                sub_cons = []
                for nst in sub_tree.children:
                    if isinstance(nst, bs4.NavigableString):
                        continue
                    sub_cons += get_constituents(nst, stack)

                if not sub_cons:
                    empty_span = (tokens_stack[-1].span[0],
                                  tokens_stack[-1].span[0])
                    empty_token = Token(doc, empty_span, '-NONE-')
                    sub_cons.append(empty_token)

                const = Constituent(doc, sub_cons, stack.pop())
                doc.add_annotation(const)

                return [const]

        tree = sentence
        get_constituents(tree, [])  # they are added to the doc in the function

    return doc


def genia_corpus_ids(path=PATH_TO_GENIA):
    ids = [os.path.basename(name[:-4])
           for name in glob.glob(os.path.join(path, 'pos+concepts', '*'))]
    return ids


def load_genia_corpus(path=PATH_TO_GENIA, text_only=False, as_generator=False):

    ids = genia_corpus_ids(path=path)
    if path != PATH_TO_GENIA:
        ids = [(id_, path) for id_ in ids]

    if text_only:
        print('Loading GENIA corpus without annotations ...')
        return list(tqdm(
            (load_genia_document(doc_id, only_text=True) for doc_id in ids),
            total=len(ids)))

    # some documents cause trouble; some are handled in the code, but not all.
    # the rest are listed in a quarantine file and will be skipped
    const_skip = 0
    for q in _CONST_QUARANTINE:
        if str(q) in ids:
            const_skip += 1

    if as_generator:
        return (load_genia_document(doc_id) for doc_id in ids)

    print('Loading GENIA corpus ...', end=' ')
    loaded_docs = []
    if _CONST_QUARANTINE:
        print(f'NOTE: {const_skip} files cannot get Constituent annotations!')
    else:
        print()
    with mp.Pool() as pool:
        for doc in tqdm(pool.imap_unordered(load_genia_document, ids),
                        total=len(ids)):
            loaded_docs.append(doc)

    return loaded_docs


def load_pmc_document(doc_id, folder_path=PATH_TO_PMC, only_text=False):

    # the documents are stored in nested folders,
    # e.g. PMC/PMC001XXXXXX.txt/PMC0012XXXXX/PMC1249490.txt
    # build this path based on the provided ID
    first_folder = 'PMC00' + doc_id[3] + 'XXXXXX.txt/'
    second_folder = 'PMC00' + doc_id[3:5] + 'XXXXX/'
    full_path = folder_path + first_folder + second_folder + doc_id
    if not full_path[-4:] == '.txt':
        full_path += '.txt'
    with open(full_path) as in_file:
        doc = Document(doc_id, in_file.read())
    if only_text:
        return doc
    else:
        sub_folder = 'PMC' + doc_id[3] + '/'
        annotation_file = os.path.join(folder_path, 'annotations/', sub_folder,
                                       doc_id + '.anno')
        doc.load_annotations_from_file(annotation_file)
        return doc


def pmc_corpus_ids(path=PATH_TO_PMC):
    doc_paths = glob.glob(path + '/PMC00*/**/*.txt', recursive=True)
    doc_ids = [os.path.basename(p)[:-4] for p in doc_paths]
    return doc_ids


def load_brown_doc(doc_id, only_text=False):
    sentences = [[tuple(t.split('/')) for t in s.strip().split()]
                 for s in brown.open(doc_id).read().split('\n') if s != '']
    text = '\n'.join(' '.join(t[0] for t in sent) for sent in sentences)
    doc = Document(doc_id, text)
    if only_text:
        return doc
    offset = 0
    for sent in sentences:
        if not sent:
            # for some reason, an empty string makes it through in doc ca19.
            # this block handles that
            offset += 1
            continue
        sent_begin = offset
        for t in sent:
            token_length = len(t[0])
            pos = t[1]
            doc.add_annotation(
                Token(doc, (offset, offset + token_length), pos.upper())
            )
            offset += token_length + 1
        sent_end = offset - 1
        doc.add_annotation(Sentence(doc, (sent_begin, sent_end)))

    return doc


def brown_corpus_ids():
    return brown.fileids()


def load_brown_corpus(text_only=False, as_generator=False):
    if as_generator:
        return (load_brown_doc(doc_id, only_text=text_only)
                for doc_id in brown_corpus_ids())
    else:
        print('Loading Brown corpus ...')
        return [load_brown_doc(doc_id, only_text=text_only)
                for doc_id in brown_corpus_ids()]


def load_acl_doc(doc_id, only_text=False, path=PATH_TO_ACL, file_number='1'):
    with open(path + 'split_files/' + doc_id + f'_{file_number}.vert') as file:
        lines = [line.strip() for line in file if not line == '']

    sentences = []
    sent_begin, sent_end = None, None
    concepts = []
    open_cons, cons_end = [], None
    tokens = []
    text = ''
    offset = 0

    for line in lines[1:-1]:
        if line.startswith('<s'):  # new sentence starts
            sent_begin = offset
        elif line.startswith('</s>'):  # sentence ends
            sent_end = offset - 1
            sentences.append((sent_begin, sent_end))
        elif line.startswith('<term'):  # new concept starts
            open_cons.append(offset)
        elif line.startswith('</term>'):  # concept ends
            if text[-1] == ' ':  # space in the end
                cons_end = offset - 1
            else:
                cons_end = offset
            concepts.append((open_cons.pop(), cons_end))
        elif line.startswith('<g/>'):  # no space
            text = text[:-1]  # delete last added space
            offset -= 1
        elif line.startswith('<'):  # another kind of tag, e.g. <doc
            continue  # skip
        else:
            token, _, pos = line.split()
            text += token + ' '
            token_begin = offset
            token_end = offset + len(token)
            tokens.append(((token_begin, token_end), pos))
            offset = len(text)
    text = text[:-1]  # delete last added space after last sentence

    doc = Document(doc_id, text)
    if only_text:
        return doc

    for sent_span in sentences:
        doc.add_annotation(Sentence(doc, sent_span))
    for token_span_pos in tokens:
        token_span, pos = token_span_pos
        doc.add_annotation(Token(doc, token_span, pos))
    for cons_span in concepts:
        doc.add_annotation(Concept(doc, cons_span))

    if file_number == '1' \
            and os.path.isfile(path + 'split_files/' + doc_id + f'_2.vert'):
        second_doc = load_acl_doc(doc_id, file_number='2')
        for c in second_doc.get_annotations(Concept):
            doc.add_annotation(Concept(doc, c.span))
    return doc


def acl_corpus_ids(path=PATH_TO_ACL):
    return sorted(set(os.path.basename(name[:-7])
                      for name in glob.glob(path + 'split_files/*')))


def load_acl_corpus(path=PATH_TO_ACL, text_only=False, as_generator=False):
    if as_generator:
        return (load_acl_doc(id_) for id_ in tqdm(acl_corpus_ids(path=path)))
    else:
        return [load_acl_doc(id_, only_text=text_only)
                for id_ in tqdm(acl_corpus_ids(path=path),
                                desc='Loading ACL 2.0 corpus')]


def load_aclarc_document(doc_id, folder_path=PATH_TO_ACLARC, only_text=False):

    # the documents are stored in nested folders,
    # e.g. PMC/PMC001XXXXXX.txt/PMC0012XXXXX/PMC1249490.txt
    # build this path based on the provided ID
    first_folder = 'txt/' + doc_id[0] + '/'
    second_folder = doc_id[:3] + '/'
    full_path = folder_path + first_folder + second_folder + doc_id
    if not full_path[-4:] == '.txt':
        full_path += '.txt'
    with open(full_path) as in_file:
        doc = Document(doc_id, in_file.read())
    if only_text:
        return doc
    else:
        annotation_file = os.path.join(folder_path, 'annotations/',
                                       doc_id + '.anno')
        doc.load_annotations_from_file(annotation_file)
        return doc


def aclarc_corpus_ids(path=PATH_TO_ACLARC):
    doc_paths = glob.glob(path + 'txt/**/*.txt', recursive=True)
    doc_ids = [os.path.basename(p)[:-4] for p in doc_paths]
    return doc_ids


def _read_mesh_terms(file, main_label, synonym_label):
    _lemma = nltk.WordNetLemmatizer()

    terms = {}
    current_term = ''
    for line in file:
        line = line.strip()
        if line.startswith(main_label):
            current_term = line.split('=')[1].strip().lower()
            term_tuple = tuple(current_term.split())
            terms[term_tuple] = current_term
            normalized_term = tuple(_lemma.lemmatize(w) for w in term_tuple)
            terms[normalized_term] = current_term

        elif line.startswith(synonym_label):
            synonym = line.split('=')[1].strip().lower()
            if ', ' in synonym:
                first, last = synonym.split(', ', maxsplit=1)
                synonym = ' '.join([last, first])
            elif '|' in synonym:
                synonym = synonym.split('|', maxsplit=1)[0]
            synonym_tuple = tuple(synonym.split())
            terms[synonym_tuple] = current_term
    return terms


def load_mesh_terms(descriptors=True, qualifiers=True, supplementary=True,
                    path=PATH_TO_MESH):
    print('Loading MeSH terms ...')
    terms = {}
    if descriptors:
        with open(path + 'd2020.bin') as f:
            terms.update(_read_mesh_terms(f, 'MH', 'ENTRY'))
    if qualifiers:
        with open(path + 'q2020.bin') as f:
            terms.update(_read_mesh_terms(f, 'SH', 'QX'))
    if supplementary:
        with open(path + 'c2020.bin') as f:
            terms.update(_read_mesh_terms(f, 'NM', 'SY'))

    return terms


def load_corpus(name: str, only_text=False, as_generator=False):
    corpus_loaders = {'genia': load_genia_corpus, 'craft': load_craft_corpus,
                      'acl': load_acl_corpus, 'brown': load_brown_corpus}
    try:
        loader = corpus_loaders[name.lower()]
        return loader(text_only=only_text, as_generator=as_generator)
    except KeyError as e:
        print('Corpus name not recognized!')
        raise e


def corpus_stats(corpora: dict):
    print('name      n_docs      tokens   concepts   types    disc.con   types'
          '    unig.con   types')
    print('-------------------------------------------------------------------'
          '--------------------')
    for name, corpus in corpora.items():
        n_docs = len(corpus)
        n_tokens = len([t for doc in corpus
                        for t in doc.get_annotations(Token)])
        concepts = [c for doc in corpus for c in doc.get_annotations(Concept)]
        dcs = [c for c in concepts if isinstance(c, DiscontinuousConcept)]
        unigram_concepts = [c for c in concepts
                            if len(c.normalized_concept()) == 1]
        n_concepts = len(concepts)
        n_dcs = len(dcs)
        n_conc_types = len(set(c.normalized_concept() for c in concepts))
        n_dc_types = len(set(c.normalized_concept() for c in dcs))
        n_uni_concepts = len(unigram_concepts)
        n_uni_types = len(set(c.normalized_concept() for c in unigram_concepts))

        print(f'{name:8}{n_docs:8}{n_tokens:12}{n_concepts:11}{n_conc_types:8}'
              f'{n_dcs:12}{n_dc_types:8}{n_uni_concepts:12}{n_uni_types:8}')


# test_craft = load_craft_document('11319941')
# test_craft_corpus = load_craft_corpus()
# test_genia = load_genia_document('98038749')
# test_genia_corpus = load_genia_corpus()
# test_pmc = load_pmc_document('PMC1249490')
# test_pmc_corpus = pmc_corpus_ids()
# test_brown_corpus = load_brown_corpus()



