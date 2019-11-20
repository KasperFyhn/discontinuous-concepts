import glob
import itertools
import re
import sys
from bs4 import BeautifulSoup
import bs4
import os
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict


class Document:
    """A representation of a document which contains raw text and annotations"""

    def __init__(self, doc_id: str, raw_text: str):
        self.id = doc_id
        self._text = raw_text
        self._annotations = defaultdict(list)

    @classmethod
    def from_pickle(cls, path_in):
        """Load a pickled Document object from a path."""

        pass

    def pickle(self, path_out):

        pass

    def add_annotation(self, annotation):

        key = type(annotation)
        self._annotations[key].append(annotation)

    def get_annotations_at(self, span, annotation_type=None):
        if not annotation_type:
            annotation_type = Annotation

        if isinstance(annotation_type, str):
            try:
                annotation_type = getattr(sys.modules[__name__],
                                          annotation_type)
            except AttributeError:
                print('No annotation type of that name!')
                return None

        if not annotation_type == Constituent:
            return [a for a in self.get_annotations(annotation_type)
                    if span[0] <= a.span[0] <= span[1]
                    or span[0] <= a.span[1] <= span[1]]
        else:
            # Constituent annotations are handled differently because they are
            # embedded; instead, return the one closest to the given span
            potential = [a for a in self.get_annotations(annotation_type)
                         if span[0] <= a.span[0] <= span[1]
                         or span[0] <= a.span[1] <= span[1]]
            start = span[0]
            end = span[1]
            closest = None
            closest_dist = None
            for p in potential:
                dist = abs(p.span[0] - start) + abs(p.span[1] - end)
                if not closest or dist < closest_dist:
                    closest = p
                    closest_dist = dist
            return [closest]

    def get_annotations(self, annotation_type):
        if isinstance(annotation_type, str):
            try:
                annotation_type = getattr(sys.modules[__name__],
                                          annotation_type)
            except AttributeError:
                print('No annotation type of that name!')
                return None
        return sorted((a for anno_list in self._annotations.values()
                       for a in anno_list if isinstance(a, annotation_type)),
                      key=lambda x: x.span)

    def get_text(self):
        return self._text

    def __str__(self):
        return self._text

    def __repr__(self):
        return f'DOC:{self.id}'


class Annotation:
    """Basic annotation class. Contains only reference to a document and the
    span(s) of the annotations."""

    def __init__(self, document: Document, span: tuple):
        self.document = document
        self.span = span

    def get_covered_text(self):
        """Returns the text covered by the annotation."""

        start = self.span[0]
        end = self.span[1]
        return self.document.get_text()[start:end]

    def get_context(self, char_window=40):
        start = self.span[0] - char_window
        if start < 0:
            start = 0
        end = self.span[1] + char_window
        if end > len(self.document.get_text()):
            end = len(self.document.get_text())
        build_string = \
            self.document.get_text()[start:self.span[0]-1] \
            + "     " + self.get_covered_text() + "     " \
            + self.document.get_text()[self.span[1]:end]
        return build_string

    def merge_with(self, another_annotation):

        another_annotation.span = (self.span[0], another_annotation.span[1])
        del self
        return another_annotation

    def __repr__(self):
        return self.__class__.__name__ + "('" + self.get_covered_text() + "'"\
               + str(self.span) + ')'


class Sentence(Annotation):
    pass


class Token(Annotation):

    def __init__(self, document: Document, span: tuple, pos_tag: str):
        super().__init__(document, span)
        self.pos = pos_tag

    def __repr__(self):
        return super().__repr__() + '\\' + self.pos


class Concept(Annotation):

    def __init__(self, document: Document, span: tuple, label: str):
        super().__init__(document, span)
        self.label = label


class DiscontinuousConcept(Concept):

    def __init__(self, document: Document, spans: list, label: str):
        full_span = (spans[0][0], spans[-1][1])
        super().__init__(document, full_span, label)
        self.spans = spans

    def get_concept(self):
        """Returns only the concept, disregarding tokens within the full span
        that are not part of the concept."""

        return ' '.join(self.document.get_text()[s[0]:s[1]] for s in self.spans)

    def __repr__(self):
        return self.__class__.__name__ + "('" + self.get_concept() + "'"\
               + str(self.spans) + ')'


class Constituent(Annotation):

    def __init__(self, document, constituents, label):
        super().__init__(document, (constituents[0].span[0],
                                    constituents[-1].span[1])
                         )
        self.constituents = constituents
        self.label = label

    def __repr__(self):
        return super().__repr__() + '\\' + self.label

    def __str__(self, depth=1):
        return self.__repr__() + '\n' + '\t'*depth \
               + ('\n' + '\t'*depth).join(
            c.__str__(depth=depth+1) if isinstance(c, Constituent) else str(c)
            for c in self.constituents
        )


def _flatten(li):
    return sum(([x] if not isinstance(x, list)
                else _flatten(x) for x in li), [])


def load_craft_document(doc_id, folder_path='data/CRAFT/txt/', only_text=False):
    """Loads in the CRAFT document with the given ID and returns it as a
    Document object with annotations."""


    path = folder_path + doc_id + '.txt'
    doc_id = os.path.basename(path)[:-4]
    with open(path) as craft_file:
        doc = Document(doc_id, craft_file.read())

    if only_text:
        return doc

    # add sentence and token annotations; sents are recognized by their labels
    # make soup from POS XML
    path_to_xml = 'data/CRAFT/part-of-speech/' + doc_id + '.xml'
    raw_xml = open(path_to_xml).read()
    pos_bs = BeautifulSoup(raw_xml, 'xml')

    # just run over all annotations and get the relevant info
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
    path_to_xml = 'data/CRAFT/concepts/' + doc_id + '.xml'
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
    path_to_tree = 'data/CRAFT/treebank/' + doc_id + '.tree'
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


def load_craft_corpus(path='./data/CRAFT/txt/'):

    ids = [os.path.basename(name[:-4]) for name in glob.glob(path + '*')]
    loaded_docs = []
    print('Loading CRAFT corpus ...')
    for doc in tqdm(mp.Pool().imap_unordered(load_craft_document, ids),
                    total=len(ids)):
        loaded_docs.append(doc)

    return loaded_docs


with open('data/GENIA/MEDLINE-to-PMID') as map_file:
    _MEDLINE_TO_PMID = eval(map_file.read())
with open('data/GENIA/genia-quarantine') as quarantine_file:
    _QUARANTINE = eval(quarantine_file.read())


def load_genia_document(doc_id, folder_path='data/GENIA/',
                        only_text=False):
    """Loads in the GENIA document with the given ID and returns it as a
    Document object with annotations."""

    # create xml soup
    xml_file = open(folder_path + 'pos+concepts/' + doc_id + '.xml')
    soup = BeautifulSoup(xml_file.read(), 'xml')

    # doc text is made up of retrievable strings between tags in title+abstract
    doc_text = ''.join(soup.title.strings) + ''.join(soup.abstract.strings)
    # some double line breaks cause trouble in the offsets: correct to single
    # some stray spaces at the beginning or end of lines do the same
    doc_text = doc_text.strip().replace('\n\n', '\n').replace('\n ', '\n')
    doc = Document(doc_id, doc_text)

    if only_text:
        return doc

    sent_offset = 0  # keeps track of how far in we are in the doc text
    # loop over sentences, extract the sentence and tokens + concepts within
    for sent in soup.find_all('sentence'):

        raw_sent = ''.join(sent.strings)  # also strings within tags
        sent_span = (sent_offset, sent_offset + len(raw_sent))
        sentence = Sentence(doc, sent_span)
        doc.add_annotation(sentence)

        # loop over tokens within the sentence, cut off the processed part of
        # the sentence along the way; else, find methods will give incorrect
        # indices for the token spans
        token_offset = 0
        edible_sent = raw_sent  # because it gets eaten through the loop

        for w in sent.find_all('w'):  # tokens are in <w> tags
            raw_token = w.string  # consists of only one string

            spaces = edible_sent.find(raw_token)  # find start index to take
            token_offset += spaces                # spaces into account

            # make the span based on the current offsets
            token_span = (sent_offset + token_offset,
                          sent_offset + token_offset + len(raw_token))

            # chop of the token part of the current sentence
            edible_sent = edible_sent[spaces + len(raw_token):]
            token_offset += len(raw_token)  # update token offset for the next

            # finalise the token annotation and add it
            pos = w['c']
            token = Token(doc, token_span, pos)

            doc.add_annotation(token)

        # loop over concepts within the sentence similar to the tokens loop
        # concepts can be embedded, though, and must be handled differently
        # there's some crazy recursive patterns somewhere, but it works
        edible_sent = raw_sent  # because it gets eaten through the loop
        concept_offset = 0
        last_end = 0  # keeps track of the current chunk of concepts we're in

        for cons in sent.find_all('cons'):

            # make sure to update how far we are in the sentence based on the
            # longest "outer" concept (despite the recursion!)
            cons_string = ''.join(cons.strings)  # can consist of several
            cons_start = edible_sent.find(cons_string)  # find the start index

            # if this concept appears after the previous longest concept,
            # chop off the sentence up until that point
            if cons_start > last_end:
                concept_offset += last_end  # need to remember the offset
                edible_sent = edible_sent[last_end:]  # chop off
                last_end = 0  # we'll update that in a bit

            # keep track of the longest concept in the current chunk
            if len(cons_string) > last_end:
                last_end = cons_start + len(cons_string)

            def resolve_spans(cons_tag: bs4.Tag):
                """Resolves the spans of a given <cons> tag by running
                recursively through them. Due to complex constructions, the
                retrieved span options are flattened in the end."""

                concept_string = ''.join(cons_tag.strings)
                start = edible_sent.find(concept_string)  # find the start index

                # if it doesn't have "sem" attribute, it's not a concept itself
                try:
                    label_ = cons_tag['sem']
                except KeyError:
                    label_ = ''

                if 'AND' in label_ or 'OR' in label_:  # coordinated concepts!
                    children_tags = [c for c in cons_tag.children
                                     if isinstance(c, bs4.Tag)]
                    cc_indexes = []
                    coord_words_indexes = []
                    for i, child in enumerate(children_tags):
                        if (child.name == 'w' and child['c'] == 'CC')\
                                or ''.join(child.stripped_strings) in {',',
                                                                       '/'}:
                            cc_indexes.append(i)
                        else:
                            coord_words_indexes.append(i)
                    coord_words_indexes = [i for i in coord_words_indexes
                                           if (i - 1) in cc_indexes
                                           or (i + 1) in cc_indexes]

                    common_before = [resolve_spans(t)
                                     for t in children_tags[0:cc_indexes[0]-1]]
                    coordinated_words = []
                    for index in coord_words_indexes:
                        coordinated_words += resolve_spans(children_tags[index])
                    common_after = [resolve_spans(t)
                                    for t in children_tags[cc_indexes[-1]+2:]]

                    return [_flatten(list(option))
                            for option in itertools.product(*common_before,
                                                            coordinated_words,
                                                            *common_after)
                            ]

                else:  # single concept
                    # make the span based on offsets and length of concept
                    return [(sent_offset + concept_offset + start,
                             sent_offset + concept_offset + start
                             + len(concept_string))]

            concept_spans = resolve_spans(cons)
            try:
                label = cons['sem']
            except KeyError:
                continue

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
                    if len(merged_spans) == 1:  # might not be discont. at all
                        concept = Concept(doc, merged_spans[0], label)
                    else:
                        concept = DiscontinuousConcept(doc, merged_spans, label)
                else:
                    concept = Concept(doc, concept_span, label)

                doc.add_annotation(concept)

        # update before moving on to the next; remember the line break
        sent_offset += len(raw_sent) + 1

    # treebank annotations are found elsewhere. Handle these similar to before
    # create xml soup
    pmid = _MEDLINE_TO_PMID[doc_id]
    xml_file = open(folder_path + 'treebank/' + pmid + '.xml')
    soup = BeautifulSoup(xml_file.read(), 'xml')

    tokens_stack = doc.get_annotations(Token)
    tokens_stack.reverse()

    for sentence in soup.find_all('sentence'):

        def get_constituents(sub_tree, stack):
            if sub_tree.name == 'tok':
                if not tokens_stack:
                    t = Token(doc, (0, 0), '-NONE-')
                else:
                    t = tokens_stack.pop()

                # handle prefixes and other split token sequences
                while sub_tree.string.replace(' ', '') != \
                        t.get_covered_text().replace(' ', ''):
                    sub_tree_str = sub_tree.string.replace(' ', '')
                    token_str = t.get_covered_text().replace(' ', '')
                    if sub_tree_str == '.':
                        tokens_stack.append(t)
                        empty_span = (t.span[0],
                                      t.span[0])
                        return Token(doc, empty_span, '-NONE-')
                    elif token_str in sub_tree_str:
                        t = t.merge_with(tokens_stack.pop())
                    elif sub_tree_str in token_str:
                        if not token_str.endswith(sub_tree_str):
                            tokens_stack.append(t)
                            empty_span = (t.span[0],
                                          t.span[0])
                            return Token(doc, empty_span, '-NONE-')
                        else:
                            return t

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
                    sub_cons += [get_constituents(nst, stack)]

                if not sub_cons:
                    empty_span = (tokens_stack[-1].span[0],
                                  tokens_stack[-1].span[0])
                    empty_token = Token(doc, empty_span, '-NONE-')
                    sub_cons.append(empty_token)

                const = Constituent(doc, sub_cons, stack.pop())
                doc.add_annotation(const)

                return const

        tree = sentence
        get_constituents(tree, [])  # they are added to the doc in the function

    return doc


def load_genia_corpus(path='data/GENIA/pos+concepts/'):

    ids = [os.path.basename(name[:-4]) for name in glob.glob(path + '*')]
    loaded_docs = []

    # some documents cause trouble; some are handled in the code, but not all
    # these are listed in a quarantine file and will be skipped
    for q in _QUARANTINE:
        ids.remove(str(q))

    print('Loading GENIA corpus ...')
    if _QUARANTINE:
        print(f'Skipping {len(_QUARANTINE)} files put in quarantine.')
    for doc in tqdm(mp.Pool().imap_unordered(load_genia_document, ids),
                    total=len(ids)):
        loaded_docs.append(doc)

    return loaded_docs


# test_craft = load_craft_document('11319941')
# test_craft_corpus = load_craft_corpus()
# test_genia = load_genia_document('95248083')
# test_genia_corpus = load_genia_corpus()
