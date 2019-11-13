import glob
import itertools
from bs4 import BeautifulSoup, Tag
import os
import multiprocessing as mp
from tqdm import tqdm


class Document:
    """A representation of a document which contains raw text and annotations"""

    def __init__(self, doc_id: str, raw_text: str):
        self.id = doc_id
        self._text = raw_text
        self._annotations = {}

    @classmethod
    def from_file(cls, path):
        """Load a document from a path."""

        try:
            with open(path) as f:
                text_id = os.path.basename(path)[:-4]
                return cls(text_id, f.read())
        except UnicodeDecodeError as e:
            print('Something went wrong in decoding', os.path.basename(path))
            print(type(e), e)

    def add_annotations(self, annotation_type, annotations: list):
        self._annotations[annotation_type] = annotations

    def get_annotations(self, annotation_type):
        try:
            return self._annotations[annotation_type]
        except KeyError:
            print('There are no annotations of type "' + annotation_type
                  + '" in document', self.id)

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
            + '    ' + self.get_covered_text() + '    '\
            + self.document.get_text()[self.span[1]:end]
        return build_string

    def __repr__(self):
        return "'" + self.get_covered_text() + "'" + str(self.span)


class Sentence(Annotation):
    pass


class Token(Annotation):

    def __init__(self, document: Document, span: tuple, pos_tag: str):
        super().__init__(document, span)
        self.pos = pos_tag

    def __repr__(self):
        return super().__repr__() + '/' + self.pos


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
        return "'" + self.get_concept() + "'" + str(self.spans)


def load_craft_document(doc_id, folder_path='data/CRAFT/txt/', only_text=False):

    doc = Document.from_file(folder_path + doc_id + '.txt')

    if only_text:
        return doc

    # add sentence and token annotations; sents are recognized by their labels
    # make soup from POS XML
    path_to_xml = 'data/CRAFT/part-of-speech/' + doc_id + '.xml'
    raw_xml = open(path_to_xml).read()
    pos_bs = BeautifulSoup(raw_xml, 'xml')

    sentences = []
    tokens = []
    # just run over all annotations and get the relevant info
    for tag in pos_bs.find_all('annotation'):
        span = (int(tag.span['start']), int(tag.span['end']))
        pos_tag = tag.find('class')['label']
        if pos_tag == 'sentence':
            sentence = Sentence(doc, span)
            sentences.append(sentence)
        else:
            token = Token(doc, span, pos_tag)
            tokens.append(token)

    # make beautiful soup from concept XML and add concept annotations
    path_to_xml = 'data/CRAFT/concepts/' + doc_id + '.xml'
    raw_xml = open(path_to_xml).read()
    concept_bs = BeautifulSoup(raw_xml, 'xml')

    concepts = []
    # just run over all annotations and get the relevant info
    for tag in concept_bs.find_all('annotation'):
        spans = tag.find_all('span')

        # if the concept is discontinuous, handle it differently: has more spans
        if len(spans) > 1:
            spans = [(int(span['start']), int(span['end']))
                     for span in spans]
            label = tag.find('class')['label']
            concept = DiscontinuousConcept(doc, spans, label)
            concepts.append(concept)
        else:
            span = (int(tag.span['start']), int(tag.span['end']))
            label = tag.find('class')['label']
            concept = Concept(doc, span, label)
            concepts.append(concept)

    doc.add_annotations('Sentence', sentences)
    doc.add_annotations('Token', tokens)
    doc.add_annotations('Concept', concepts)

    return doc


def load_craft_corpus(path='./data/CRAFT/txt/'):

    ids = [os.path.basename(name[:-4]) for name in glob.glob(path + '*')]
    loaded_docs = []
    print('Loading CRAFT corpus ...')
    for doc in tqdm(mp.Pool().imap_unordered(load_craft_document, ids),
                    total=len(ids)):
        loaded_docs.append(doc)

    return loaded_docs


def load_genia_document(doc_id, folder_path='data/GENIA/pos+concepts/',
                        only_text=False):
    """Load a GENIA document from XML."""

    # create xml soup
    xml_file = open(folder_path + doc_id + '.xml')
    soup = BeautifulSoup(xml_file.read(), 'xml')

    # doc text is made up of retrievable strings between tags in title+abstract
    doc_text = ''.join(soup.title.strings) + ''.join(soup.abstract.strings)
    # some double line breaks cause trouble in the offsets: correct to single
    doc_text = doc_text.strip().replace('\n\n', '\n')
    doc = Document(doc_id, doc_text)

    if only_text:
        return doc

    sent_offset = 0  # keeps track of how far in we are in the doc text
    sentences = []
    tokens = []
    concepts = []
    # loop over sentences, extract the sentence and tokens + concepts within
    for sent in soup.find_all('sentence'):

        raw_sent = ''.join(sent.strings)  # also strings within tags
        sent_span = (sent_offset, sent_offset + len(raw_sent))
        sentence = Sentence(doc, sent_span)
        sentences.append(sentence)

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
            tokens.append(token)

        # loop over concepts within the sentence similar to the tokens loop
        # concepts can be embedded, though, and must be handled differently
        # there's some crazy recursive patterns somewhere, but it works
        edible_sent = raw_sent  # because it gets eaten through the loop
        concept_offset = 0
        last_end = 0  # keeps track of the current chunk of concepts we're in

        for cons in sent.find_all('cons'):

            def resolve_spans(cons_tag: Tag):
                """Resolves the spans of a given <cons> tag by running
                recursively through them. Due to complex constructions, the
                retrieved span options are flattened in the end."""

                concept_string = ''.join(cons_tag.strings)  # can consist of several
                start = edible_sent.find(concept_string)  # find the start index

                # if it doesn't have "sem" attribute, it's not a concept itself
                try:
                    label_ = cons_tag['sem']
                except KeyError:
                    label_ = ''

                if 'AND' in label_ or 'OR' in label_:  # coordinated concepts!
                    children_tags = [c for c in cons_tag.children
                                     if isinstance(c, Tag)]
                    cc_index = -1
                    for i, child in enumerate(children_tags):
                        if ''.join(child.stripped_strings) in {'and', 'or'}:
                            cc_index = i
                            break

                    common_before = [resolve_spans(t)
                                     for t in children_tags[:cc_index-1]]
                    first = resolve_spans(children_tags[cc_index-1])
                    second = resolve_spans(children_tags[cc_index+1])
                    common_after = [resolve_spans(t)
                                    for t in children_tags[cc_index+2:]]

                    def flatten(li):
                        return sum(([x] if not isinstance(x, list)
                                    else flatten(x) for x in li), [])

                    return [flatten(list(option))
                            for option in itertools.product(*common_before,
                                                            first + second,
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

                concepts.append(concept)

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

        # update before moving on to the next; remember the line break
        sent_offset += len(raw_sent) + 1

        # add all annotations
        doc.add_annotations('Sentence', sentences)
        doc.add_annotations('Token', tokens)
        doc.add_annotations('Concept', concepts)

    return doc


def load_genia_corpus(path='data/GENIA/pos+concepts/'):

    ids = [os.path.basename(name[:-4]) for name in glob.glob(path + '*')]
    loaded_docs = []
    print('Loading GENIA corpus ...')
    for doc in tqdm(mp.Pool().imap_unordered(load_genia_document, ids),
                    total=len(ids)):
        loaded_docs.append(doc)

    return loaded_docs


genia = load_genia_corpus()
craft = load_craft_corpus()
