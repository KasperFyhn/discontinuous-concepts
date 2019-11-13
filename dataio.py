from bs4 import BeautifulSoup
import os


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


def load_craft_document(doc_id, folder_path='data/CRAFT/txt/',
                        annotation_types=('structural', 'concept')):

    doc = Document.from_file(folder_path + doc_id + '.txt')

    if 'structural' in annotation_types:
        # make beautiful soup from xml
        path_to_xml = 'data/CRAFT/part-of-speech/' + doc_id + '.xml'
        raw_xml = open(path_to_xml).read()
        pos_bs = BeautifulSoup(raw_xml, 'xml')

        # add sentence and token annotations
        sentences = []
        tokens = []
        for tag in pos_bs.find_all('annotation'):
            span = (int(tag.span['start']), int(tag.span['end']))
            pos_tag = tag.find('class')['label']
            if pos_tag == 'sentence':
                sentence = Sentence(doc, span)
                sentences.append(sentence)
            else:
                token = Token(doc, span, pos_tag)
                tokens.append(token)

        doc.add_annotations('Sentence', sentences)
        doc.add_annotations('Token', tokens)

    if 'concept' in annotation_types:
        # make beautiful soup from xml
        path_to_xml = 'data/CRAFT/concepts/' + doc_id + '.xml'
        raw_xml = open(path_to_xml).read()
        concept_bs = BeautifulSoup(raw_xml, 'xml')

        # add sentence and token annotations
        concepts = []
        for tag in concept_bs.find_all('annotation'):
            spans = tag.find_all('span')
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

        doc.add_annotations('Concept', concepts)

    return doc


def load_genia_document(doc_id, folder_path='data/GENIA/pos+concepts/',
                        annotation_types=('structural', 'concept')):

    xml_file = open(folder_path + doc_id + '.xml')
    soup = BeautifulSoup(xml_file.read(), 'xml')

    doc_text = ''.join(soup.title.strings) + ''.join(soup.abstract.strings)
    doc_text = doc_text.strip().replace('\n\n', '\n')
    doc = Document(doc_id, doc_text)

    if 'structural' in annotation_types:
        sent_offset = 0
        sentences = []
        tokens = []
        for sent in soup.find_all('sentence'):
            raw_sent = ''.join(sent.strings)
            sent_span = (sent_offset, sent_offset + len(raw_sent))
            sentence = Sentence(doc, sent_span)
            sentences.append(sentence)
            extra_offset = len(raw_sent) + 1

            token_offset = 0
            for w in sent.find_all('w'):
                raw_token = w.string
                spaces = raw_sent.find(raw_token)
                token_offset += spaces
                token_span = (sent_offset + token_offset,
                        sent_offset + token_offset + len(raw_token))
                raw_sent = raw_sent[spaces + len(raw_token):]
                token_offset += len(raw_token)
                pos = w['c']
                token = Token(doc, token_span, pos)
                tokens.append(token)

            sent_offset += extra_offset

        doc.add_annotations('Sentence', sentences)
        doc.add_annotations('Token', tokens)




    return doc


test = load_genia_document('90110496')




