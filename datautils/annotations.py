import os
import re
from colorama import Style, Back, Fore
import sys
from collections import defaultdict
import nltk

_COLOR_BEGIN = Back.CYAN + Fore.BLACK
_COLOR_END = Style.RESET_ALL


class Document:
    """A representation of a document which contains raw text and annotations"""

    def __init__(self, doc_id: str, raw_text: str):
        self.id = doc_id
        self._text = raw_text
        self._annotations = defaultdict(set)
        self._span_starts = defaultdict(set)
        self._span_ends = defaultdict(set)

    def load_annotations_from_file(self, path_in):
        """Load saved annotations to this Document from the given path.
        (So far only supports: Annotation, Sentence, Token)"""
        with open(path_in) as file_in:
            lines = file_in.read().split('\n')[:-1]
        for line in lines:
            try:
                anno = Annotation.from_short_repr(self, line)
                self.add_annotation(anno)
            except Exception as e:
                print('An annotation was not loaded! The error was caused by:')
                print(line)
                print(e)
                continue

    def save_annotations_to_file(self, path_to_folder):
        """Save annotations in this Document to a file: [doc_id].anno.
        (So far only supports: Annotation, Sentence, Token)"""

        with open(path_to_folder + self.id + '.anno', 'w+') as out_file:
            for annotation in self.get_annotations(Annotation):
                if type(annotation) in {Annotation, Sentence, Token}:
                    print(annotation.short_repr(), file=out_file)

    def add_annotation(self, annotation):

        for type_name in annotation.__class__.__mro__:
            key = type_name.__name__
            if key == 'object':
                continue
            self._annotations[key].add(annotation)

        span_start = annotation.span[0]
        self._span_starts[span_start].add(annotation)
        span_end = annotation.span[1]
        self._span_ends[span_end].add(annotation)

    def remove_annotation(self, annotation):
        for type_name in annotation.__class__.__mro__:
            key = type_name.__name__
            if key == 'object':
                continue
            self._annotations[key].remove(annotation)

        span_start = annotation.span[0]
        self._span_starts[span_start].remove(annotation)
        span_end = annotation.span[1]
        self._span_ends[span_end].remove(annotation)

    def get_annotations_at(self, span, annotation_type=None):
        # TODO: does not retrieve e.g. a Sentence which spans out over both ends

        if not annotation_type:
            annotation_type = 'Annotation'

        if isinstance(annotation_type, str):
            try:
                annotation_type = getattr(sys.modules[__name__],
                                          annotation_type)
            except AttributeError:
                print('No annotation type of that name!')
                return []

        if annotation_type == Constituent:

            start = span[0]
            end = span[1]
            for i in range(start, -1, -1):
                for annotation in self._span_starts[i]:
                    if (isinstance(annotation, Constituent)
                            and annotation.span[1] >= end):
                        return [annotation]

        else:
            return sorted(
                {a for li in [self._span_starts[index]
                              for index in range(span[0], span[1])]
                             + [self._span_ends[index]
                                for index in range(span[0]+1, span[1]+1)]
                 for a in li if isinstance(a, annotation_type)},
                key=lambda x: x.span
            )

    def get_annotations(self, annotation_type):

        if not isinstance(annotation_type, str):
            annotation_type = annotation_type.__name__

        return sorted(self._annotations[annotation_type], key=lambda x: x.span)

    def get_text(self):
        return self._text

    def __eq__(self, other):
        return self.id == other.id and self.get_text() == other.get_text()

    def __hash__(self):
        return hash((self.id, self._text))

    def __str__(self):
        return self._text

    def __repr__(self):
        return f'Document({self.id}: "{self._text[:30]}...")'


class Annotation:
    """Basic annotation class. Contains only reference to a document and the
    span of the annotation."""

    def __init__(self, document: Document, span: tuple):
        self.document = document
        self._doc_id = document.id
        self.span = span

    def get_covered_text(self):
        """Returns the text covered by the annotation."""

        start = self.span[0]
        end = self.span[1]
        return self.document.get_text()[start:end]

    def get_tokens(self):
        return self.document.get_annotations_at(self.span, Token)

    def pos_sequence(self):
        return ''.join(t.mapped_pos() for t in self. get_tokens())

    def __len__(self):
        return len(self.get_tokens())

    def get_context(self, char_window=20):
        start = self.span[0] - char_window
        if start < 0:
            start = 0
        end = self.span[1] + char_window
        if end > len(self.document.get_text()):
            end = len(self.document.get_text())
        build_string = self.document.get_text()[start:self.span[0]]\
                       + _COLOR_BEGIN + self.get_covered_text() + _COLOR_END \
                       + self.document.get_text()[self.span[1]:end]
        build_string = build_string.replace('\n', ' ')
        return build_string

    def merge_with(self, another_annotation):
        if self.document is not another_annotation.document:
            raise ValueError(
                'Cannot merge annotations from different documents!'
            )
        doc = self.document
        doc.remove_annotation(self)
        doc.remove_annotation(another_annotation)
        another_annotation.span = (self.span[0], another_annotation.span[1])
        doc.add_annotation(another_annotation)
        return another_annotation

    def __eq__(self, other):
        return type(self) == type(other) and self.document == other.document\
               and self.span == other.span

    def __hash__(self):
        return hash((type(self), self._doc_id, self.span))

    def __repr__(self):
        return self.__class__.__name__ + "('" + self.get_covered_text() + "'"\
               + str(self.span) + ')'

    def short_repr(self):
        return self.__class__.__name__ + ':' + str([self.span])

    @staticmethod
    def from_short_repr(doc, short_repr: str):
        first_colon = short_repr.find(':')
        anno_class = getattr(sys.modules[__name__], short_repr[:first_colon])
        args = eval(short_repr[first_colon+1:])
        return anno_class(doc, *args)


class Sentence(Annotation):
    pass


class PosTagMap(dict):
    def __getitem__(self, item):
        if item in self.keys():
            return super(PosTagMap, self).__getitem__(item)
        else:
            return item[0]


POS_TAG_MAP = PosTagMap()
POS_TAG_MAP.update({
    'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',  # nouns
    'JJ': 'a', 'JJR': 'a', 'JJS': 'a',  # adjectives
    'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v',  # verbs
    'VBZ': 'v',  # verbs
    'RB': 'r', 'RBS': 'r', 'RBR': 'r',  # adv's
    'CD': 'd',  # cardinal digit
    'CC': 'c',  # coordinating conjunction
    'POS': 's',  # possessive marker
    'IN': 'p',  # preposition
    'DT': 't',  # determiner
    'HYPH': '-',  # hyphen
    ',': ',', '.': '.',  # punctuation
    '*': 'f', 'AFX': 'f',  # affix/part of compound word
    '-NONE-': 'Ø', '': 'Ø'  # null element
})
LEMMATIZER = nltk.WordNetLemmatizer()


class Token(Annotation):

    def __init__(self, document: Document, span: tuple, pos_tag: str):
        super().__init__(document, span)
        self.pos = pos_tag
        self._lemma = None

    def mapped_pos(self):
        return POS_TAG_MAP[self.pos]

    def lemma(self):
        if not self._lemma:
            normalized = self.get_covered_text().lower()
            pos = self.mapped_pos()
            lemma = LEMMATIZER.lemmatize(normalized, pos) if pos in 'anvr'\
                else LEMMATIZER.lemmatize(normalized)
            self._lemma = lemma
        return self._lemma

    def tokens_between(self, other_token, ignore_pos=''):
        tokens_between = self.document.get_annotations_at(
            (self.span[1], other_token.span[0]), Token)
        tokens_between = [t for t in tokens_between
                          if t.mapped_pos() not in ignore_pos]
        return len(tokens_between)

    def __eq__(self, other):
        return super().__eq__(other) and self.pos == other.pos

    def __hash__(self):
        return hash((type(self), self._doc_id, self.span, self.pos))

    def __repr__(self):
        return super().__repr__() + '/' + self.pos

    def short_repr(self):
        return self.__class__.__name__ + ':' + str([self.span, self.pos])


class Concept(Annotation):

    def __init__(self, document: Document, span: tuple, label: str = ''):
        super().__init__(document, span)
        self.label = label

    def normalized_concept(self):
        tokens = self.get_tokens()
        return tuple(t.lemma() for t in tokens)


class DiscontinuousConcept(Concept):

    def __init__(self, document: Document, spans: list, label: str = ''):
        full_span = (spans[0][0], spans[-1][1])
        super().__init__(document, full_span, label)
        self.spans = spans

    def get_tokens(self):
        return [t for span in self.spans
                for t in self.document.get_annotations_at(span, Token)]

    def get_spanned_tokens(self):
        return super().get_tokens()

    def get_bridges(self):
        bridges = []
        for i in range(len(self.spans) - 1):
            span1 = self.spans[i]
            span2 = self.spans[i + 1]
            span1_tokens = self.document.get_annotations_at(span1, Token)
            span2_tokens = self.document.get_annotations_at(span2, Token)
            bridge = (span1_tokens[-1], span2_tokens[0])
            bridges.append(bridge)
        return bridges

    def contains_illegal_bridges(self):
        for i in range(len(self.spans) - 1):
            span1 = self.spans[i]
            span2 = self.spans[i + 1]
            gap = (span1[1], span2[0])
            span1_tokens = {t.lemma() for t
                            in self.document.get_annotations_at(span1, Token)}
            gap_tokens = {t.lemma() for t
                          in self.document.get_annotations_at(gap, Token)}
            span2_tokens = {t.lemma()for t
                            in self.document.get_annotations_at(span2, Token)}
            if span1_tokens.issubset(gap_tokens)\
                    or span2_tokens.issubset(gap_tokens):
                return True
        return False  # no illegal bridges were found

    def get_covered_text(self):
        """Returns only the concept, disregarding tokens within the full span
        that are not part of the concept."""
        return ' '.join(self.document.get_text()[s[0]:s[1]] for s in self.spans)

    def get_spanned_text(self):
        return super().get_covered_text()

    def get_context(self, char_window=20):
        start = self.span[0] - char_window
        if start < 0:
            start = 0
        end = self.span[1] + char_window
        if end > len(self.document.get_text()):
            end = len(self.document.get_text())
        span1 = self.spans[0]
        marked_text = _COLOR_BEGIN\
                      + self.document.get_text()[span1[0]:span1[1]]\
                      + _COLOR_END
        for i in range(1, len(self.spans)):
            prev_span = self.spans[i-1]
            span = self.spans[i]
            marked_text += self.document.get_text()[prev_span[1]:span[0]]
            marked_text += _COLOR_BEGIN\
                           + self.document.get_text()[span[0]:span[1]]\
                           + _COLOR_END
        build_string = self.document.get_text()[start:self.span[0]]\
                       + marked_text\
                       + self.document.get_text()[self.span[1]:end]
        build_string = build_string.replace('\n', ' ')
        return build_string

    def __eq__(self, other):
        return super().__eq__(other) and self.spans == other.spans

    def __hash__(self):
        return hash((type(self), self._doc_id, self.span,
                     tuple(self.spans)))

    def __repr__(self):
        return self.__class__.__name__ + "('" + self.get_covered_text() + "'"\
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

    def structure(self):
        return '(' + self.label + ' ' + ' '.join(
            c.structure() if isinstance(c, Constituent) else c.mapped_pos()
            for c in self.constituents) + ')'
