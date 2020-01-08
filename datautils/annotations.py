import sys
from collections import defaultdict


class Document:
    """A representation of a document which contains raw text and annotations.py"""

    def __init__(self, doc_id: str, raw_text: str):
        self.id = doc_id
        self._text = raw_text
        self._annotations = defaultdict(list)
        self._span_starts = defaultdict(list)
        self._span_ends = defaultdict(list)

    @classmethod
    def from_pickle(cls, path_in):
        """Load a pickled Document object from a path."""

        pass

    def pickle(self, path_out):

        pass

    def add_annotation(self, annotation):

        for type_name in annotation.__class__.__mro__:
            key = type_name.__name__
            if key == 'object':
                continue
            self._annotations[key].append(annotation)

        span_start = annotation.span[0]
        self._span_starts[span_start].append(annotation)
        span_end = annotation.span[1]
        self._span_starts[span_end].append(annotation)

    def get_annotations_at(self, span, annotation_type=None):
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
            max_index = len(self._text)
            potential = {a for l in [self._span_starts[index]
                                     for index in range(span[0]+1)]
                                    + [self._span_ends[index]
                                       for index in range(span[1], max_index)]
                         for a in l if isinstance(a, annotation_type)
                         and a.span[0] <= span[0] and span[1] <= a.span[1]}
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

        else:
            return sorted(
                {a for l in [self._span_starts[index]
                             for index in range(span[0], span[1])]
                            + [self._span_ends[index]
                               for index in range(span[0], span[1])]
                 for a in l if isinstance(a, annotation_type)},
                key=lambda x: x.span
            )

    def get_annotations(self, annotation_type):

        if not isinstance(annotation_type, str):
            annotation_type = annotation_type.__name__

        return sorted(self._annotations[annotation_type], key=lambda x: x.span)

    def get_text(self):
        return self._text

    def __str__(self):
        return self._text

    def __repr__(self):
        return f'Document({self.id})'


class Annotation:
    """Basic annotation class. Contains only reference to a document and the
    span of the annotation."""

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