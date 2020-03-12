from datautils import annotations as anno


class EvaluationReport:

    def __init__(self, predicted, expected):
        self.predicted = set(predicted)
        self.expected = set(expected)
        self.ok = self.predicted.intersection(self.expected)

    def precision(self):
        try:
            return len(self.ok) / len(self.predicted)
        except ZeroDivisionError:
            return 0

    def false_positives(self):
        return self.predicted.difference(self.ok)

    def recall(self):
        try:
            return len(self.ok) / len(self.expected)
        except ZeroDivisionError:
            return 0

    def false_negatives(self):
        return self.expected.difference(self.ok)


class DocumentReport(EvaluationReport):

    def __init__(self, annotation_type: type, predicted_doc: anno.Document,
                 expected_doc: anno.Document):
        predicted = predicted_doc.get_annotations(annotation_type)
        expected = expected_doc.get_annotations(annotation_type)
        super().__init__(predicted, expected)
        self.predicted_doc = predicted_doc
        self.expected_doc = expected_doc


class CorpusReport(EvaluationReport):

    def __init__(self, annotation_type: type, predicted_docs, expected_docs):
        if isinstance(predicted_docs, list) or isinstance(expected_docs, list):
            predicted_docs = {doc.id: doc for doc in predicted_docs}
            expected_docs = {doc.id: doc for doc in expected_docs}
        self.doc_reports = {
            id_: DocumentReport(annotation_type, predicted_docs[id_],
                                expected_docs[id_])
            for id_ in predicted_docs}
        predicted = {a for dr in self.doc_reports.values()
                     for a in dr.predicted}
        expected = {a for dr in self.doc_reports.values()
                    for a in dr.expected}
        super().__init__(predicted, expected)

    def precision_values(self):
        return [dr.precision() for dr in self.doc_reports.values()]

    def recall_values(self):
        return [dr.recall() for dr in self.doc_reports.values()]

    def performance_summary(self):
        recall_values = self.recall_values()
        max_recall = max(recall_values)
        min_recall = min(recall_values)
        prec_values = self.precision_values()
        max_prec = max(prec_values)
        min_prec = min(prec_values)
        print('Summary of CorpusReport')
        print(f'Recall:    {self.recall():.3f}   (highest: {max_recall:.3f}'
              f', lowest: {min_recall:.3f})')
        print(f'Precision: {self.precision():.3f}   (highest: {max_prec:.3f}'
              f', lowest: {min_prec:.3f})')



class TypesReport(EvaluationReport):

    pass


# PERFORMANCE MEASURES
def gold_standard_concepts(corpus, discontinuous=True):
    print('Retrieving gold standard concepts ...', end=' ', flush=True)
    all_concepts = set()
    skipped = set()
    for doc in corpus:
        concepts = doc.get_annotations(anno.Concept)
        for c in concepts:
            if not discontinuous and isinstance(c, anno.DiscontinuousConcept):
                continue  # skip DiscontinuousConcept if not allowed
            else:
                c_tokens = c.get_tokens()
                if len(c_tokens) == 0 or \
                        not (c.span[0] == c_tokens[0].span[0]
                             and c.span[1] == c_tokens[-1].span[1]):
                    # concept span does not equal token span, e.g. if only
                    # part of a token constitutes a concept
                    skipped.add(c.get_covered_text())
                else:
                    all_concepts.add(c.normalized_concept())
    print(f'Skipped {len(skipped)} concepts not bounded at tokens boundaries.')
    return all_concepts


def recall_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(exp)


def precision_types(predicted, expected):
    pred = set(predicted)
    exp = set(expected)
    return len(pred.intersection(exp)) / len(pred)


def precision_at_k(predicted_ranked: list, expected: set, k_values):
    for k in k_values:
        if k > len(predicted_ranked):
            k = len(predicted_ranked)
        predicted_at_k = set(predicted_ranked[:k])
        prec_at_k = len(predicted_at_k.intersection(expected)) / k
        print(f'P@{k:<6}', prec_at_k)
        if k == len(predicted_ranked):
            break


def f1_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def performance(predicted, expected):
    precision = precision_types(predicted, expected)
    recall = recall_types(predicted, expected)
    f1 = f1_measure(precision, recall)
    print('Precision:  ', round(precision, 3))
    print('Recall:     ', round(recall, 3))
    print('F1-measure: ', round(f1, 3))
