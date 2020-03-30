import re

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

    def f1_measure(self, beta=1):
        precision = self.precision()
        recall = self.recall()
        return (1 + beta) * precision * recall / (beta * precision + recall)


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
        self._anno_type = annotation_type
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
        print('Summary of CorpusReport for', self._anno_type.__name__)
        print(f'Recall:    {self.recall():.3f}   (highest: {max_recall:.3f}'
              f', lowest: {min_recall:.3f})')
        print(f'Precision: {self.precision():.3f}   (highest: {max_prec:.3f}'
              f', lowest: {min_prec:.3f})')

    def error_analysis(self, gold_concepts, verified_concepts, max_n,
                       pos_filter, gold_counter, freq_threshold):
        gold_concepts = set(gold_concepts)
        if self.false_positives():
            gold_fps = {c for c in self.false_positives()
                        if c.normalized_concept() in gold_concepts}
            percent = len(gold_fps) / len(self.false_positives()) * 100
            print(f'{len(gold_fps)} ({percent:.2f}%) FP\'s occur elsewhere as '
                  f'a gold standard concept.')

            verified_concepts = set(verified_concepts)
            verified_fps = {c for c in self.false_positives()
                            if c.normalized_concept() in verified_concepts}
            percent = len(verified_fps) / len(self.false_positives()) * 100
            print(f'{len(verified_fps)} ({percent:.2f}%) FP\'s were verified.')

        if self.false_negatives():
            over_max = {c for c in self.false_negatives() if len(c) > max_n}
            percent = len(over_max) / len(self.false_negatives()) * 100
            print(f'{len(over_max)} ({percent:.2f}%) FN\'s are above max n.')

            non_captured = {c for c in self.false_negatives()
                            if not re.match(pos_filter, c.pos_sequence())}
            percent = len(non_captured) / len(self.false_negatives()) * 100
            print(f'{len(non_captured)} ({percent:.2f}%) FN\'s cannot be '
                  'captured by the used POS-tag filter.')

            below_threshold = {c for c in self.false_negatives()
                               if gold_counter[c.normalized_concept()]
                               < freq_threshold}
            percent = len(below_threshold) / len(self.false_negatives()) * 100
            print(f'{len(below_threshold)} ({percent:.2f}%) FN\'s occur less '
                  'often than the frequency threshold.')


class TypesReport(EvaluationReport):

    def __init__(self, ranked_final, gold_concepts):
        self.ranked_final = ranked_final
        super().__init__(set(ranked_final), gold_concepts)

    def precision_at_k(self, k_values=(100, 200, 500, 1000, 5000)):
        print('Precision at k:')
        for k in k_values:
            if k > len(self.ranked_final):
                k = len(self.ranked_final)
            predicted_at_k = set(self.ranked_final[:k])
            prec_at_k = len(predicted_at_k.intersection(self.expected)) / k
            print(f'\tP@{k:<6}', round(prec_at_k, 3))
            if k == len(self.ranked_final):
                break

    def performance_summary(self, k=(100, 200, 500, 1000, 5000)):
        print('Summary of TypesReport')
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1_measure()
        print('Precision:  ', round(precision, 3))
        print('Recall:     ', round(recall, 3))
        print('F1-measure: ', round(f1, 3))
        self.precision_at_k(k)

    def error_analysis(self, verified_concepts, max_n, gold_counter,
                       freq_threshold):
        verified_concepts = set(verified_concepts)
        verified_fps = verified_concepts.intersection(self.false_positives())
        percent = len(verified_fps) / len(self.false_positives()) * 100
        print(f'{len(verified_fps)} ({percent:.2f}%) of FP\'s were verified.')

        over_max = {c for c in self.false_negatives() if len(c) > max_n}
        percent = len(over_max) / len(self.false_negatives()) * 100
        print(f'{len(over_max)} ({percent:.2f}%) of FN\'s are above max n.')

        below_threshold = {c for c in self.false_negatives()
                           if gold_counter[c] < freq_threshold}
        percent = len(below_threshold) / len(self.false_negatives()) * 100
        print(f'{len(below_threshold)} ({percent:.2f}%) FN\'s occur less '
              'often than the frequency threshold.')


# PERFORMANCE MEASURES
def gold_standard_concepts(corpus, continuous=True, discontinuous=True,
                           exclude_unigrams=True):
    print('Retrieving gold standard concepts ...', end=' ', flush=True)
    all_concepts = set()
    skipped = set()
    for doc in corpus:
        concepts = doc.get_annotations(anno.Concept)
        for c in concepts:
            if not continuous and not isinstance(c, anno.DiscontinuousConcept):
                continue  # skip non-DiscontinuousConcept if not allowed
            elif not discontinuous and isinstance(c, anno.DiscontinuousConcept):
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
    if exclude_unigrams:
        all_concepts = {c for c in all_concepts if not len(c) == 1}
    return all_concepts


