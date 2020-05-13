import re

from datautils import annotations as anno


class EvaluationReport:

    def __init__(self, predicted, expected):
        self.predicted = set(predicted)
        self.expected = set(expected)
        self.ok = self.predicted.intersection(self.expected)
        self._alternatively_validated = set()
        self._ignored_fns = set()

    def true_positives(self):
        return self.ok

    def false_positives(self):
        return self.predicted.difference(self.ok)

    def false_negatives(self):
        return self.expected.difference(self.ok)

    def precision(self):
        try:
            return len(self.ok) / len(self.predicted)
        except ZeroDivisionError:
            return 0

    def recall(self):
        try:
            return len(self.ok) / len(self.expected)
        except ZeroDivisionError:
            return 0

    def f1_measure(self, beta=1):
        try:
            precision = self.precision()
            recall = self.recall()
            return (1 + beta) * precision * recall / (beta * precision + recall)
        except ZeroDivisionError:
            return 0

    def corrected_precision(self):
        ok = self.predicted.intersection(
            self.expected.union(self._alternatively_validated)
        )
        try:
            return len(ok) / len(self.predicted)
        except ZeroDivisionError:
            return 0

    def actual_false_positives(self):
        try:
            return self.predicted.difference(
                self.expected.union(self._alternatively_validated))
        except ZeroDivisionError:
            return 0

    def corrected_recall(self):
        considered = self.expected.difference(self._ignored_fns)
        ok = self.predicted.intersection(considered)
        try:
            return len(ok) / len(considered)
        except ZeroDivisionError:
            return 0

    def actual_false_negatives(self):
        return self.expected.difference(self.predicted.union(self._ignored_fns))


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
        print('# extracted concepts:', len(self.predicted))
        print(f'Precision: {self.precision():.3f}   (highest: {max_prec:.3f}'
              f', lowest: {min_prec:.3f})')
        print(f'Recall:    {self.recall():.3f}   (highest: {max_recall:.3f}'
              f', lowest: {min_recall:.3f})')

    def error_analysis(self, gold_concepts, verified_concepts, max_n,
                       pos_filter, gold_counter, freq_threshold):
        print('Error analysis of CorpusReport for', self._anno_type.__name__)
        if self.false_positives():
            # some concepts are not annotated, but occur elsewhere
            gold_concepts = set(gold_concepts)
            gold_fps = {c for c in self.false_positives()
                        if c.normalized_concept() in gold_concepts}
            percent = len(gold_fps) / len(self.false_positives()) * 100
            print(f'{len(gold_fps)} ({percent:.2f}%) FP\'s occur elsewhere as '
                  f'a gold standard concept.')

            # some concepts can be verified from other sources, e.g. ontologies
            verified_concepts = set(verified_concepts)
            verified_fps = {c for c in self.false_positives()
                            if c.normalized_concept() in verified_concepts}
            percent = len(verified_fps) / len(self.false_positives()) * 100
            print(f'{len(verified_fps)} ({percent:.2f}%) FP\'s were verified '
                  'in ontology source(s).')

            # all analyzed FP's
            analyzed_fps = set.union(gold_fps, verified_fps)
            self._alternatively_validated.update(analyzed_fps)
            percent = len(analyzed_fps) / len(self.false_positives()) * 100
            print(f'{len(analyzed_fps)} ({percent:.2f}%) FP\'s were accounted '
                  'for in this analysis.')

            print('Corrected precision:', round(self.corrected_precision(), 3))

        if self.false_negatives():
            # some concepts cannot be captured if they are longer than max n
            over_max = {c for c in self.false_negatives() if len(c) > max_n}
            percent = len(over_max) / len(self.false_negatives()) * 100
            print(f'{len(over_max)} ({percent:.2f}%) FN\'s are above max n.')

            # some concepts are not captured by the POS-filter(s)
            non_captured = {c for c in self.false_negatives()
                            if not re.match(pos_filter, c.pos_sequence())}
            percent = len(non_captured) / len(self.false_negatives()) * 100
            print(f'{len(non_captured)} ({percent:.2f}%) FN\'s cannot be '
                  'captured by the used POS-tag filter.')

            # many low-frequency concepts will be filtered out
            below_threshold = {
                c for c in self.false_negatives()
                if 0 < gold_counter[c.normalized_concept()] < freq_threshold
            }
            percent = len(below_threshold) / len(self.false_negatives()) * 100
            print(f'{len(below_threshold)} ({percent:.2f}%) FN\'s occur less '
                  'often than the frequency threshold.')

            # all analyzed FN's
            analyzed_fns = set.union(over_max, non_captured, below_threshold)
            percent = len(analyzed_fns) / len(self.false_negatives()) * 100
            print(f'{len(analyzed_fns)} ({percent:.2f}%) FN\'s were accounted '
                  'for in this analysis.')
            self._ignored_fns.update(analyzed_fns)

            print('Corrected recall:', round(self.corrected_recall(), 3))


class TypesReport(EvaluationReport):

    def __init__(self, ranked_final, gold_concepts):
        self.ranked_final = ranked_final
        super().__init__(set(ranked_final), gold_concepts)

    def precision_at_k(self, k_values=(100, 200, 500, 1000, 5000)):
        print('Precision at k:')
        for k in k_values:
            if k > len(self.ranked_final):
                k = len(self.ranked_final)
                if k == 0:
                    print('No annotations!')
                    break
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
        print('# types     ', len(self.predicted))
        print('Precision:  ', round(precision, 3))
        print('Recall:     ', round(recall, 3))
        print('F1-measure: ', round(f1, 3))
        self.precision_at_k(k)

    def error_analysis(self, verified, max_n, gold_counter,
                       freq_threshold):
        print('Error analysis of TypesReport')
        if self.false_positives():
            # some concepts can be verified from other sources, e.g. ontologies
            verified = set(verified)
            verified_fps = verified.intersection(self.false_positives())
            percent = len(verified_fps) / len(self.false_positives()) * 100
            print(f'{len(verified_fps)} ({percent:.2f}%) FP\'s were verified.')

            # all analyzed FP's
            analyzed_fps = set.union(verified_fps)
            percent = len(analyzed_fps) / len(self.false_positives()) * 100
            print(f'{len(analyzed_fps)} ({percent:.2f}%) FP\'s were accounted '
                  'for in this analysis.')
            self._alternatively_validated.update(analyzed_fps)

            print('Corrected precision:', round(self.corrected_precision(), 3))

        if self.false_negatives():
            # some concepts cannot be captured if they are longer than max n
            over_max = {c for c in self.false_negatives() if len(c) > max_n}
            percent = len(over_max) / len(self.false_negatives()) * 100
            print(f'{len(over_max)} ({percent:.2f}%) FN\'s are above max n.')

            # many low-frequency concepts will be filtered out
            below_threshold = {c for c in self.false_negatives()
                               if 0 < gold_counter[c] < freq_threshold}
            percent = len(below_threshold) / len(self.false_negatives()) * 100
            print(f'{len(below_threshold)} ({percent:.2f}%) FN\'s occur less '
                  'often than the frequency threshold.')

            # all analyzed FN's
            analyzed_fns = set.union(over_max, below_threshold)
            percent = len(analyzed_fns) / len(self.false_negatives()) * 100
            print(f'{len(analyzed_fns)} ({percent:.2f}%) FN\'s were accounted '
                  'for in this analysis.')
            self._ignored_fns.update(analyzed_fns)

            print('Corrected recall:', round(self.corrected_recall(), 3))


# PERFORMANCE MEASURES
def gold_standard_concepts(corpus, continuous=True, discontinuous=True,
                           exclude_unigrams=True, pos_filter=None):
    print('Retrieving gold standard concepts ...', end=' ', flush=True)
    all_concepts = set()
    filtered = set()
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
                    skipped.add(c)
                else:
                    if pos_filter and not re.fullmatch(pos_filter,
                                                       c.pos_sequence()):
                        filtered.add(c)
                    else:
                        all_concepts.add(c.normalized_concept())
    print(f'Skipped {len(skipped)} concepts not bounded at tokens boundaries '
          f'and filtered out {len(filtered)} with the POS-tag filter:',
          str(pos_filter))
    if exclude_unigrams:
        all_concepts = {c for c in all_concepts if not len(c) == 1}
    return all_concepts


