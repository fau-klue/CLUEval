from itertools import chain
from collections import defaultdict

from .utils import majority_vote
from .data import ParsedSpan, SpanComponent, UnifiedSpan, Token


class MultiHeadSpanTokenUnifier:

    def __init__(self, spans: list[SpanComponent], tokens: list[Token]):
        self.spans = spans
        self.index_to_token_mapping = self.map_token_to_index(tokens)

    def __call__(self):
        for span in self.spans:
            concatenated_tokens = []
            labels = []
            token_id_start = 0
            token_id_end = 0
            domain = None
            for position in range(span.position_start, span.position_end+1):
                token = self.index_to_token_mapping.get(position)
                domain = token.domain
                if token.position == span.position_start:
                    token_id_start = token.token_id
                if token.position == span.position_end:
                    token_id_end = token.token_id

                labels.append(token.label)
                concatenated_tokens.append(token.token)

            if all(isinstance(label, list) for label in labels):
                labels = self.transpose(labels)
                majority_label = [majority_vote(label) for label in labels]
            else:
                majority_label = majority_vote(labels)

            yield UnifiedSpan(position_start=span.position_start,
                              position_end=span.position_end,
                              token_id_start=token_id_start,
                              token_id_end=token_id_end,
                              text=" ".join(concatenated_tokens),
                              label=majority_label,
                              doc_id=span.doc_id,
                              domain=domain
                              )

    @staticmethod
    def transpose(labels: list[list[str]]):
        """

        :param labels:
        :return:
        """
        return [list(row) for row in zip(*labels)]

    @staticmethod
    def map_token_to_index(list_of_tokens: list[Token]):
        mapping = defaultdict(Token)
        for token in list_of_tokens:
            mapping[token.position] = token
        return mapping


class OverlapComponentUnifier:

    def __init__(self, spans: list[ParsedSpan]):
        self.spans = spans

    def __call__(self):
        intermediate_combined_spans = []
        components = self.get_overlap_components()
        for component in components:
            intermediate_combined_spans.append(self.combined_span_from_component(component))
        return sorted(intermediate_combined_spans, key=lambda span: (span.position_start, span.position_end))

    def get_overlap_components(self):
        """
        Group spans into overlap components.

        Spans are grouped per doc_id and sorted by position, then swept once:
        a span joins the current component if it starts at or before the
        component's running max end. Because the sweep only ever needs the
        running max end (not all pairwise comparisons), this is O(n log n)
        instead of the O(n ** 2) pairwise overlap check used before.

        """
        overlap_components = []
        spans_by_doc = defaultdict(list)
        for span in self.spans:
            spans_by_doc[span.doc_id].append(span)

        for doc_spans in spans_by_doc.values():
            sorted_spans = sorted(doc_spans, key=lambda span: (span.position_start, span.position_end))
            current_component = []
            current_max_end = None
            for span in sorted_spans:
                if current_component and span.position_start <= current_max_end:
                    current_component.append(span)
                    current_max_end = max(current_max_end, span.position_end)
                else:
                    if current_component:
                        overlap_components.append(current_component)
                    current_component = [span]
                    current_max_end = span.position_end
            if current_component:
                overlap_components.append(current_component)
        return overlap_components

    @staticmethod
    def combined_span_from_component(component: list[ParsedSpan]):
        """

        :param component:
        :return:
        """
        flatten_component = list(chain.from_iterable([[span.position_start, span.position_end] for span in component]))
        doc_id = list(set([span.doc_id for span in component]))[0]
        return SpanComponent(position_start=min(flatten_component), position_end=max(flatten_component), doc_id=doc_id)
