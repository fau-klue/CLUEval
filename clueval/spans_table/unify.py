import networkx as nx

from itertools import chain
from collections import  defaultdict

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
            doc_token_id_start = 0
            doc_token_id_end = 0
            domain = None
            for position in range(span.start_id, span.end_id+1):
                token = self.index_to_token_mapping.get(position)
                domain = token.domain
                if token.position == span.start_id:
                    doc_token_id_start = token.token_id
                if token.position == span.end_id:
                    doc_token_id_end = token.token_id

                labels.append(token.label)
                concatenated_tokens.append(token.token)

            if all(isinstance(label, list) for label in labels):
                labels = self.transpose(labels)
                majority_label = [majority_vote(label) for label in labels]
            else:
                majority_label = majority_vote(labels)

            yield UnifiedSpan(start_id=span.start_id,
                                  end_id=span.end_id,
                                  doc_token_id_start=doc_token_id_start,
                                  doc_token_id_end=doc_token_id_end,
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
    def map_token_to_index(list_of_tokens:list[Token]):
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
        return sorted(intermediate_combined_spans, key=lambda span: (span.start_id, span.end_id))

    def get_overlap_components(self):
        """

        :return:
        """
        overlap_components = []
        graph = nx.Graph()
        for i, span in enumerate(self.spans):
            graph.add_node(i, span=span)

        for i, s1 in enumerate(self.spans):
            # for j in range(i+1, len(self.spans)):
            for j, s2 in enumerate(self.spans[i+1:], start=i+1):
                if self.overlap(s1, s2):
                    graph.add_edge(i, j)

        for component in nx.connected_components(graph):
            overlap_components.append([graph.nodes[i]["span"] for i in component])
        return overlap_components

    @staticmethod
    def combined_span_from_component(component: list[ParsedSpan]):
        """

        :param component:
        :return:
        """
        flatten_component = list(chain.from_iterable([[span.start_id, span.end_id] for span in component]))
        doc_id = list(set([span.doc_id for span in component]))[0]
        return SpanComponent(start_id=min(flatten_component), end_id=max(flatten_component), doc_id=doc_id)

    @staticmethod
    def overlap(x: ParsedSpan, y: ParsedSpan):
        """

        :param x:
        :param y:
        :return:
        """
        if x.start_id <= y.end_id and y.start_id <= x.end_id:
            return True
        else:
            return False
