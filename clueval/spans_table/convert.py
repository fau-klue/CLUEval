from .data import ParsedSpan, Token
from .parser import BioToSpanParser
from .unify import OverlapComponentUnifier, MultiHeadSpanTokenUnifier

from collections import defaultdict
import pandas as pd


class Convert:
    def __init__(self, path_to_file:str, annotation_layer:str|list[str] | None=None, domain_column:int | None=None, doc_id_column:int | None=None):
        self.path_to_file = path_to_file
        self.annotation_layer = annotation_layer
        self.domain_column = domain_column
        self.doc_id_column = doc_id_column

    def __call__(self, head: int = None):
        if not head:
            spans_df = self.build_unified_dataframe()
            spans_df = spans_df.join(spans_df["label"].apply(pd.Series).add_prefix("head_"))
        else:
            spans_df = self.build_head_wise_dataframe(head=head)
            spans_df = spans_df.join(spans_df["label"].apply(pd.Series).add_prefix("head_"))
        return spans_df.sort_values(by=["start_id", "end_id"])

    def build_unified_dataframe(self):
        doc_to_spans_mapping, doc_to_tokens_mapping, list_of_doc_ids = self.parse()
        unified_candidate_spans = []
        for doc_id in list_of_doc_ids:
            spans_by_doc_id = doc_to_spans_mapping[doc_id]
            tokens_by_doc_id = doc_to_tokens_mapping[doc_id]

            # Map overlap components to a unified span
            component_unifier = OverlapComponentUnifier(spans_by_doc_id)
            intermediate_overlap_components = component_unifier()

            span_token_unifier = MultiHeadSpanTokenUnifier(intermediate_overlap_components, tokens_by_doc_id)
            unified_spans = [span for span in span_token_unifier()]
            unified_candidate_spans.extend(unified_spans)
        return pd.DataFrame(unified_candidate_spans)

    def build_head_wise_dataframe(self, head: int=1):
        self.parse()
        head_spans = []
        head = head
        #TODO: implement option to build dataframe from one particular head
        return pd.DataFrame(head_spans)


    def parse(self):
        if not self.annotation_layer:
            raise ValueError("No input for annotation_layer")
        if type(self.annotation_layer) == str:
            n_tag_columns = 1
        else:
            n_tag_columns = len(self.annotation_layer)

        # Convert BIO to spans
        parser = BioToSpanParser(self.path_to_file)
        list_of_spans, list_of_tokens = parser(tag_column=1,
                                               n_tag_columns=n_tag_columns,
                                               domain_column=self.domain_column,
                                               doc_id_column=self.doc_id_column,
                                               extract_tokens=True
                                               )

        if isinstance(self.annotation_layer, list):
            for i in range(1, len(self.annotation_layer)):
                spans_per_layer, _ = parser(tag_column=i+1,
                                            domain_column=self.domain_column,
                                            doc_id_column=self.doc_id_column,
                                            extract_tokens=False
                                            )
                list_of_spans.extend(spans_per_layer)

        #  Partition spans and Token objects by doc_id
        doc_ids = sorted(list(set(token.doc_id for token in list_of_tokens)))
        doc_to_spans_dict = self.doc_to_object_mapping(list_of_spans)
        doc_to_tokens_dict = self.doc_to_object_mapping(list_of_tokens)
        return doc_to_spans_dict, doc_to_tokens_dict, doc_ids

    @staticmethod
    def doc_to_object_mapping(list_of_object:list[ParsedSpan|Token]):
        """

        :param list_of_object:
        :return:
        """
        mapping = defaultdict(list)
        for data_object in list_of_object:
            mapping[data_object.doc_id].append(data_object)
        return mapping