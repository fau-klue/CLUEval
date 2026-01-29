import pandas as pd
from collections import defaultdict

from .utils import majority_vote
from .data import ParsedSpan, Token
from .parser import BioToSpanParser
from .unify import OverlapComponentUnifier, MultiHeadSpanTokenUnifier



class Convert:
    def __init__(self, path_to_file:str, annotation_layer:str|list[str]| None=None, token_id_column:int|None=None, domain_column:int|None=None, doc_id_column:int|None=None):
        self.path_to_file = path_to_file
        self.annotation_layer = annotation_layer
        self.token_id_column = token_id_column
        self.domain_column = domain_column
        self.doc_id_column = doc_id_column


    def __call__(self, id_prefix="id", head:int|None=None):
        # spans_df = self.build_unified_dataframe()
        if head is not None:
            spans_df = self.build_head_wise_dataframe(head=head)
        else:
            spans_df = self.build_unified_dataframe()
            spans_df = (spans_df.join(spans_df["label"].apply(pd.Series).add_prefix("head_")).drop(columns="label"))

        spans_df.rename(columns={"start_id": "start", "end_id": "end"}, inplace=True)
        spans_df = spans_df.sort_values(by=["start", "end"])
        spans_df = self._assign_span_ids(spans_df, prefix=id_prefix)
        return spans_df

    def build_unified_dataframe(self):
        doc_to_spans_mapping, doc_to_tokens_mapping, list_of_doc_ids = self.parse()
        all_unified_spans = []
        for doc_id in list_of_doc_ids:
            spans_by_doc_id = doc_to_spans_mapping[doc_id]
            tokens_by_doc_id = doc_to_tokens_mapping[doc_id]

            # Map overlap components to a unified span
            component_unifier = OverlapComponentUnifier(spans_by_doc_id)
            intermediate_overlap_components = component_unifier()

            span_token_unifier = MultiHeadSpanTokenUnifier(intermediate_overlap_components, tokens_by_doc_id)
            unified_spans = [span for span in span_token_unifier()]
            all_unified_spans.extend(unified_spans)
        return pd.DataFrame(all_unified_spans)

    def build_head_wise_dataframe(self, head: int=1):
        doc_to_spans_mapping, doc_to_tokens_mapping, list_of_doc_ids =  self.parse()
        head_spans = []
        head = head
        # TODO: NU -- implement option to build dataframe from one particular head
        for doc_id in list_of_doc_ids:
            spans_by_doc_id = doc_to_spans_mapping[doc_id]
            tokens_by_doc_id = doc_to_tokens_mapping[doc_id]
            filtered_spans = [span for span in spans_by_doc_id if span.head == head]
            for span in filtered_spans:
                # token = tokens_by_doc_id[0]
                # print(f"Span Start: {span.start_id}, Span end: {span.end_id}")
                # print(token, "---", type(type(token)))
                span_tokens = [token for token in tokens_by_doc_id
                               if span.start_id <= token.position <= span.end_id]
                span_tokens.sort(key=lambda  t: t.position)
                if span_tokens:
                    text = " ".join([token.token for token in span_tokens])
                    doc_token_id_start = span_tokens[0].token_id
                    doc_token_id_end = span_tokens[-1].token_id
                    # Get Label
                    labels = [token.label[head] for token in span_tokens]
                    label = majority_vote(labels)
                    domain = span_tokens[0].domain
                else:
                    text = ""
                    doc_token_id_start = None
                    doc_token_id_end = None
                    label = None
                    domain = None

                head_spans.append({
                "doc_id": span.doc_id,
                "start_id": span.start_id,
                "end_id": span.end_id,
                "doc_token_id_start": doc_token_id_start,
                "doc_token_id_end": doc_token_id_end,
                "text": text,
                "label": label,
                "domain": domain
            })
    
        df = pd.DataFrame(head_spans)
        # Rename columns for consistency with build_unified_dataframe
        if not df.empty:
            df = df.rename(columns={'start_id': 'start', 'end_id': 'end'})
            df = df.sort_values(by=['start', 'end']).reset_index(drop=True)
            df = self._assign_span_ids(df, prefix=f"head{head}_")
        
        return df


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
                                               doc_token_id_column=self.token_id_column,
                                               domain_column=self.domain_column,
                                               doc_id_column=self.doc_id_column,
                                               extract_tokens=True
                                               )

        if isinstance(self.annotation_layer, list):
            for i in range(1, len(self.annotation_layer)):
                spans_per_layer, _ = parser(tag_column=i+1,
                                            doc_token_id_column=self.token_id_column,
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

    @staticmethod
    def _assign_span_ids(inp_data: pd.DataFrame, prefix: str = "id"):
        """
        Assign IDs to annotated spans using the given prefix.
        :param inp_data: Pandas dataframe with extracted spans
        :param prefix: IDs prefix
        """
        inp_data["id"] = [f"{prefix}{i + 1:06d}" for i in range(inp_data.shape[0])]
        return inp_data
