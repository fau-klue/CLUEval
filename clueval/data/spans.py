import os
import re
from typing import List

import pandas as pd


class Convert:
    """ Convert BIO into spans."""
    def __init__(self, path_to_file: os.path.abspath):
        self.path_to_file = path_to_file

    def __call__(self, tag_column: int = 1, tag_name: str = "ner_tags", domain_column: int = None, prefix: str = "id"):
        span_dictionary = {"start": [],
                           "end": [],
                           "text": [],
                           f"{tag_name}": [],
                           "doc_id": [],
                           "domain": []
                           }
        # Extract spans from BIO
        domain_dict = {"domain": []}
        for start_id, end_id, tag, domain, doc_id, tokens in self.to_span(tag_column=tag_column, domain_column=domain_column):
            span_dictionary["start"].append(start_id)
            span_dictionary["end"].append(end_id)
            span_dictionary["text"].append(tokens)
            span_dictionary[f"{tag_name}"].append(tag)
            span_dictionary["doc_id"].append(doc_id)
            span_dictionary["domain"].append(domain)
        dataframe = pd.DataFrame.from_dict(span_dictionary)
        dataframe = self._assign_span_ids(dataframe, prefix=prefix)
        return dataframe


    def to_span(self, tag_column=1, domain_column: int = None, doc_id_column: int = None):
        """
        Extract predicted spans from BIO file.
        Iterate over each line and check whether predicted tag for current lines header is 'O'. If not do:
            - extract domain, predicted tag and token from line
            - extract token position as start id
            - check if next predicted tag is also 'O', begin of new tag or inside ('I-') span but doesn't have the
            same class. If yes do:
                - extract token position as end id
                - return spans information
        """
        # TODO: Simplify this part for more clarity
        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            lines = [line.strip() for line in in_f.readlines() if line.strip()]
            token_id, start_id = 0, 0
            tokens, labels = [], []
            domain, doc_id = None, None

            for i, current_line in enumerate(lines):
                # Extract document id if available
                if "newdoc id" in current_line:
                    doc_id = current_line.split("=")[1].strip()
                if doc_id_column:
                    doc_id = doc_id_column
                current_line = current_line.strip().split("\t")
                # Extract next line if possible
                try:
                    next_line = lines[i + 1].strip().split("\t")
                except IndexError:
                    next_line = []
                # Merge single tokens to annotated spans
                if len(current_line) > 1:
                    token_id += 1
                    token = current_line[0]
                    current_tag = current_line[tag_column]
                    # Start processing line if current tag is not "O"
                    if current_tag != "O":
                        if domain_column is not None:
                            domain = current_line[domain_column].lower()
                        tokens.append(current_line[0])
                        label = re.sub(r"^[BI]-", "", current_tag)
                        if not labels:
                            # Begin of current span
                            start_id = token_id
                            labels.append(label)
                        # Extract next label for comparison
                        if len(next_line) > 1:
                            next_tag = next_line[tag_column]
                            next_label = re.sub(r"^[BI]-", "", next_tag)
                        # Generate current span if next tag is 'O', if new span starts with 'B-' or
                        # if new entity tag -> Doesn't matter if it starts with 'I-' instead of 'B-'
                        if len(next_line) == 1 or (next_tag.startswith("O") or next_tag.startswith("B-") or next_label != label):
                            yield start_id, token_id, label, domain, doc_id, " ".join(tokens)
                            tokens, labels = [], []
                    else:
                        if tokens:
                            yield start_id, token_id, label, domain, doc_id, " ".join(tokens)
                            tokens, labels = [], []

    @staticmethod
    def _assign_span_ids(inp_data: pd.DataFrame, prefix: str = "id"):
        """
        Assign IDs to annotated spans using the given prefix.
        :param inp_data: Pandas dataframe with extracted spans
        :param prefix: IDs prefix
        """
        inp_data["id"] = [f"{prefix}{i + 1:06d}" for i in range(inp_data.shape[0])]
        return inp_data

    @staticmethod
    def _line_generator(read_lines: List):
        for i, line in enumerate(read_lines):
            try:
                yield i, line, read_lines[i + 1]
            except IndexError:
                yield i, line, None