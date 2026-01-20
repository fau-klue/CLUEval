import re
import pandas as pd

from .data import ParsedSpan, Token

class BioToSpanParser:
    """ Convert BIO into spans."""
    def __init__(self, path_to_file):
        self.path_to_file: str = path_to_file

    def __call__(self,
                 tag_column: int = 1,
                 n_tag_columns: int = 1,
                 doc_token_id_column: int = None,
                 domain_column: int = None,
                 doc_id_column: int = None,
                 extract_tokens: bool = False
                 ):
        spans = []
        tokens = []
        # Extract spans from BIO
        for span in self.extract_spans_from_iob(tag_column=tag_column,
                                                doc_token_id_column=doc_token_id_column
                                                ):
            spans.append(span)

        # Extract tokens from BIO
        if extract_tokens:
            for token in self.extract_tokens_from_iob(n_tag_columns=n_tag_columns,
                                                      doc_token_id_column=doc_token_id_column,
                                                      domain_column=domain_column):
                tokens.append(token)

        return spans, tokens

    def extract_tokens_from_iob(self, n_tag_columns=1, doc_token_id_column: int=None, domain_column: int=None):
        """
        :param n_tag_columns:
        :param doc_token_id_column:
        :param domain_column:

        """

        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            position = 0
            token_id = None
            doc_id = None
            domain = None
            lines = in_f.readlines()
            for i, line in enumerate(lines):
                # Extract document id if available
                current_line = line.strip().split("\t")
                if len(current_line) > 1:
                    # Extract document id if exists
                    if doc_token_id_column:
                        token_id = current_line[doc_token_id_column]
                        doc_id = "_".join(token_id.strip().split("_")[:2])
                    position += 1
                    # We assume that the tag column is directly adjacent to the token column
                    token = current_line[0]
                    if n_tag_columns == 1:
                        label = re.sub(r"[BI]-", "", current_line[1])
                    else:
                        label = [re.sub(r"[BI]-", "", tag) for tag in current_line[1:n_tag_columns+1]]
                    # Extract token_id and domain from BIO file if available
                    if domain_column is not None:
                        domain = current_line[domain_column].lower()
                    yield Token(position=position,
                                token_id=token_id,
                                token=token,
                                label=label,
                                doc_id=doc_id,
                                domain=domain)

    def extract_spans_from_iob(self, tag_column=1, doc_token_id_column: int = None):
        """
        Extract predicted spans from BIO file.
        Iterate over each line and check whether predicted tag for current lines header is 'O'. If not do:
            - extract domain, predicted tag and token from line
            - extract token position as start id
            - check if next predicted tag is also 'O', begin of new tag or inside ('I-') span but doesn't have the
            same class. If yes do:
                - extract token position as end id
                - return spans information
        :param tag_column:
        :param doc_token_id_column:
        """
        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            # doc_token_id is the predefined token_id in each document while token_id is the token position in the whole dataset
            token_id = 0
            start_id = None
            current_doc_id = None
            label = None

            lines = in_f.readlines()
            for i, line in enumerate(lines):
                current_line = line.strip().split("\t")


                # Extract next line if possible
                try:
                    next_line = lines[i + 1].strip().split("\t")
                except IndexError:
                    next_line = []

                # Extract spans based on predicted tags
                if len(current_line) > 1:
                    # Extract document id if exists
                    if doc_token_id_column:
                        doc_id = "_".join(current_line[doc_token_id_column].strip().split("_")[:2])
                        # Check if doc_id != current_doc_id
                        if doc_id != current_doc_id:
                            current_doc_id = doc_id
                    token_id += 1
                    current_tag = current_line[tag_column]
                    # Start processing line if current tag is not "O"
                    if current_tag != "O":
                        current_label = re.sub(r"^[BI]-", "", current_tag)
                        if label is None:
                            # Begin of current span
                            start_id = token_id
                            label = current_label
                        # Extract next label for comparison
                        if len(next_line) > 1:
                            next_tag = next_line[tag_column]
                            next_label = re.sub(r"^[BI]-", "", next_tag)
                        # Generate current span if next tag is 'O', if new span starts with 'B-' or
                        # if new entity tag -> Doesn't matter if it starts with 'I-' instead of 'B-'
                        if len(next_line) == 1 or next_line == [] or next_tag.startswith("B-") or next_label != label:
                            yield ParsedSpan(start_id=start_id,
                                             end_id=token_id,
                                             doc_id=current_doc_id,
                                             head=tag_column)
                            label = None