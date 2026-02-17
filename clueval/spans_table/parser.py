import re
from .data import ParsedSpan, Token


class BioToSentenceParser:
    def __init__(self, path):
        self.path = path
        self.token_id = 0
        
    def __call__(self):
        sents = dict(token_ids=[], sents=[])
        with open(self.path, "r", encoding="utf-8") as infile:
            readfile = infile.readlines()
            for token_ids, sent in self._generate(readfile):
                sents["token_ids"].append(token_ids)
                sents["sents"].append(sent)
        return sents
        
    def _generate(self, readfile):
        sent, token_ids = [], []
        for i, line in enumerate(readfile):
            if i != len(readfile) - 1:
                if line.strip() != "":
                    token = line.split("\t")[0]
                    sent.append(token)
                    token_ids.append(self.token_id+1)
                    self.token_id += 1
                else:
                    yield token_ids, sent
                    token_ids, sent = [], []
            else:
                yield token_ids, sent
                token_ids, sent = [], []

class BioToSpanParser:
    """Convert BIO into spans."""

    def __init__(self, path_to_file):
        self.path_to_file: str = path_to_file

    def __call__(
        self,
        tag_column: int = 1,
        n_tag_columns: int = 1,
        token_id_column: int | None = None,
        domain_column: int | None = None,
        doc_id_column: int | None = None,
        extract_tokens: bool | None = False,
    ):
        spans = []
        tokens = []
        # Extract spans from BIO
        for span in self.extract_spans_from_iob(
            tag_column=tag_column,
            doc_id_column=doc_id_column
        ):
            spans.append(span)

        # Extract tokens from BIO
        if extract_tokens:
            for token in self.extract_tokens_from_iob(
                n_tag_columns=n_tag_columns,
                token_id_column=token_id_column,
                domain_column=domain_column,
                doc_id_column=doc_id_column,
            ):
                tokens.append(token)

        return spans, tokens

    def extract_tokens_from_iob(
        self,
        n_tag_columns=1,
        token_id_column: int | None = None,
        domain_column: int | None = None,
        doc_id_column: int | None = None,
    ):
        """
        :param n_tag_columns:
        :param token_id_column:
        :param domain_column:
        :param doc_id_column:
        """

        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            position = 0
            token_id = None
            doc_id = None
            domain = None
            lines = in_f.readlines()
            for i, line in enumerate(lines):
                current_line = line.strip().split("\t")
                if len(current_line) > 1:
                    position += 1
                    # Extract document id if available
                    if doc_id_column is not None:
                        doc_id = current_line[doc_id_column]
                    else:
                        doc_id = None
                    # Extract token id if exists
                    if token_id_column:
                        token_id = current_line[token_id_column]
                    # We assume that the tag column is directly adjacent to the token column
                    token = current_line[0]
                    if n_tag_columns == 1:
                        label = re.sub(r"[BI]-", "", current_line[1])
                    else:
                        label = [
                            re.sub(r"[BI]-", "", tag)
                            for tag in current_line[1 : n_tag_columns + 1]
                        ]
                    # Extract token_id and domain from BIO file if available
                    if domain_column is not None:
                        domain = current_line[domain_column].lower()
                    yield Token(
                        position=position,
                        token_id=token_id,
                        token=token,
                        label=label,
                        doc_id=doc_id,
                        domain=domain,
                    )

    def extract_spans_from_iob(self, tag_column=1, doc_id_column: int | None = None):
        """
        Extract predicted spans from BIO file.
        Iterate over each line and check whether predicted tag for current lines header is 'O'. If not do:
            - extract domain, predicted tag and token from line
            - extract token position as start id
            - check if next predicted tag is also 'O', begin of new tag or inside ('I-') span but doesn't have the
            same class. If yes do:
                - extract token position as end id
                - return spans information
        :param doc_id_column:
        :param tag_column:
        """
        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            # doc_token_id is the predefined token_id in each document while token_id is the token position in the whole dataset
            token_id = 0
            start_id = 0
            doc_id = None
            current_doc_id = None
            label = None

            lines = in_f.readlines()
            for i, line in enumerate(lines):
                current_line = line.strip().split("\t")

                if doc_id_column is not None:
                    doc_id = current_line[doc_id_column]

                # Extract next line if possible
                try:
                    next_line = lines[i + 1].strip().split("\t")
                except IndexError:
                    next_line = []

                # Extract spans based on predicted tags
                if len(current_line) > 1:
                    token_id += 1
                    # Check if doc_id != current_doc_id
                    if doc_id != current_doc_id:
                        current_doc_id = doc_id
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
                        if (
                            len(next_line) == 1
                            or next_line == []
                            or next_tag.startswith("B-")
                            or next_label != label
                        ):
                            yield ParsedSpan(
                                position_start=start_id,
                                position_end=token_id,
                                doc_id=current_doc_id,
                                head=tag_column,
                            )
                            label = None
