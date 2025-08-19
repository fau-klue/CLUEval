import os
import re
import pandas as pd

from typing import Dict, List


class Convert:
    """ Convert BIO into spans."""
    def __init__(self, path_to_file: os.path.abspath):
        self.path_to_file = path_to_file

    def __call__(self, gold=True, tag_column_id: int = 1):
        # TODO: Refactor this to a cleaner version later
        span_dictionary = dict(start=[],
                               end=[],
                               anon=[],
                               cat=[],
                               risk=[],
                               domain=[],
                               set=[],
                               verdict=[],
                               text=[]
                               )
        if gold:
            span_dictionary.pop("anon")
            for start_id, end_id, cat, risk, domain, verdict_id, tokens in self.gold():
                span_dictionary["start"].append(start_id)
                span_dictionary["end"].append(end_id)
                span_dictionary["cat"].append(cat)
                span_dictionary["risk"].append(risk)
                span_dictionary["domain"].append(domain)
                span_dictionary["set"].append("test")
                span_dictionary["verdict"].append(verdict_id)
                span_dictionary["text"].append(tokens)
        else:
            # Extract prediction spans
            if tag_column_id == 1:
                span_dictionary.pop("cat")
                span_dictionary.pop("risk")
                for start_id, end_id, tag, domain, verdict_id, tokens in self.predictions(tag_column_id=tag_column_id):
                    span_dictionary["start"].append(start_id)
                    span_dictionary["end"].append(end_id)
                    span_dictionary["anon"].append(tag)
                    span_dictionary["domain"].append(domain)
                    span_dictionary["set"].append("test")
                    span_dictionary["verdict"].append(verdict_id)
                    span_dictionary["text"].append(tokens)

            if tag_column_id == 2:
                span_dictionary.pop("anon")
                span_dictionary.pop("risk")
                for start_id, end_id, tag, domain, verdict_id, tokens in self.predictions(tag_column_id=tag_column_id):
                    span_dictionary["start"].append(start_id)
                    span_dictionary["end"].append(end_id)
                    span_dictionary["cat"].append(tag)
                    span_dictionary["domain"].append(domain)
                    span_dictionary["set"].append("test")
                    span_dictionary["verdict"].append(verdict_id)
                    span_dictionary["text"].append(tokens)

            if tag_column_id == 3:
                span_dictionary.pop("anon")
                span_dictionary.pop("cat")
                for start_id, end_id, tag, domain, verdict_id, tokens in self.predictions(tag_column_id=tag_column_id):
                    span_dictionary["start"].append(start_id)
                    span_dictionary["end"].append(end_id)
                    span_dictionary["risk"].append(tag)
                    span_dictionary["domain"].append(domain)
                    span_dictionary["set"].append("test")
                    span_dictionary["verdict"].append(verdict_id)
                    span_dictionary["text"].append(tokens)
        return pd.DataFrame.from_dict(span_dictionary)

    @staticmethod
    def _line_generator(read_lines: List):
        for i, line in enumerate(read_lines):
            try:
                yield i, line, read_lines[i + 1]
            except IndexError:
                yield i, line, None

    def gold(self):
        """ Extract annotated spans from BIO format and save them as tsv.
        Iterate over each line in BIO and check whether current annotation is 'O'. If not do:
            - extract domain, token and check whether current line begins a new span. If yes do:
                - extract entity classes and risks
                - get token position as start id
                - check if next line is outside or new span. If yes do:
                    - get token position as end id and return spans information
        Check whether current line is inside an existing span and next line is either outside or begin of new one.
        If yes do:
            - Extract token position and return spans information
        """
        with (open(self.path_to_file, "r", encoding="utf-8") as in_f):
            read_lines = in_f.readlines()
            token_id = 0
            start_id = 0
            end_id = 0
            domain = None
            cat = None
            risk = None
            verdict_id = None
            tokens = []

            for i, line in enumerate(read_lines):
                current_line = line.strip()
                # Extract verdict id from document
                if "newdoc id" in current_line:
                    verdict_id = current_line.split("=")[1]
                # Ignore document, sentence id and empty string
                if ("newdoc id" not in current_line and
                        "sent_id" not in current_line and
                        current_line != ""):
                    token_id += 1
                    try:
                        current_line = current_line.split("\t")
                        next_line = read_lines[i + 1].strip().split("\t")
                        if current_line[1] != "O":
                            domain = current_line[-1].lower()
                            tokens.append(current_line[0])
                            if current_line[1].startswith("B-"):
                                cat = current_line[2].strip("B-")
                                risk = current_line[3].strip("B-")
                                start_id = token_id
                                if (next_line[1].startswith("B-")
                                        or next_line[1] == "O"):
                                    end_id = token_id
                                    yield start_id, end_id, cat, risk, domain, verdict_id, " ".join(tokens)
                                    tokens = []
                        if current_line[1].startswith("I-") and (
                                "B-" in next_line[1] or
                                "O" in next_line[1]):
                            end_id = token_id
                            yield start_id, end_id, cat, risk, domain, verdict_id, " ".join(tokens)
                            tokens = []
                    except IndexError:
                        if tokens:
                            end_id = token_id
                            yield start_id, end_id, cat, risk, domain, verdict_id, " ".join(tokens)
                            tokens = []

    def predictions(self, tag_column_id=1):
        """
        Extract predicted spans from BIO file.
        Iterate over each line and check whether predicted tag for current lines header is 'O'. If not do:
            - extract domain, predicted tag and token from line
            - extract token position as start id
            - check if next predicted tag is also 'O', begin of new tag or inside ('I-') span but doesn't have the
            same class. If yes do:
                - extract token position as end id
                - return spans information
        :param tag_column_id: Indicates which prediction class should be extracted.
        Options: 1: anon, 2: entity classes and 3: risk. Default: 1
        """
        with open(self.path_to_file, "r", encoding="utf-8") as in_f:
            read_lines = in_f.readlines()
            token_id = 0
            start_id = 0
            end_id = 0
            verdict_id = None
            domain = None
            tags_list = []
            tokens = []

            for i, line in enumerate(read_lines):
                current_line = line.strip()

                # Extract verdict id if available
                if "newdoc id" in current_line:
                    verdict_id = current_line.split("=")[1]

                # Extract predicted spans
                if ("newdoc id" not in current_line
                        and "sent_id" not in current_line
                        and current_line != ""
                ):
                    token_id += 1
                    current_line = current_line.split("\t")
                    next_line = read_lines[i + 1].strip().split("\t")
                    # Start processing line if current tag is not "O"
                    if current_line[tag_column_id] != "O":
                        verdict_id = current_line[-1].lower()
                        domain = current_line[-2].lower()
                        tag = re.sub(r"^(B-)|^(I-)", "", current_line[tag_column_id])
                        tokens.append(current_line[0])
                        try:
                            if not tags_list:
                                # Begin of current span
                                start_id = token_id
                                tags_list.append(tag)
                            # Generate current span
                            if (next_line[tag_column_id].startswith("O")  # If next tag is 'O'
                                    or next_line[tag_column_id].startswith("B-")  # If new span (start with 'B-'
                                    or re.sub(r"^(B-)|^(I-)", "", next_line[tag_column_id]) != tag
                                    # If new tag starts with 'I-' instead of 'B-'
                            ):
                                end_id = token_id
                                yield start_id, end_id, tag, domain, verdict_id, " ".join(tokens)
                                tokens = []
                                tags_list = []
                        except IndexError:
                            # Mostly due do last line
                            if tokens:
                                end_id = token_id
                                yield start_id, end_id, tag, domain, verdict_id, " ".join(tokens)
                                tokens = []
                                tags_list = []
