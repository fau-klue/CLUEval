import pandas as pd

class ErrorTable:
    def __init__(self, match_table: pd.DataFrame, candidate_table: pd.DataFrame, token_position_sentence_mapping:dict):
        self.match_table = match_table
        self.candidate_table = candidate_table
        self.token_position_sentence_mapping = token_position_sentence_mapping
    
    def __call__(self,  annotation_layer:str|list[str], windows:int=10):
        overlaps = []
        intermediate_table = self.match_table[["start", "end", "token_id_start", "token_id_end", "domain", "text", "status"] + [layer for layer in annotation_layer if not layer.endswith("_Y")]].copy()
        # Check for all overlapping spans from candidate table
        for i, row in intermediate_table.iterrows():
            overlap = self.candidate_table[~((row["end"] < self.candidate_table["start"]) | (self.candidate_table["end"] < row["start"]))]
            if not overlap.empty:
                _overlap = overlap.copy()
                _overlap["start_X"] = row["start"]
                _overlap["end_X"] = row["end"]
                overlaps.append(_overlap)
        # Handle empty overlap
        try:
            overlap_df = pd.concat(overlaps)
            # Left join -> Merge all possible overlaps to intermediate_table
            joined_overlap_df = intermediate_table.merge(overlap_df, how="left", left_on=["start", "end"],
                                                         right_on=["start_X", "end_X"], suffixes=("", "_Y"))
            joined_overlap_df.loc[joined_overlap_df[["start_Y", "end_Y"]].isna().any(axis=1), ["start_Y", "end_Y"]] = -100
            joined_overlap_df[["start_Y", "end_Y"]] = joined_overlap_df[["start_Y", "end_Y"]].astype("Int64")
        except ValueError:
            joined_overlap_df = intermediate_table
            joined_overlap_df["start_Y"] = -100
            joined_overlap_df["end_Y"] = -100
            joined_overlap_df["token_id_start_Y"] = -100
            joined_overlap_df["token_id_end_Y"] = -100
            joined_overlap_df["text_Y"] = None

        erroneous_table = self.extract_and_highlight_spans(joined_overlap_df, self.token_position_sentence_mapping, annotation_layer=annotation_layer, windows=windows)
        erroneous_table = erroneous_table[["token_id_start",
                                              "token_id_end",
                                              "token_id_start_Y",
                                              "token_id_end_Y",
                                              "domain",
                                               *annotation_layer,
                                               "text",
                                               "text_Y",
                                               "context",
                                               "status"]].fillna("---").rename(columns={"doc_token_id_start_Y": "token_id_pred_start",
                                                                                        "doc_token_id_end_Y": "token_id_pred_end",
                                                                                        "text": "reference",
                                                                                        "text_Y": "candidate",
                                                                                        "status": "error_type"
                                                                                        }
                                ).sort_values(by=["error_type","token_id_start"]).reset_index(drop=True)

        return erroneous_table

    @staticmethod
    def extract_and_highlight_spans(input_df:pd.DataFrame, gold_sentence_mapping:dict,  annotation_layer: str|list[str], windows:int=10):
        #TODO: Revise this method: Highlight all candidate spans for each reference span. At the moment, candidate spans are considered separately for the same gold annotation
        # Extract contexts for manual analysis

        input_df = input_df.copy()
        if isinstance(annotation_layer, str):
            annotation_layer = [annotation_layer]

        grouped_df = input_df.groupby(["start", "end"])
        dict_of_erroneous_spans = {"start": [],
                                   "end": [],
                                   "token_id_start": [],
                                   "token_id_end": [],
                                   "token_id_start_Y": [],
                                   "token_id_end_Y": [],
                                   **{layer: [] for layer in annotation_layer},
                                   "doc_id": [],
                                   "domain": [],
                                   "status": [],
                                   "text": [],
                                   "text_Y": [],
                                   "context": []
                                   }


        for group_pos, group in grouped_df:
            # span start, end positions
            ref_start = int(group_pos[0])
            ref_end = int(group_pos[1])
            cand_token_positions = set([int(entry)  for positions in zip(group["start_Y"].values, group["end_Y"].values) for entry in positions])
            cand_token_id_start = None if group["token_id_start_Y"].isna().all() else " ".join([str(entry) if entry else "" for entry in group["token_id_start_Y"]])
            cand_token_id_end = None if group["token_id_end_Y"].isna().all() else " ".join(str(entry) if entry else "" for entry in group["token_id_end_Y"])


            reference_text = group["text"].iloc[0].split()

            group["number_overlapping_tokens_with_x"] = group.apply(
                lambda r: len([token for token in r["text_Y"].split() if token in reference_text]), axis=1)

            # update dict_of_erroneous_spans
            dict_of_erroneous_spans["start"].append(ref_start)
            dict_of_erroneous_spans["end"].append(ref_end)
            dict_of_erroneous_spans["token_id_start"].append(group["token_id_start"].iloc[0])
            dict_of_erroneous_spans["token_id_end"].append(group["token_id_end"].iloc[0])
            dict_of_erroneous_spans["token_id_start_Y"].append(cand_token_id_start)
            dict_of_erroneous_spans["token_id_end_Y"].append(cand_token_id_end)
            dict_of_erroneous_spans["doc_id"].append(group["doc_id"].iloc[0])
            dict_of_erroneous_spans["domain"].append(group["domain"].iloc[0])
            dict_of_erroneous_spans["status"].append(group["status"].iloc[0])
            dict_of_erroneous_spans["text"].append(" ".join(reference_text))
            dict_of_erroneous_spans["text_Y"].append(" | ".join([text for text in group["text_Y"].values]))

            context = []
            for layer in annotation_layer:
                if layer.endswith("_Y"):
                    dict_of_erroneous_spans[layer].append(group.loc[group["number_overlapping_tokens_with_x"].idxmax()][layer])
                else:
                    dict_of_erroneous_spans[layer].append(group[layer].iloc[0])

            for j, token_ids in enumerate(gold_sentence_mapping["token_ids"]):
                if ref_start in token_ids:
                    sentence = gold_sentence_mapping["sents"][j]
                    left_windows = max(0, token_ids.index(ref_start) - windows)
                    right_windows = min(len(sentence), token_ids.index(ref_end) + windows)

                    # Assign token status according to corpus position:
                    # 0: Token does not belong to any span
                    # 1: Token contained in both ref. and candidate spans
                    # 2: Token occurs only in reference
                    # 3: Token appears only in candidate
                    token_status = [0] * len(token_ids)
                    for k in range(left_windows, right_windows):
                        token_id = token_ids[k]
                        token_in_ref = ref_start <= token_id <= ref_end
                        token_in_cand = token_id in cand_token_positions # cand_start <= token_id <= cand_end
                        # Token in both reference and candidate segment
                        if token_in_ref and token_in_cand:
                            token_status[k] = 1
                        # Token in reference span
                        elif token_in_ref:
                            token_status[k] = 2
                        # token in candidate span
                        elif token_in_cand:
                            token_status[k] = 3

                    # Trim context according to windows size
                    trimmed_sentence = sentence[left_windows:right_windows]
                    trimmed_token_status = token_status[left_windows:right_windows]

                    # highlight segments
                    si = 0
                    while si < len(trimmed_token_status):
                        st = trimmed_token_status[si]
                        # Ignore tokens that do not belong to any span
                        if st == 0:
                            context.append(trimmed_sentence[si])
                            si += 1
                            continue
                        # Determine the start and end tokens for each span based on token status
                        sj = si
                        while sj < len(trimmed_token_status) and trimmed_token_status[sj] == st:
                            sj += 1
                        segment = " ".join(trimmed_sentence[si:sj])
                        if st == 1:
                            context.append(f"🟩{segment}🟩") # Both ref. and cand
                        elif st == 2:
                            context.append(f"🟥{segment}🟥") # Ref. only
                        elif st == 3:
                            context.append(f"🟧{segment}🟧") # Cand. only
                        si = sj

                    context = " ".join(context)
                    if left_windows != 0:
                        context = "[...] " + context
                    if right_windows != len(sentence):
                        context += " [...]"

            dict_of_erroneous_spans["context"].append(context)
        highlighted_error_df = pd.DataFrame.from_dict(dict_of_erroneous_spans)
        return highlighted_error_df
