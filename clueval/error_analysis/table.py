import pandas as pd
import re

class TableForErrorAnalysis:
    def __init__(self, x:pd.DataFrame, token_id_sentence_mapping:dict):
        self.x = x
        self.token_id_sentence_mapping = token_id_sentence_mapping
    
    def __call__(self,  headers:list[str], filter_head:str, filter_head_value:str, windows:int=10):
        # Keep rows in x according to filter_head_value and where status is not 'exact'
        filtered_x = self.x.loc[(self.x[filter_head] == filter_head_value) & ~(self.x["status"] == "exact")]
        erroneous_span_df = self.extract_context_for_error_spans(filtered_x, self.token_id_sentence_mapping, windows=windows)
        erroneous_span_df = erroneous_span_df[["doc_token_id_start",
                                              "doc_token_id_end", 
                                              "doc_token_id_start_Y", 
                                              "doc_token_id_end_Y", 
                                              "domain", *headers, "text", "text_Y", "sent", "status"]].fillna("---").rename(columns={"doc_token_id_start": "token_id_start",
                                    "doc_token_id_end": "token_id_end",
                                    "doc_token_id_start_Y": "token_id_pred_start",
                                    "doc_token_id_end_Y": "token_id_pred_end", 
                                    "text": "gold_span", "text_Y": "pred_span", 
                                    "sent": "context", "status": "error_type"
                                     }
                                ).sort_values(by=["error_type","token_id_start"]).reset_index(drop=True)

        return erroneous_span_df

    @staticmethod
    def extract_context_for_error_spans(input_df:pd.DataFrame, gold_sentence_mapping:dict, windows:int=10):
        # Extract contexts for manual analysis
        input_df = input_df.copy()
        for i, row in input_df.iterrows():
            # span start, end positions
            start = row["start"]
            end = row["end"]
            sent_windows = ""
            for j, token_ids in enumerate(gold_sentence_mapping["token_ids"]):
                if start in token_ids:
                    sentence = gold_sentence_mapping["sents"][j]
                    left_windows = max(0, token_ids.index(start) - windows)
                    right_windows = min(len(sentence), token_ids.index(end) + windows)
                    sent_windows += " ".join(sentence[left_windows:right_windows])
                    if token_ids.index(start) != 0:
                        sent_windows = "[...] " + sent_windows
                    if right_windows != len(sentence):
                        sent_windows +=  " [...]"
            input_df.loc[i, "sent"] = f"{sent_windows}"
        return input_df

    @staticmethod
    def align_prediction_with_gold_annotation(input_df: pd.DataFrame):
        """ Check whether a gold span has been split by the prediction. """
        for i, row in input_df.iterrows():
            try:
                if (input_df["start"].iloc[i+1] != -100 
                    and (row["start_Y"] >= input_df["start"].iloc[i+1] 
                    and row["end_Y"] <= input_df["end"].iloc[i+1])):
                    input_df.loc[i, "start"] = input_df["start"].iloc[i+1]
                    input_df.loc[i, "end"] = input_df["end"].iloc[i+1]
                    input_df.loc[i, "doc_token_id_start"] = input_df["doc_token_id_start"].iloc[i+1]
                    input_df.loc[i, "doc_token_id_end"] = input_df["doc_token_id_end"].iloc[i+1]
                    input_df.loc[i, "text"] = input_df["text"].iloc[i+1]
            except IndexError:
                pass
        return input_df

def _tokenize_with_offsets(text):
    """
    """ 
    tokens = []
    for m in re.finditer(r"\w+[.,]?|\w+[.,]?", text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens

def _find_exact_or_token_subspans(target, context, min_tokens=1, min_chars=3):
    """
    Return list of (start, end) spans in context that correspond to:
      - an exact match of target (preferred), or
      - all continuous substrings formed by consecutive tokens from target
        that occur in context, subject to minimum token/char length to avoid tiny matches.

    The returned spans are on character offsets in context and are merged/coalesced.
    """
    if not target:
        return []

    # exact match preferred
    idx = context.find(target)
    if idx >= 0:
        return [(idx, idx + len(target))]

    # fallback: token-based continuous substrings (only substrings made of consecutive tokens)
    target_tokens = _tokenize_with_offsets(target)
    context_tokens = _tokenize_with_offsets(context)
    
    # Build list of strings for joined token subsequences from target (with their char lengths)
    target_token_texts = [t for (t, s, e) in target_tokens]
    n = len(target_token_texts)
    spans = []

    # For efficiency: only consider subsequences with at least min_tokens tokens or min_chars chars
    for i in range(n):
        for j in range(i, n + 1):  # j is exclusive; ensures at least min_tokens
            subseq_text = " ".join(target_token_texts[i:j])
            if len(subseq_text) < min_chars:
                continue
            # find all occurrences of this subseq in context
            start_search = 0
            while True:
                pos = context.find(subseq_text, start_search)
                if pos == -1:
                    break
                spans.append((pos, pos + len(subseq_text)))
                start_search = pos + 1

    if not spans:
        # final fallback: allow single-token matches if token is long enough
        for tok, _, _ in target_tokens:
            if len(tok) >= max(min_chars, 4):   # require reasonably long single token
                for m in re.finditer(re.escape(tok), context):
                    spans.append((m.start(), m.end()))

    if not spans:
        return []

    # Merge overlapping/adjacent spans
    spans = sorted(spans)
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(s, e) for s, e in merged]

def highlight_text_with_squares(row, min_tokens=1, min_chars=3):
    """
    row: dict with keys 'context', 'gold_span', 'pred_span'
    min_tokens/min_chars: control the minimum size of a recovered fallback span
    """
    context = row.get("context", "")
    gold_text = row.get("gold_span", "") or ""
    pred_text = row.get("pred_span", "") or ""
    # Find gold spans: prefer exact match, else token-subsequence-based spans
    gold_spans = _find_exact_or_token_subspans(gold_text, context, min_tokens=min_tokens, min_chars=min_chars)

    # Find pred spans: prefer exact match, else token-subsequence-based spans
    pred_spans = _find_exact_or_token_subspans(pred_text, context, min_tokens=min_tokens, min_chars=min_chars)

    # Character-level mask
    char_status = [0] * len(context)  # 0 none, 1 gold, 2 pred, 3 both

    for s, e in gold_spans:
        # guard against bad spans
        s = max(0, min(s, len(context)))
        e = max(0, min(e, len(context)))
        for i in range(s, e):
            char_status[i] |= 1

    for s, e in pred_spans:
        s = max(0, min(s, len(context)))
        e = max(0, min(e, len(context)))
        for i in range(s, e):
            char_status[i] |= 2

    # Build output with contiguous same-status segments
    out = []
    i = 0
    while i < len(context):
        st = char_status[i]
        if st == 0:
            out.append(context[i])
            i += 1
            continue
        j = i
        while j < len(context) and char_status[j] == st:
            j += 1
        seg = context[i:j]
        if st == 1:
            out.append(f"🟨{seg}🟨")
        elif st == 2:
            out.append(f"🟥{seg}🟥")
        elif st == 3:
            out.append(f"🟩{seg}🟩")
        i = j

    return "".join(out)


