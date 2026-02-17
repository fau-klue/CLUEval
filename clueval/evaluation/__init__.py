#!/usr/bin/env python3

from clueval.spans_table import Match, Convert
from .metrics import (
    MetricsForSpansAnonymisation,
    MetricsForCategoricalSpansAnonymisation,
)
import pandas as pd

def main(
    path_reference: str,
    path_candidate: str,
    annotation_layer: str | list[str],
    token_id_column: int | None = None,
    domain_column: int | None = None,
    doc_id_column: int | None = None,
    filter_head: str | None = None,
    head_value: str | None = None,
    categorical_evaluation: bool = False,
    categorical_head: str | list[str] | None = None,
    lenient_level: int = 0,
):
    list_of_span_evaluation = []

    if not annotation_layer:
        raise ValueError("No input for annotation_layer")
    if isinstance(annotation_layer, str):
        annotation_layer = [annotation_layer]

    # Convert BIO to spans tables
    reference_converter = Convert(
        path_reference,
        annotation_layer=annotation_layer,
        token_id_column=token_id_column,
        domain_column=domain_column,
        doc_id_column=doc_id_column,
    )
    reference_df = reference_converter()

    candidate_converter = Convert(
        path_candidate,
        annotation_layer=annotation_layer,
        token_id_column=token_id_column,
        domain_column=domain_column,
        doc_id_column=doc_id_column,
    )
    candidate_df = candidate_converter()

    # Evaluation metrics
    span_match_recall = Match(reference_df, candidate_df, annotation_layer=annotation_layer)
    span_match_precision = Match(candidate_df, reference_df, annotation_layer=annotation_layer)

    # Spans evaluation
    matched_span_recall = span_match_recall(on=["start", "end"])
    matched_span_precision = span_match_precision(on=["start", "end"])
    span_metrics = MetricsForSpansAnonymisation(
        precision_table=matched_span_precision, recall_table=matched_span_recall
    )(lenient_level=lenient_level, row_name="Span")

    list_of_span_evaluation.append(span_metrics)

    # Compute span metrics by filtered head value
    if filter_head:
        if not head_value:
            raise ValueError(f"Can not filter {filter_head} by None")

        filtered_span_metrics = MetricsForSpansAnonymisation(
            precision_table=matched_span_precision[
                matched_span_precision[filter_head + "_Y"] == head_value
            ],
            recall_table=matched_span_recall[
                matched_span_recall[filter_head] == head_value
            ],
        )(lenient_level=lenient_level, row_name=head_value.capitalize())
        list_of_span_evaluation.append(filtered_span_metrics)

    # Evaluation
    spans_eval_df = (
        pd.concat(list_of_span_evaluation)[["P", "R", "F1", "TP_Precision", "TP_Recall", "FP", "FN", "Support"]]
        .reset_index()
        .rename(columns={"index": "Span", "support": "Support"})
    )
    spans_eval_df["Label"] = "Span"
    spans_eval_df.rename(columns={"Span": "Level"}, inplace=True)

    # Compute metrics for categorical spans
    if categorical_evaluation:
        if not categorical_head:
            raise ValueError(f"Can not filter {categorical_head} by None")
        list_of_categorical_evaluations = []
        if isinstance(categorical_head, str):
            categorical_metrics = MetricsForCategoricalSpansAnonymisation(
                matched_span_precision,
                matched_span_recall,
                classification_head=categorical_head,
            )(lenient_level=lenient_level)
            list_of_categorical_evaluations.append(categorical_metrics)
        else:
            for head in categorical_head:
                categorical_metrics = MetricsForCategoricalSpansAnonymisation(
                    matched_span_precision,
                    matched_span_recall,
                    classification_head=head,
                )(lenient_level=lenient_level)[["P", "R", "F1", "TP_Precision", "TP_Recall", "FP", "FN", "Support"]]
                categorical_metrics["Level"] =  head
                list_of_categorical_evaluations.append(categorical_metrics)
        categorical_eval_df = pd.concat(list_of_categorical_evaluations)
        categorical_eval_df["Label"] = categorical_eval_df.index
        categorical_eval_df.reset_index(drop=True, inplace=True)
        return pd.concat([spans_eval_df, categorical_eval_df]).reset_index(drop=True)
    else:
        return spans_eval_df