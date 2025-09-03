#!/usr/bin/env python3

import pandas as pd
from ..data import Convert, Join, JoinAnnotationLayers
from .metrics import MetricsForCategoricalSpansAnonymisation, MetricsForSpansAnonymisation


def evaluate(path_reference, path_candidate):
    # Convert BIO to spans tables
    # Reference
    convert_ref_spans = Convert(path_reference)
    ref_anon_df = convert_ref_spans(tag_column=1, tag_name="anon", prefix="anon")
    ref_entity_df = convert_ref_spans(tag_column=2, tag_name="cat", prefix="cat")
    ref_risk_df = convert_ref_spans(tag_column=3, tag_name="risk", prefix="risk")
    reference_df = JoinAnnotationLayers(
            JoinAnnotationLayers(ref_anon_df, ref_entity_df)(lcat="anon", rcat="cat", on=["start", "end"]),
            ref_risk_df)(lcat="anon", rcat="risk", on=["start", "end"])
    reference_df["id"] = reference_df["id"].str.replace("anon", "gold")

    # Candidate
    convert_cand_spans = Convert(path_candidate)
    cand_anon_df = convert_cand_spans(tag_column=1, tag_name="anon", prefix="anon")
    cand_entity_df = convert_cand_spans(tag_column=2, tag_name="cat", prefix="cat")
    cand_risk_df = convert_cand_spans(tag_column=3, tag_name="risk", prefix="risk")
    candidate_df = JoinAnnotationLayers(
            JoinAnnotationLayers(cand_anon_df, cand_entity_df)(lcat="anon", rcat="cat", on=["start", "end"]),
            cand_risk_df)(lcat="anon", rcat="risk", on=["start", "end"])

    # Join reference with candidate dataframes
    anon_df = Join(reference_df, cand_anon_df)(on=["start", "end"])
    entity_df = Join(reference_df, cand_entity_df)(on=["start", "end"])
    risk_df = Join(reference_df, cand_risk_df)(on=["start", "end"])

    reference_cand_df = Join(reference_df, candidate_df)(on=["start", "end"])
    reference_high_risk = Join(reference_df.loc[reference_df.risk == "hoch"], candidate_df)(on=["start", "end"])
    # Evaluation
    spans_evaluation = pd.concat([MetricsForSpansAnonymisation(anon_df)(lenient=True, row_name="Anon"),
                                      MetricsForSpansAnonymisation(entity_df)(lenient=True, row_name="Entity"),
                                      MetricsForSpansAnonymisation(risk_df)(lenient=True, row_name="Risk"),
                                      MetricsForSpansAnonymisation(reference_cand_df)(lenient=True, row_name="All"),
                                      MetricsForSpansAnonymisation(reference_high_risk)(lenient=True, row_name="High risk")
                                      ])[["P", "R", "F1", "FN", "FP", "support"]].reset_index().rename(
            columns={"index": "Span", "support": "Support"})
    spans_evaluation['Level'] = 'span'
    spans_evaluation.rename(columns={'Span': 'Value'}, inplace=True)

    entity_spans_evaluation = MetricsForCategoricalSpansAnonymisation(
            entity_df, column="cat")()[["P", "R", "F1", "support"]].reset_index().rename(
            columns={"index": "Entity", "support": "Support"})
    entity_spans_evaluation['Level'] = 'entity'
    entity_spans_evaluation.rename(columns={'Entity': 'Value'}, inplace=True)

    risk_spans_evaluation = MetricsForCategoricalSpansAnonymisation(
            risk_df, column="risk")()[["P", "R", "F1", "support"]].reset_index().rename(
            columns={"index": "Risk", "support": "Support"})
    risk_spans_evaluation['Level'] = 'risk'
    risk_spans_evaluation.rename(columns={'Risk': 'Value'}, inplace=True)

    df = pd.concat([spans_evaluation, entity_spans_evaluation, risk_spans_evaluation])

    return df
