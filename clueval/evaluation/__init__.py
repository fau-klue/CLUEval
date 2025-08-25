#!/usr/bin/env python3

import pandas as pd

from ..data import Convert, Join, JoinMultiAnnotations
from .metrics import (MetricsForCategoricalSpansAnonymisation,
                      MetricsForSpansAnonymisation)


def evaluate(path_reference, path_candidate):

    # Convert BIO to spans tables
    convert_ref_spans = Convert(path_reference)
    reference_df = convert_ref_spans(prefix="gold")

    convert_cand_spans = Convert(path_candidate)
    anon_df = convert_cand_spans(gold=False, tag_column=1, prefix="anon")
    entity_df = convert_cand_spans(gold=False, tag_column=2, prefix="entity")
    risk_df = convert_cand_spans(gold=False, tag_column=3, prefix="risk")

    # Join span tables
    reference_anon_df = Join(reference_df, anon_df)(on=["start", "end"])
    reference_entity_df = Join(reference_df, entity_df)(on=["start", "end"])
    reference_risk_df = Join(reference_df, risk_df)(on=["start", "end"])

    # Multitask Join
    all_df = JoinMultiAnnotations(anon_df, entity_df, risk_df)(on=["start", "end"])
    reference_all = Join(reference_df, all_df)(on=["start", "end"])
    reference_high_risk = Join(reference_df.loc[reference_df.risk == "hoch"], all_df)(on=["start", "end"])

    spans_evaluation = pd.concat([MetricsForSpansAnonymisation(reference_anon_df)(lenient=True, row_name="Anon"),
                                  MetricsForSpansAnonymisation(reference_entity_df)(lenient=True, row_name="Entity"),
                                  MetricsForSpansAnonymisation(reference_risk_df)(lenient=True, row_name="Risk"),
                                  MetricsForSpansAnonymisation(reference_all)(lenient=True, row_name="All"),
                                  MetricsForSpansAnonymisation(reference_high_risk)(lenient=True, row_name="High risk")
                                  ])[["P", "R", "F1", "FN", "FP", "support"]].reset_index().rename(columns={"index": "Span", "support": "Support"})
    spans_evaluation['Level'] = 'span'
    spans_evaluation.rename(columns={'Span': 'Value'}, inplace=True)

    entity_spans_evaluation = MetricsForCategoricalSpansAnonymisation(
        reference_entity_df, column="cat")()[["P", "R", "F1", "support"]].reset_index().rename(columns={"index": "Entity", "support": "Support"})
    entity_spans_evaluation['Level'] = 'entity'
    entity_spans_evaluation.rename(columns={'Entity': 'Value'}, inplace=True)

    risk_spans_evaluation = MetricsForCategoricalSpansAnonymisation(
        reference_risk_df, column="risk")()[["P", "R", "F1", "support"]].reset_index().rename(columns={"index": "Risk", "support": "Support"})
    risk_spans_evaluation['Level'] = 'risk'
    risk_spans_evaluation.rename(columns={'Risk': 'Value'}, inplace=True)

    df = pd.concat([spans_evaluation, entity_spans_evaluation, risk_spans_evaluation])

    return df
