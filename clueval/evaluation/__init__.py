#!/usr/bin/env python3

import pandas as pd
from ..spans_table import BioToSpanParser, Join#, JoinAnnotationLayers
from .metrics import MetricsForCategoricalSpansAnonymisation, MetricsForSpansAnonymisation

def evaluate(path_reference:str,
             path_candidate:str,
             annotation_layer:str | list[str],
             domain_column: int | None=None,
             filter_column: str | None = None,
             filter_value: str | None = None,
             categorical_evaluation: bool = False
             ):
    if not annotation_layer:
        raise ValueError("No input for annotation_layer")
    if type(annotation_layer) == str:
        tag_name = annotation_layer
    else:
        tag_name = annotation_layer[0]


    # Convert BIO to spans tables
    # Reference
    convert_ref_spans = Convert(path_reference)
    # Candidate
    convert_cand_spans = Convert(path_candidate)
    reference_df = convert_ref_spans(tag_column=1,
                                     tag_name=tag_name,
                                     prefix=tag_name,
                                     domain_column=domain_column
                                     )
    candidate_df = convert_cand_spans(tag_column=1,
                                      tag_name=tag_name,
                                      prefix=tag_name,
                                      domain_column=domain_column
                                      )

    # Store candidate dataframes for later joins with reference_df
    candidate_dfs = {f"{tag_name}": candidate_df}
    # Join different layers into one dataframe
    if type(annotation_layer) == list:
        for i in range(1, len(annotation_layer)):
            # Reference
            ref_layer_df = convert_ref_spans(tag_column=i+1,
                                             tag_name=annotation_layer[i],
                                             prefix=annotation_layer[i]
                                             )
            reference_df = JoinAnnotationLayers(reference_df, ref_layer_df)(lcat=annotation_layer[0],
                                                                        rcat=annotation_layer[i],
                                                                        on=["start", "end"]
                                                                        )
            # Candidate
            cand_layer_df = convert_cand_spans(tag_column=i+1,
                                               tag_name=annotation_layer[i],
                                               prefix=annotation_layer[i]
                                               )
            candidate_df = JoinAnnotationLayers(candidate_df, cand_layer_df)(lcat=annotation_layer[0],
                                                                        rcat=annotation_layer[i],
                                                                        on=["start", "end"]
                                                                        )
            candidate_dfs.update({f"{annotation_layer[i]}": cand_layer_df})

    reference_df["id"] = reference_df["id"].str.replace(annotation_layer[0], "gold")
    # Join reference_df with candidate_df to compute result for all spans
    reference_cand_df = Join(reference_df, candidate_df)(on=["start", "end"])

    # Join reference with candidate dataframes and compute spans metrics
    list_of_spans_evaluations = [MetricsForSpansAnonymisation(Join(reference_df, cand_df)(on=["start", "end"]))(lenient=True, row_name=layer.capitalize()) for layer, cand_df in candidate_dfs.items()]
    list_of_spans_evaluations.append(MetricsForSpansAnonymisation(reference_cand_df)(lenient=True, row_name="All"))
    # Compute span by filtered value
    if filter_column:
        if not filter_value:
            raise ValueError(f"Can not filter {filter_column} by None")
        reference_cand_filter = Join(reference_df.loc[reference_df[filter_column] == filter_value], candidate_df)(
        on=["start", "end"])
        list_of_spans_evaluations.append(MetricsForSpansAnonymisation(reference_cand_filter)(lenient=True, row_name=filter_value.capitalize()))

    # Evaluation
    spans_eval_df = pd.concat(list_of_spans_evaluations)[["P", "R", "F1", "FN", "FP", "support"]].reset_index().rename(
            columns={"index": "Span", "support": "Support"})
    spans_eval_df['Level'] = "Span"
    spans_eval_df.rename(columns={"Span": "Value"}, inplace=True)

    # Compute metrics for categorical spans
    if categorical_evaluation:
        list_of_categorical_evaluations = []
        for layer, cand_df in candidate_dfs.items():
            categorical_df =  MetricsForCategoricalSpansAnonymisation(Join(reference_df, cand_df)(on=["start", "end"]),
                                                    column=layer)()[["P", "R", "F1", "support", "FN", "FP"]].reset_index().rename(
                columns={"index": f"{layer.capitalize()}", "support": "Support"}
            )
            categorical_df["Level"] = layer.capitalize()
            categorical_df.rename(columns={f"{layer.capitalize()}": "Value"}, inplace=True)
            list_of_categorical_evaluations.append(categorical_df)
        categorical_eval_df = pd.concat(list_of_categorical_evaluations)
        return pd.concat([spans_eval_df, categorical_eval_df])
    else:
        return spans_eval_df