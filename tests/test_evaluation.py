import pandas as pd
from clueval.evaluation import main

def test_evaluate(p1, p2):
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence")
    main(p1, p2, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")

def test_evaluate_same_file(p1):
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence")
    main(p1, p1, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")

def test_span_evaluation(p1, p2):
    # test exact metrics
    *_, span_evaluation = main(p1, p2, annotation_layer="confidence")
    assert isinstance(span_evaluation, pd.DataFrame) and not span_evaluation.empty
    assert span_evaluation["P"].values.round(4) == 13.6364
    assert span_evaluation["R"].values.round(4) == 27.2727

    # test lenient level 1
    *_, span_evaluation = main(p1, p2, annotation_layer="confidence", lenient_level=1)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values.round(4) == 81.8182
    assert span_evaluation["R"].values.round(4) == 45.4546

    # test lenient level 2
    *_, span_evaluation = main(p1, p2, annotation_layer="confidence", lenient_level=2)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values.round(4) == 81.8182
    assert span_evaluation["R"].values.round(4) == 54.5454

    # test lenient level 2
    *_, span_evaluation = main(p1, p2, annotation_layer="confidence", lenient_level=3)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values.round(4) == 81.8182
    assert span_evaluation["R"].values.round(4) == 54.5454

    # test span support
    assert span_evaluation["Support"].values[0] == 11

    # test recall TP for lenient level 3
    assert span_evaluation["TP_Recall"].values[0] == 6

def test_categorical_evaluation(p1, p2):
    # test exact metrics
    *_, categorical_evaluation = main(p1, p2, annotation_layer="confidence", categorical_evaluation=True,  categorical_head="confidence")
    assert isinstance(categorical_evaluation, pd.DataFrame) and not categorical_evaluation.empty
    assert "Label" in categorical_evaluation.columns.tolist()
    assert categorical_evaluation["Label"].tolist() == ["Span", "Hoch", "Mittel", "Niedrig"]
    # Support
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["Support"].values == 7
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["Support"].values == 2
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["Support"].values  == 2

    # test number of TP_Recall / FN
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["TP_Recall"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["FN"].values == 6
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["TP_Recall"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["FN"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["TP_Recall"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["FN"].values == 1

    # test number of TP_Precision / FP for the category 'Hoch'
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["TP_Precision"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["FP"].values == 11
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["TP_Precision"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["FP"].values == 5
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["TP_Precision"].values == 1
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["FP"].values == 3