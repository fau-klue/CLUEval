import pandas as pd
from clueval.evaluation import evaluate


def test_evaluate(p1, p2):
    evaluate(p1, p2, annotation_layer="confidence")
    evaluate(p1, p2, annotation_layer="confidence")
    evaluate(p1, p2, annotation_layer="confidence")
    evaluate(p1, p2, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")


def test_evaluate_same_file(p1):
    evaluate(p1, p1, annotation_layer="confidence")
    evaluate(p1, p1, annotation_layer="confidence")
    evaluate(p1, p1, annotation_layer="confidence")
    evaluate(p1, p1, annotation_layer="confidence", categorical_evaluation=True, categorical_head="confidence")


def test_span_evaluation(p1, p2, precision_table, recall_table):
    # test exact metrics
    *_, span_evaluation = evaluate(p1, p2, annotation_layer="confidence")
    assert isinstance(span_evaluation, pd.DataFrame) and not span_evaluation.empty
    assert span_evaluation["P"].values == round(precision_table.loc[precision_table["status"] == "exact"].shape[0] / precision_table.shape[0] * 100, 4)
    assert span_evaluation["R"].values == round(recall_table.loc[recall_table["status"] == "exact"].shape[0] / recall_table.shape[0] * 100, 4)


    # test lenient level 1
    *_, span_evaluation = evaluate(p1, p2, annotation_layer="confidence", lenient_level=1)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(precision_table.loc[precision_table["status"].isin(["exact", "contained"])].shape[0] / precision_table.shape[0] * 100, 4)
    assert span_evaluation["R"].values == round(recall_table.loc[recall_table["status"].isin(["exact", "contained"])].shape[0] / recall_table.shape[0] * 100, 4)

    # test lenient level 2
    *_, span_evaluation = evaluate(p1, p2, annotation_layer="confidence", lenient_level=2)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(precision_table.loc[precision_table["status"].isin(["exact", "contained", "tiled"])].shape[0] / precision_table.shape[0] * 100, 4)
    assert span_evaluation["R"].values == round(recall_table.loc[recall_table["status"].isin(["exact", "contained", "tiled"])].shape[0] / recall_table.shape[0] * 100, 4)

    # test lenient level 2
    *_, span_evaluation = evaluate(p1, p2, annotation_layer="confidence", lenient_level=3)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(precision_table.loc[precision_table["status"].isin(["exact", "contained", "tiled", "covered"])].shape[0] / precision_table.shape[0] * 100, 4)
    assert span_evaluation["R"].values == round(recall_table.loc[recall_table["status"].isin(["exact", "contained", "tiled", "covered"])].shape[0] / recall_table.shape[0] * 100, 4)

    # test span support
    assert span_evaluation["Support"].values[0] == recall_table.shape[0]

    # test recall TP for lenient level 3
    assert span_evaluation["TP_Recall"].values[0] == recall_table[recall_table["status"].isin(["exact", "contained", "tiled", "covered"])].shape[0]


def test_span_evaluation_short(p1s, p2s):
    # test exact metrics
    *_, span_evaluation = evaluate(p1s, p2s, annotation_layer="confidence")
    assert isinstance(span_evaluation, pd.DataFrame) and not span_evaluation.empty
    assert span_evaluation["P"].values == round(2/19 * 100, 4)
    assert span_evaluation["R"].values == round(2/9 * 100, 4)

    # test lenient level 1
    *_, span_evaluation = evaluate(p1s, p2s, annotation_layer="confidence", lenient_level=1)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(15/19 * 100, 4)
    assert span_evaluation["R"].values == round(4/9 * 100, 4)

    # test lenient level 2
    *_, span_evaluation = evaluate(p1s, p2s, annotation_layer="confidence", lenient_level=2)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(15/19 * 100, 4)
    assert span_evaluation["R"].values == round(5/9 * 100, 4)

    # test lenient level 2
    *_, span_evaluation = evaluate(p1s, p2s, annotation_layer="confidence", lenient_level=3)
    assert isinstance(span_evaluation, pd.DataFrame)
    assert span_evaluation["P"].values == round(15/19 * 100, 4)
    assert span_evaluation["R"].values == round(6/9 * 100, 4)

    # test span support
    assert span_evaluation["Support"].values[0] == 9


def test_categorical_evaluation(p1, p2, precision_table, recall_table):
    # test exact metrics
    *_, categorical_evaluation = evaluate(p1, p2, annotation_layer="confidence", categorical_evaluation=True,  categorical_head="confidence")
    assert isinstance(categorical_evaluation, pd.DataFrame) and not categorical_evaluation.empty
    assert "Label" in categorical_evaluation.columns.tolist()
    assert categorical_evaluation["Label"].tolist() == ["Span", "Hoch", "Mittel", "Niedrig"]
    # Support
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["Support"].values == recall_table.loc[recall_table["Risk"] == "hoch"].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["Support"].values == recall_table.loc[recall_table["Risk"] == "mittel"].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["Support"].values == recall_table.loc[recall_table["Risk"] == "niedrig"].shape[0]

    # test number of TP_Recall / FN
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["TP_Recall"].values == recall_table.loc[(recall_table["Risk"] == "hoch") & (recall_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["FN"].values == recall_table.loc[(recall_table["Risk"] == "hoch") & (recall_table["status"] != "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["TP_Recall"].values == recall_table.loc[(recall_table["Risk"] == "mittel") & (recall_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["FN"].values == recall_table.loc[(recall_table["Risk"] == "mittel") & (recall_table["status"] != "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["TP_Recall"].values == recall_table.loc[(recall_table["Risk"] == "niedrig") & (recall_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["FN"].values == recall_table.loc[(recall_table["Risk"] == "niedrig") & (recall_table["status"] != "exact")].shape[0]

    # test number of TP_Precision / FP
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["TP_Precision"].values == precision_table.loc[(precision_table["Risk"] == "hoch") & (precision_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Hoch"]["FP"].values == precision_table.loc[(precision_table["Risk"] == "hoch") & (precision_table["status"] != "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["TP_Precision"].values == precision_table.loc[(precision_table["Risk"] == "mittel") & (precision_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Mittel"]["FP"].values == precision_table.loc[(precision_table["Risk"] == "mittel") & (precision_table["status"] != "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["TP_Precision"].values == precision_table.loc[(precision_table["Risk"] == "niedrig") & (precision_table["status"] == "exact")].shape[0]
    assert categorical_evaluation[categorical_evaluation["Label"] == "Niedrig"]["FP"].values == precision_table.loc[(precision_table["Risk"] == "niedrig") & (precision_table["status"] != "exact")].shape[0]
