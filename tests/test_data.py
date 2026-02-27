from clueval.spans_table import Convert, Match
# import pytest

# TODO: Revise test and integrate recall and precision.tsv from SE. Or in test_evaluation.

def test_converter(p1):

    converter = Convert(p1, annotation_layer=["confidence"], token_id_column=2, doc_id_column=3, domain_column=4)
    # print(converter)
    df = converter(id_prefix="reference")

    # test not empty
    assert df.shape[0] != 0

    # test number of spans
    assert df.shape[0] == 11

    # test required columns
    required_columns = ["start", "end", "token_id_start", "token_id_end", "text", "doc_id", "domain", "confidence", "id"]
    assert sorted(df.columns.tolist()) == sorted(required_columns)

    # test id prefix
    assert df["id"][0].startswith("reference")

    # test span start
    assert df["start"].values.tolist() == [1, 4, 9, 12, 17, 22, 28, 34, 45, 53, 74]

    # test span end
    assert df["end"].values.tolist() == [2, 8, 10, 16, 20, 27, 32, 38, 50, 55, 78]

    # test first span in reference
    assert df["text"][0] == "AMTSGERICHT ERLANGEN"

    # test last span in reference
    assert df["text"][10] == "Prof. Dr. Ing. Helmut Zöller"

    # test doc_id
    assert df["doc_id"].notna().all()
    assert df["doc_id"][0] == "fictitious_1512"
    assert df["doc_id"][1] == "fictitious_1512"

    # test token_id
    assert df["token_id_start"].notna().all()
    assert df["token_id_end"].notna().all()

    assert df["token_id_start"][0] == "token_1"
    assert df["token_id_end"][0] == "token_2"

    assert df["token_id_start"][10] == "token_74"
    assert df["token_id_end"][10] == "token_78"

    # test domain
    assert df["domain"].notna().all()
    assert len(df["domain"].unique()) == 1
    assert df["domain"].unique()[0] == "fictitious_domain"


def test_match(p1, p2, precision_table, recall_table):
    p1_converter = Convert(p1, annotation_layer=["confidence"],token_id_column=2, doc_id_column=3, domain_column=4)
    p2_converter = Convert(p2, annotation_layer=["confidence"], token_id_column=2, doc_id_column=3, domain_column=4)


    ref = p1_converter(id_prefix="ref")
    cand = p2_converter(id_prefix="cand")

    recall_match = Match(ref, cand, annotation_layer=["confidence"])(on=["start", "end"])
    precision_match = Match(cand, ref, annotation_layer=["confidence"])(on=["start", "end"])

    # test overlap cases in recall table
    assert recall_match["status"][0] == recall_table["status"][0]
    assert recall_match["status"][1] == recall_table["status"][1]
    assert recall_match["status"][2] == recall_table["status"][2]
    assert recall_match["status"][3] == recall_table["status"][3]

    assert recall_match["status"].value_counts()["exact"] == recall_table["status"].value_counts()["exact"]
    assert recall_match["status"].value_counts()["contained"] == recall_table["status"].value_counts()["contained"]
    assert recall_match["status"].value_counts()["tiled"] == recall_table["status"].value_counts()["tiled"]
    assert recall_match["status"].value_counts()["covered"] == recall_table["status"].value_counts()["covered"]
    assert recall_match["status"].value_counts()["unmatched"] == recall_table["status"].value_counts()["unmatched"]

    # test overlap cases in precision table
    assert precision_match["status"][0] == precision_table["status"][0]
    assert precision_match["status"][1] == precision_table["status"][1]
    assert precision_match["status"][3] == precision_table["status"][3]
    assert "tiled" not in precision_match["status"].unique().tolist()
    assert "covered" not in precision_match["status"].unique().tolist()

    assert precision_match["status"].value_counts()["exact"] == precision_table["status"].value_counts()["exact"]
    assert precision_match["status"].value_counts()["contained"] == precision_table["status"].value_counts()["contained"]
    assert precision_match["status"].value_counts()["unmatched"] == precision_table["status"].value_counts()["unmatched"]


    # compare confidence label
    assert recall_match["confidence"][0] == recall_table["Risk_Y"][0]
    assert recall_match["confidence"][1] == recall_table["Risk_Y"][1]

    # test label for longest overlap span
    assert recall_match["confidence_Y"][1] == "hoch"

    # test token id
    assert recall_match["token_id_start"][0] == recall_table["token_id_start"][0]
    assert recall_match["token_id_start"].iloc[-1] == recall_table["token_id_start"].iloc[-1]
    assert recall_match["token_id_end"][0] == recall_table["token_id_end"][0]
    assert recall_match["token_id_end"].iloc[-1] == recall_table["token_id_end"].iloc[-1]

    assert precision_match["token_id_start"][0] == precision_table["token_id_start"][0]
    assert precision_match["token_id_start"].iloc[-1] == precision_table["token_id_start"].iloc[-1]
    assert precision_match["token_id_end"][0] == precision_table["token_id_end"][0]
    assert precision_match["token_id_end"].iloc[-1] == precision_table["token_id_end"].iloc[-1]

