from clueval.spans_table import Convert, Match
# import pytest

def test_converter(p1):

    converter = Convert(p1, annotation_layer=["confidence"], token_id_column=2, doc_id_column=3, domain_column=4)
    print(converter)
    df = converter(id_prefix="reference")
   
    # test not empty
    assert df.shape[0] != 0

    # test number of spans
    assert df.shape[0] == 11

    # test required columns
    required_columns = ["start", "end", "token_id_start" , "token_id_end", "text" , "doc_id", "domain", "confidence", "id"]
    assert sorted(df.columns.tolist()) == sorted(required_columns)

    # test id prefix
    assert df["id"][0].startswith("reference")

    # test span start
    assert df["start"].values.tolist() == [2, 5, 10, 13, 18, 23, 29, 35, 46, 54, 75]

    # test span end
    assert df["end"].values.tolist() == [3, 9, 11, 17, 21, 28, 33, 39, 51, 56, 79]

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


def test_match(p1, p2):
    p1_converter = Convert(p1, annotation_layer=["confidence"])
    p2_converter = Convert(p2, annotation_layer=["confidence"])

    ref = p1_converter(id_prefix="ref")
    cand = p2_converter(id_prefix="cand")

    recall_match = Match(ref, cand, annotation_layer=["confidence"])(on=["start", "end"])[["start", "end", "confidence", "confidence_Y", "status"]]
    precision_match = Match(cand, ref, annotation_layer=["confidence"])(on=["start", "end"])[["start", "end", "confidence", "confidence_Y", "status"]]

    # test overlap cases in recall table
    assert recall_match["status"][0] == "subset"
    assert recall_match["status"][1] == "tiling"
    assert recall_match["status"][2] == "exact"
    assert recall_match["status"][3] == "unmatch"

    # test overlap cases in precision table
    assert precision_match["status"][0] == "unmatch"
    assert precision_match["status"][1] == "subset"
    assert precision_match["status"][3] == "exact"
    assert "tiling" not in precision_match["status"].unique()

    # compare confidence label
    assert recall_match["confidence"][0] == recall_match["confidence_Y"][0]
    assert recall_match["confidence"][1] == recall_match["confidence_Y"][1]
    # test label for longest overlap span
    assert recall_match["confidence_Y"][1] == "hoch"