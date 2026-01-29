from clueval.spans_table import Convert

# import pytest

def test_converter(p1):

    converter = Convert(p1, annotation_layer=["head_0", "head_1", "head_2"])
    print(converter)
    df = converter(id_prefix="reference")
   
    # test not empty
    assert df.shape[0] != 0
    # test required columns
    required_columns = ["start", "end", "doc_token_id_start" , "doc_token_id_end", "text" , "doc_id", "domain", "head_0", "head_1", "head_2", "id"]

    assert sorted(df.columns.tolist()) == sorted(required_columns)

     # test id prefix
    assert df1["id"][0].startswith("reference")
