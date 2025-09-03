from clueval.data import Convert

# import pytest


def test_converter(p1):

    converter = Convert(p1)
    print(converter)
    df1 = converter(tag_column=1, tag_name="layer1", prefix="reference")
    df2 = converter(tag_column=2, tag_name="layer2", prefix="reference")
    df3 = converter(tag_column=3, tag_name="layer3", prefix="reference")


    # test not empty
    assert df1.shape[0] != 0
    assert df2.shape[0] != 0
    assert df3.shape[0] != 0

    # test required columns
    df1_required_columns = ['start', 'end', 'layer1', 'domain', 'doc_id', 'text', 'id']
    assert df1.columns.tolist() == df1_required_columns

    df2_required_columns = ['start', 'end', 'layer2', 'domain', 'doc_id', 'text', 'id']
    assert df2.columns.tolist() == df2_required_columns

    df3_required_columns = ['start', 'end', 'layer3', 'domain', 'doc_id', 'text', 'id']
    assert df3.columns.tolist() == df3_required_columns

    # test id prefix
    assert df1["id"][0].startswith("reference")