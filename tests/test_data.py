from clueval.data import Convert

# import pytest


def test_converter(p1):

    converter = Convert(p1)
    dataframe = converter(prefix="gold")

    # test not empty
    assert dataframe.shape[0] != 0

    # test required columns
    required_columns = ['start', 'end', 'cat', 'risk', 'domain', 'set', 'verdict', 'text', 'id']
    assert dataframe.columns.tolist() == required_columns

    # test id prefix
    assert dataframe["id"][0].startswith("gold")
