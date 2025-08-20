from clueval.data import Convert 
import pytest

converter = Convert("./test_data/fiktives-urteil-p1.bio")
dataframe = converter(prefix="gold")

def test_not_empty_dataframe():
    assert dataframe.shape[0] != 0

def test_required_columns():
    required_columns = ['start', 'end', 'cat', 'risk', 'domain', 'set', 'verdict', 'text', 'id']
    assert dataframe.columns.tolist() == required_columns

def test_id_prefix():
    assert dataframe["id"][0].startswith("gold")