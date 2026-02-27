import pytest
import pandas as pd


@pytest.fixture
def p1():
    """ annotation 1 """
    return "tests/data/reference.bio"


@pytest.fixture
def p2():
    """ annotation 1 """
    return "tests/data/candidate.bio"


@pytest.fixture
def p1s():
    """ annotation 1 """
    return "tests/data/reference-short.bio"


@pytest.fixture
def p2s():
    """ annotation 1 """
    return "tests/data/candidate-short.bio"

@pytest.fixture
def precision_table():
    """ prepared precision table """
    return pd.read_csv("tests/data/precision_table.tsv", sep="\t")

@pytest.fixture
def recall_table():
    """ prepared recall table """
    return pd.read_csv("tests/data/recall_table.tsv", sep="\t")
