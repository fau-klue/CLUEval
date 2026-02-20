import pytest


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
