import pytest


@pytest.fixture
def p1():
    """ annotation 1 """
    return "tests/data/reference.bio"


@pytest.fixture
def p2():
    """ annotation 1 """
    return "tests/data/candidate.bio"
