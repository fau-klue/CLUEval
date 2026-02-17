# tests/data/edge_cases.py
# DataFrame fixtures for testing the 5 match classes from algorithms.md
import pytest
import pandas as pd

# Column schema for spans DataFrames
COLUMNS = ["start", "end", "doc_token_id_start", "doc_token_id_end", "text", 
           "doc_id", "domain", "head_0", "head_1", "head_2", "id"]


@pytest.fixture
def exact_match_ref():
    """Class 1: Reference spans for exact match test"""
    return pd.DataFrame([
        [0, 1, "t0", "t1", "token1 token2", "doc1", "test", "anon", "type", "low", "id1"],
        [5, 7, "t5", "t7", "andere Wörter", "doc1", "test", "anon", "name", "high", "id2"],
    ], columns=COLUMNS)


@pytest.fixture
def exact_match_cand(exact_match_ref):
    """Class 1: Candidate spans identical to reference"""
    return exact_match_ref.copy()


@pytest.fixture
def superset_ref():
    """Class 2: Gold has smaller span (token2-3 only)"""
    return pd.DataFrame([
        [2, 3, "t2", "t3", "token2 token3", "doc1", "test", "anon", "type", "low", "id1"],
    ], columns=COLUMNS)


@pytest.fixture
def superset_cand():
    """Class 2: Predicted span is superset (token1-4)"""
    return pd.DataFrame([
        [1, 4, "t1", "t4", "token1 token2 token3 token4", "doc1", "test", "anon", "type", "low", "id1"],
    ], columns=COLUMNS)


@pytest.fixture
def tiling_ref():
    """Class 3: Gold has one continuous span (token0-3)"""
    return pd.DataFrame([
        [0, 3, "t0", "t3", "token1 token2 token3 token4", "doc1", "test", "anon", "type", "low", "id1"],
    ], columns=COLUMNS)


@pytest.fixture
def tiling_cand():
    """Class 3: Two adjacent spans that tile the gold span"""
    return pd.DataFrame([
        [0, 1, "t0", "t1", "token1 token2", "doc1", "test", "anon", "type", "low", "id1"],
        [2, 3, "t2", "t3", "token3 token4", "doc1", "test", "anon", "type", "low", "id2"],
    ], columns=COLUMNS)


@pytest.fixture
def overlap_ref():
    """Class 4: Gold span (token2-4)"""
    return pd.DataFrame([
        [2, 4, "t2", "t4", "token3 token4 token5", "doc1", "test", "anon", "type", "low", "id1"],
    ], columns=COLUMNS)


@pytest.fixture
def overlap_cand():
    """Class 4: Adjacent spans that extend beyond gold"""
    return pd.DataFrame([
        [1, 3, "t1", "t3", "token2 token3 token4", "doc1", "test", "anon", "type", "low", "id1"],
        [4, 5, "t4", "t5", "token5 token6", "doc1", "test", "anon", "type", "low", "id2"],
    ], columns=COLUMNS)


@pytest.fixture
def mismatch_ref():
    """Class 5: Gold has continuous span (token0-3)"""
    return pd.DataFrame([
        [0, 3, "t0", "t3", "token1 token2 token3 token4", "doc1", "test", "anon", "type", "low", "id1"],
    ], columns=COLUMNS)


@pytest.fixture
def mismatch_cand():
    """Class 5: Non-adjacent spans with gap → should be FN"""
    return pd.DataFrame([
        [0, 0, "t0", "t0", "token1", "doc1", "test", "anon", "type", "low", "id1"],
        [3, 3, "t3", "t3", "token4", "doc1", "test", "anon", "type", "low", "id2"],
    ], columns=COLUMNS)


@pytest.fixture
def empty_ref():
    """Edge case: Empty reference (no spans)"""
    return pd.DataFrame(columns=COLUMNS)


@pytest.fixture
def empty_cand():
    """Edge case: Empty candidate (no spans)"""
    return pd.DataFrame(columns=COLUMNS)