from clueval.spans_table import Match
from clueval.evaluation import MetricsForSpansAnonymisation
import pytest

# Import fixtures from edge_cases.py
pytest_plugins = ["tests.data.edge_cases"]


class TestMatchClasses:
    """Test the 5 classification cases from algorithms.md"""

    def test_exact_match(self, exact_match_ref, exact_match_cand):
        """Class 1: Exact match should have status='exact' and perfect metrics"""
        recall_match = Match(exact_match_ref, exact_match_cand)
        precision_match = Match(exact_match_cand, exact_match_ref)
        
        matched_recall = recall_match(on=["start", "end"])
        matched_precision = precision_match(on=["start", "end"])
        
        # All spans should have status 'exact'
        assert all(matched_recall["status"] == "exact")
        assert all(matched_precision["status"] == "exact")
        
        # Metrics should be perfect (100% = 100.0)
        metrics = MetricsForSpansAnonymisation(matched_recall, matched_precision)
        result = metrics(lenient_level=0, row_name="Exact")
        assert result.loc["Exact", "P"] == 100.0
        assert result.loc["Exact", "R"] == 100.0
        assert result.loc["Exact", "F1"] == 100.0

    def test_superset_match(self, superset_ref, superset_cand):
        """Class 2: Superset - predicted span contains entire gold span"""
        recall_match = Match(superset_ref, superset_cand)
        matched_recall = recall_match(on=["start", "end"])
        
        # Gold span should be matched as 'subset' (contained in larger P span)
        assert matched_recall.iloc[0]["status"] == "subset"

    def test_tiling_match(self, tiling_ref, tiling_cand):
        """Class 3: Multiple adjacent P spans exactly tile G span"""
        recall_match = Match(tiling_ref, tiling_cand)
        matched_recall = recall_match(on=["start", "end"])
        
        # Gold span should be matched as 'tiling'
        assert matched_recall.iloc[0]["status"] == "tiling"

    def test_overlap_match(self, overlap_ref, overlap_cand):
        """Class 4: Adjacent P spans extend beyond G but cover it"""
        recall_match = Match(overlap_ref, overlap_cand)
        matched_recall = recall_match(on=["start", "end"])
        
        # Gold span should be matched as 'overlap' (covered by adjacent spans)
        assert matched_recall.iloc[0]["status"] == "overlap"

    def test_mismatch_is_fn(self, mismatch_ref, mismatch_cand):
        """Class 5: Non-adjacent partial overlap should be FN (mismatch)"""
        recall_match = Match(mismatch_ref, mismatch_cand)
        matched_recall = recall_match(on=["start", "end"])
        
        # Gold span should be 'mismatch' (FN) due to gap in P spans
        assert matched_recall.iloc[0]["status"] == "mismatch"

    def test_empty_spans(self, empty_ref, empty_cand):
        """Edge case: No spans in either reference or candidate"""
        recall_match = Match(empty_ref, empty_cand)
        matched_recall = recall_match(on=["start", "end"])
        
        # Should return empty DataFrame
        assert matched_recall.empty


class TestLenientLevels:
    """Test different lenient_level settings affect metrics
    
    Lenient levels from metrics.py:
    - 0: ["exact"]
    - 1: ["exact", "subset"]
    - 2: ["exact", "subset", "tiling"]
    - 3: ["exact", "subset", "tiling", "overlap"]
    """

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_lenient_levels_tiling(self, tiling_ref, tiling_cand, level):
        """Tiling should count as TP at lenient levels >= 2"""
        recall_match = Match(tiling_ref, tiling_cand)
        precision_match = Match(tiling_cand, tiling_ref)
        
        matched_recall = recall_match(on=["start", "end"])
        matched_precision = precision_match(on=["start", "end"])
        
        metrics = MetricsForSpansAnonymisation(matched_recall, matched_precision)
        result = metrics(lenient_level=level, row_name="Test")
        
        # At level >= 2, tiling should be counted as TP (100%)
        if level >= 2:
            assert result.loc["Test", "R"] == 100.0
        else:
            # At levels 0-1, tiling is not counted as TP
            assert result.loc["Test", "R"] == 0.0

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_lenient_levels_superset(self, superset_ref, superset_cand, level):
        """Superset (subset status) should count as TP at lenient levels >= 1"""
        recall_match = Match(superset_ref, superset_cand)
        precision_match = Match(superset_cand, superset_ref)
        
        matched_recall = recall_match(on=["start", "end"])
        matched_precision = precision_match(on=["start", "end"])
        
        metrics = MetricsForSpansAnonymisation(matched_recall, matched_precision)
        result = metrics(lenient_level=level, row_name="Test")
        
        # At level >= 1, subset should be counted as TP for recall (100%)
        if level >= 1:
            assert result.loc["Test", "R"] == 100.0
        else:
            # At level 0, only exact matches count
            assert result.loc["Test", "R"] == 0.0