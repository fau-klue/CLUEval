from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Metrics(ABC):
    """ Abstract class with methods for computing classification metrics."""
    @abstractmethod
    def __call__(self, lenient: bool = False):
        pass

    @staticmethod
    def precision(true_positives: int, pre_denominator: int):
        """ Compute precision scores. TP / TP + FP"""
        return round(100 * true_positives / pre_denominator, 5)

    @staticmethod
    def recall(true_positives: int, recall_denominator: int):
        """ Compute recall scores. TP / TP + FN"""
        return round(100 * true_positives / recall_denominator, 5)

    @staticmethod
    def f1(precision, recall):
        """ Compute F1. 2*P*R / (P+R)"""
        return round(2 * precision * recall / (precision + recall), 5)


class MetricsForSpansAnonymisation(Metrics):
    def __init__(self, spans_df: pd.DataFrame):
        self.spans_df = spans_df.loc[~((spans_df.start.isna()) & (spans_df.status == "overlap"))]

    def __call__(self, lenient: bool = False, row_name: str = None):
        """
        Compute evaluation metrics and return a dataframe containing following information:
        Precision, Recall, F1, False Neg., False Pos., Support and Row Name
        :param lenient: Boolean indication for relaxed spans evaluation. Default: False -> Compute metrics for exact match
        :param row_name: Row name as index
        """
        metrics = dict(P=0.0,
                       R=0.0,
                       F1=0.0,
                       FN=0,
                       FP=0,
                       support=0,
                       row_name=row_name
                       )

        n_left_spans = self.spans_df[self.spans_df.status != "FP"]["status"].count()
        n_right_spans = self.spans_df[self.spans_df.status != "FN"]["status"].count()
        # Lenient will also accept overlapped spans for computing Precision and Recall
        if lenient:
            # True positives cases for Precision: TPs + gold annotation that are longer than predictions
            tp_precision = self.spans_df[self.spans_df.status.str.contains("TP|super")]["status"].count()
            # True positives cases for Recall: TPs + predictions that are longer than reference
            tp_recall = self.spans_df[self.spans_df.status.str.contains("TP|sub")]["status"].count()
            # Compute metrics
            metrics["P"] = self.precision(tp_precision, n_right_spans)
            metrics["R"] = self.recall(tp_recall, n_left_spans)
            metrics["F1"] = self.f1(metrics["P"], metrics["R"])
            metrics["FN"] = n_left_spans - tp_recall
            metrics["FP"] = n_right_spans - tp_precision
        else:
            tp = self.spans_df[self.spans_df.status == "TP"]["status"].count()
            metrics["P"] = self.precision(tp, n_right_spans)
            metrics["R"] = self.recall(tp, n_left_spans)
            metrics["F1"] = self.f1(metrics["P"], metrics["R"])
            metrics["FN"] = n_left_spans - tp
            metrics["FP"] = n_right_spans - tp
        metrics["support"] = self.spans_df["text"].count()
        return pd.DataFrame(metrics, index=[row_name])


class MetricsForCategoricalSpansAnonymisation(Metrics):
    def __init__(self, spans_df: pd.DataFrame, column: str = "cat", suffix: str = "_Y"):
        """
        Compute metrics for each information category or risk.
        :param spans_df: Spans dataframe containing annotations (left) and predictions (right).
        :param column: Entity class column
        :param suffix: Suffix to extract column from right hand side table.
        """
        self.spans_df = spans_df
        self.status = spans_df.status
        self.left_categories = self.spans_df[column]
        self.right_categories = self.spans_df[column + suffix]
        self.all_categories = sorted([cat for cat in pd.Series(self.left_categories.values,
                                                               self.right_categories.values).unique()
                                      if cat is not np.nan
                                      ]
                                     )
        self.left_categories = self.left_categories.fillna("")
        self.right_categories = self.right_categories.fillna("")
        self.exact_match_ids = self.left_categories == self.right_categories
        self.tp_ids = self.status == "TP"

    def __call__(self, lenient: bool = True):
        """
        :param lenient: Indicate if relaxed evaluation should be used. Default: True
        """
        if lenient:
            tp_precision_ids = self.tp_ids | (self.status == "super")
            tp_recall_ids = self.tp_ids | (self.status == "sub")
        else:
            tp_precision_ids = tp_recall_ids = self.tp_ids
        return pd.concat([self.compute_categorical_metrics(cat, tp_precision_ids, tp_recall_ids)
                          for cat in self.all_categories
                          ]
                         )

    def compute_categorical_metrics(self, category: str, tp_precision: int, tp_recall: int):
        """
        Method to compute classification metrics for given category.
        Get number of annotations and predictions for category. Check whether it is not 0. If not do:
            - determine whether given category is included in annotations or in predictions as sub_ids
            - compute TP for precision (tp_precision + sub_ids + exact matches)
            - compute TP for recall (tp_recall + sub_ids + exact matches)
            - compute precision, recall and f1
        :param category: Input category
        :param tp_precision: Pre-determined tp for precision
        :param tp_recall: Pre-determined tp for recall
        """
        metrics = dict(P=0.0,
                       R=0.0,
                       F1=0.0,
                       FN=0.0,
                       FP=0.0,
                       support=0,
                       row_name=category
                       )
        n_category_left = sum(self.left_categories == category)
        n_category_right = sum(self.right_categories == category)
        if not n_category_right == 0 and not n_category_left == 0:
            sub_ids = (self.left_categories == category) | (self.right_categories == category)
            tp_precision = sum(tp_precision & sub_ids & self.exact_match_ids)
            tp_recall = sum(tp_recall & sub_ids & self.exact_match_ids)
            metrics["P"] = self.precision(tp_precision, n_category_right)
            metrics["R"] = self.recall(tp_recall, n_category_left)
            # Return F1 = 0.0 if ZeroDivisionError occurs.
            # This is the case when we evaluate AG models on OLG data and do not have predictions for certain categories.
            try:
                metrics["F1"] = self.f1(metrics["P"], metrics["R"])
            except ZeroDivisionError:
                metrics["F1"] = 0.0
        metrics["FN"] = n_category_left - tp_recall
        metrics["FP"] = n_category_right - tp_precision
        metrics["support"] = n_category_left
        return pd.DataFrame(metrics, index=[category])

    def annotate(self, tp_recall_ids):
        # TODO: Refactor this part?
        errors = pd.Series(["" for _ in range(self.status.shape[0])])
        partial_match_ids = self.status.isin(["sub", "super", "overlap"])
        errors.iloc[tp_recall_ids & self.exact_match_ids] = "TP"
        errors.iloc[tp_recall_ids & ~self.exact_match_ids] = "wrong"
        errors.iloc[self.status == "FP"] = "FP"
        errors.iloc[self.status == "FN"] = "FN"
        errors.iloc[partial_match_ids & ~tp_recall_ids & self.exact_match_ids] = "partial"
        errors.iloc[partial_match_ids & ~tp_recall_ids & ~self.exact_match_ids] = "partwrong"
        self.spans_df["error"] = errors
        return self.spans_df
