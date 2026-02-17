from abc import ABC, abstractmethod
import pandas as pd


class Metrics(ABC):
    """Abstract class with methods for computing classification metrics."""

    @abstractmethod
    def compute_metrics(self, lenient_level: int = 0, **kwargs):
        pass

    @staticmethod
    def precision(true_positives: int, pre_denominator: int):
        """Compute precision scores. TP / TP + FP"""
        return round(100 * true_positives / pre_denominator, 5)

    @staticmethod
    def recall(true_positives: int, recall_denominator: int):
        """Compute recall scores. TP / TP + FN"""
        return round(100 * true_positives / recall_denominator, 5)

    @staticmethod
    def f1(precision, recall):
        """Compute F1. 2*P*R / (P+R)"""
        return round(2 * precision * recall / (precision + recall), 5)


class MetricsForSpansAnonymisation(Metrics):
    def __init__(self, precision_table: pd.DataFrame, recall_table: pd.DataFrame):
        self.precision_table = precision_table
        self.recall_table = recall_table
        self.lenient_levels = {
            0: ["exact"],
            1: ["exact", "subset"],
            2: ["exact", "subset", "tiling"],
            3: ["exact", "subset", "tiling", "overlap"],
        }
        self.metrics = dict(
            P=0.0,
            R=0.0,
            F1=0.0,
            TP_Precision=0,
            TP_Recall=0,
            FN=0,
            FP=0,
            Support=0,
            row_name="",
        )

    def __call__(self, **kwargs):
        self.compute_metrics(**kwargs)
        return pd.DataFrame(self.metrics, index=[self.metrics["row_name"]])

    def compute_metrics(self, lenient_level: int = 0, row_name: str = None):
        """
        Compute evaluation metrics:
        Precision, Recall, F1, TP_Precision, TP_Recall, FN, FP, Support and Row Name
        :param lenient_level: Decide whether to include lenient spans or not.
                Default: 0: only exact matches.
                Options:
                - 1: ["exact", "subset"],
                - 2: ["exact", "subset", "tiling"],
                - 3: ["exact", "subset", "tiling", "overlap"]
        :param row_name: Row name as index
        """
        if 0 <= lenient_level <= 3:
            # True positive cases for Precision: exact matches and accepted lenient spans
            tp_precision = self.precision_table[
                self.precision_table["status"].isin(self.lenient_levels[lenient_level])
            ].shape[0]
            # True positive cases for Recall: exact matches and accepted lenient spans
            tp_recall = self.recall_table[
                self.recall_table["status"].isin(self.lenient_levels[lenient_level])
            ].shape[0]

            # Update metrics
            self.metrics["P"] = self.precision(
                tp_precision, self.precision_table.shape[0]
            )
            self.metrics["R"] = self.recall(tp_recall, self.recall_table.shape[0])
            self.metrics["F1"] = self.f1(self.metrics["P"], self.metrics["R"])
            self.metrics["TP_Precision"] = self.precision_table[
                self.precision_table["status"].isin(self.lenient_levels[lenient_level])
            ].shape[0]
            self.metrics["TP_Recall"] = self.recall_table[
                self.recall_table["status"].isin(self.lenient_levels[lenient_level])
            ].shape[0]
            self.metrics["FN"] = self.recall_table.shape[0] - tp_recall
            self.metrics["FP"] = self.precision_table.shape[0] - tp_precision
            self.metrics["Support"] = self.recall_table.shape[0]
            self.metrics["row_name"] = row_name
        else:
            raise ValueError(
                f"{lenient_level} is not allowed! Only levels between 0 and 3"
            )


class MetricsForCategoricalSpansAnonymisation(Metrics):
    def __init__(
        self,
        precision_table: pd.DataFrame,
        recall_table: pd.DataFrame,
        classification_head: str = "head_0",
        suffix: str = "_Y",
    ):
        self.classification_head = classification_head
        self.suffix = suffix
        self.precision_table = precision_table[
            ["status", classification_head, classification_head + suffix]
        ]
        self.precision_cls_head = precision_table[classification_head]

        self.recall_table = recall_table[
            ["status", classification_head, classification_head + suffix]
        ]
        self.recall_cls_head = recall_table[classification_head]

        # Merge categories from precision and recall tables for given classification head
        self.categories = sorted(
            [
                cat
                for cat in pd.concat(
                    [self.precision_cls_head, self.recall_cls_head]
                ).unique()
                if cat != "O"
            ]
        )
        self.lenient_levels = {
            0: ["exact"],
            1: ["exact", "subset"],
            2: ["exact", "subset", "tiling"],
            3: ["exact", "subset", "tiling", "overlap"],
        }
        self.metrics = dict(
            P=0.0,
            R=0.0,
            F1=0.0,
            TP_Precision=0,
            TP_Recall=0,
            FN=0,
            FP=0,
            Support=0,
            row_name="",
        )

    def __call__(self, lenient_level, input_category):
        categorical_metrics = []
        for cat in self.categories:
            self.compute_metrics(lenient_level, input_category=cat)
            categorical_metrics.append(
                pd.DataFrame(self.metrics, index=[cat.capitalize()]).drop(
                    columns="row_name"
                )
            )
        return pd.concat(categorical_metrics)

    def compute_metrics(self, lenient_level: int = 0, input_category: str = None):
        """
        Method to compute classification metrics for given category.
        Compute evaluation metrics:
        Precision, Recall, F1, TP_Precision, TP_Recall, FN, FP, Support and Row Name
        :param input_category: Input category
        :param lenient_level: Decide whether to include lenient spans or not.
                Default: 0: only exact matches.
                Options:
                - 1: ["exact", "subset"],
                - 2: ["exact", "subset", "tiling"],
                - 3: ["exact", "subset", "tiling", "overlap"]
        """
        # Conditions for TP:
        # 1. Exact match and additional lenient levels
        # 2. x_head equals input category
        # 3. x_head == x_head_Y, where x_head is the reference column and x_head_Y is the candidate column
        tp_precision = self.precision_table[
            (self.precision_table["status"].isin(self.lenient_levels[lenient_level]))
            & (self.precision_table[self.classification_head] == input_category)
            & (
                self.precision_table[self.classification_head]
                == self.precision_table[self.classification_head + self.suffix]
            )
        ].shape[0]
        tp_recall = self.recall_table[
            (self.recall_table["status"].isin(self.lenient_levels[lenient_level]))
            & (self.recall_table[self.classification_head] == input_category)
            & (
                self.recall_table[self.classification_head]
                == self.recall_table[self.classification_head + self.suffix]
            )
        ].shape[0]

        # Total amount of input category as denominator
        n_category_precision = sum(self.precision_cls_head == input_category)
        n_category_recall = sum(self.recall_cls_head == input_category)

        # Update metrics
        if n_category_precision != 0:
            self.metrics["P"] = self.precision(tp_precision, n_category_precision)
        else:
            self.metrics["P"] = 0.0
        if n_category_recall != 0:
            self.metrics["R"] = self.recall(tp_recall, n_category_recall)
        else:
            self.metrics["R"] = 0.0
        # Return F1 = 0.0 if ZeroDivisionError occurs.
        try:
            self.metrics["F1"] = self.f1(self.metrics["P"], self.metrics["R"])
        except ZeroDivisionError:
            self.metrics["F1"] = 0.0

        self.metrics["TP_Precision"] = tp_precision
        self.metrics["TP_Recall"] = tp_recall
        self.metrics["FN"] = n_category_recall - tp_recall
        self.metrics["FP"] = n_category_precision - tp_precision
        self.metrics["Support"] = n_category_recall
        self.metrics["row_name"] = input_category
