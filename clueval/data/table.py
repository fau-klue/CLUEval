import pandas as pd
import numpy as np
from typing import List


class Relevel:
    def __init__(self, dataframe: pd.DataFrame, path_to_tag_set: str = None, tag_set: str = "ag"):
        """
        Reorder information category.
        :param dataframe: Input dataframe containing spans
        :param path_to_tag_set: Path to tags set from resource
        :param tag_set: Options: 'ag' and 'olg'. Default: 'ag'
        """
        self.dataframe = dataframe
        if path_to_tag_set:
            self.tags_df = pd.read_csv(path_to_tag_set, delimiter="\t")
            if tag_set == "ag":
                self.tags_list = [self.tags_df["label_OLG"].iloc[i] for i, tag in enumerate(self.tags_df["label_AG"])
                                  if not pd.isnull(tag) and not pd.isnull(self.tags_df["label_OLG"].iloc[i])
                                  ]
            if tag_set == "olg":
                self.tags_list = [tag for tag in self.tags_df["label_OLG"] if not pd.isnull(tag)]
        else:
            self.tags_list = []

    def __call__(self, cat: bool = False, prefix: str = "id"):
        if self.tags_list:
            dataframe = self._relevel_category(self.dataframe, cat=cat)
            dataframe = self._assign_span_ids(dataframe, prefix=prefix)
        else:
            dataframe = self._assign_span_ids(self.dataframe, prefix=prefix)
        return dataframe

    def _relevel_category(self, inp_data: pd.DataFrame, cat: bool = False):
        """
        Add IDs to information classes to get proper ordering.
        :param inp_data: Input dataframe containing annotated spans
        :param cat: Boolean indication for information class
        """
        if cat:
            cat_replacement = {cat: f"0{i+1}_{cat}" if len(str(i+1)) == 1 else f"{i+1}_{cat}"
                               for i, cat in enumerate(self.tags_list)
                               }
            inp_data["cat"] = inp_data.cat.replace(cat_replacement)
            # inp_data["cat"].replace(cat_replacement, inplace=True)
        return inp_data

    @staticmethod
    def _assign_span_ids(inp_data: pd.DataFrame, prefix: str = "id"):
        """
        Assign IDs to annotated spans using the given prefix.
        :param inp_data: Pandas dataframe with extracted spans
        :param prefix: IDs prefix
        """
        inp_data["id"] = [f"{prefix}{i + 1:06d}" for i in range(inp_data.shape[0])]
        return inp_data


class Join:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Join tables and match spans from two dataframes to get exact no, partial and exact matches.
        :param x: Left spans dataframe
        :param y: Right spans dataframe
        """
        self.x = x
        self.y = y

    def __call__(self, on: str | List[str], how: str = "inner", suffixes: tuple = ("", "_Y")):
        """
        :param on: Column or columns for joining
        :param how: Join strategy. Options: 'inner', 'left'. Default: 'inner'
        :param suffixes: Columns suffixes after tables join
        """
        exact_match_df = self.get_exact_match(on=on, how=how, suffixes=suffixes)
        rest_match_df = self.match_rest(exact_match_df, suffixes=suffixes)

        # Concatenate both dataframes to a single one
        all_df = pd.concat([exact_match_df, rest_match_df])
        # Drop redundant columns
        all_df[[f"start{suffixes[1]}", f"end{suffixes[1]}"]] = all_df[["start", "end"]].where(
            all_df["status"] == "TP", all_df[[f"start{suffixes[1]}", f"end{suffixes[1]}"]].values
        )
        all_df[["domain", "set"]] = all_df[[f"domain{suffixes[1]}", f"set{suffixes[1]}"]].where(
            all_df["status"] == "FP", all_df[["domain", "set"]].values
        )
        all_df["verdict"] = all_df[f"verdict{suffixes[1]}"].where(all_df["verdict"].isna(), all_df["verdict"].values)
        all_df.drop(columns=[f"domain{suffixes[1]}",
                             f"set{suffixes[1]}",
                             f"verdict{suffixes[1]}",
                             "id_L",
                             "id_R"], inplace=True)

        # Reorder table
        all_df = all_df.sort_values(by=["start", f"start{suffixes[1]}"]).reset_index(drop=True)
        return all_df

    def get_exact_match(self, on: str | List[str], how: str = "inner", suffixes: tuple = ("", "_Y")):
        """
        Inner join x and y on given columns to get exact matches between x and y.
        :param x: Input dataframe left
        :param y: Input dataframe right
        :param on: List of columns for joining
        :param how: Join strategy. Default: 'inner'. Option: 'left'
        :param suffixes: Tuple of columns suffixes after merging tables
        """
        return self.x.merge(self.y, on=on, how=how, suffixes=suffixes).dropna(subset=[f"text{suffixes[1]}"]).assign(
            status="TP")

    def match_rest(self, match_df: pd.DataFrame, suffixes: tuple = ("", "_Y")):
        """
        Extract remaining rows to 'rest' (no or partial match) dataframe and assign new id column to table.
        :param match_df: Pandas dataframe containing exact matches to filter out non-exact matches
        :param suffixes: Tuple of columns suffixes after joining tables
        """
        # Get rows in x but not in exact matches and insert new id column to dataframe
        rest_x = self.x.loc[~self.x["id"].isin(match_df["id"])].reset_index(drop=True)
        rest_x = rest_x.assign(id_R=["" for _ in range(rest_x.shape[0])])
        # Get rows in y but not in exact matches
        rest_y = self.y.loc[~self.y["id"].isin(match_df[f"id{suffixes[1]}"])].reset_index(drop=True)
        rest_y = rest_y.assign(id_L=["" for _ in range(rest_y.shape[0])])
        # Apply _get_overlap() function to extract partial matches between rest_x and rest_y
        # Join both tables to a unified one afterward
        rest_x, rest_y = self._get_overlaps(rest_x, rest_y)
        rest = rest_x.merge(rest_y, left_on="id_R", right_on="id", how="outer", suffixes=suffixes)
        return self._assign_status_to_rest_match(rest)

    @staticmethod
    def _get_overlaps(left: pd.DataFrame, right: pd.DataFrame):
        """
        Determine whether a span in the left table has at least one overlapping with spans from the right one
        (Comparison from left to right). Get max overlapping for each comparison and assign the respective id to span.
        A 1-1 mapping is not enforced here to avoid spurious FN where the classifier has predicted multiple annotation
        spans as one long single sequence -> Assign multiple gold annotations to one single prediction.
        :param left: Input dataframe on the left side
        :param right: Input dataframe on the right side.
        """
        # Check whether there is at least one overlapping between left and right
        for i in range(left.shape[0]):
            overlap = np.maximum(0, np.minimum(left.end.iloc[i], right.end) - np.maximum(left.start.iloc[i],
                                                                                         right.start) + 1)
            # Get id for overlapping if exists
            if max(overlap) > 0:
                id_max_overlap = overlap.idxmax()
                # Assign span id from right to corresponded span from left
                # left.id_R.iloc[i] = right.id.iloc[id_max_overlap]
                left.loc[i, "id_R"] = right.id.iloc[id_max_overlap]
                # Assign span id from left to corresponded span from right
                # right.id_L.iloc[id_max_overlap] = left.id.iloc[i]
                right.loc[id_max_overlap, "id_L"] = left.id.iloc[i]
        return left, right

    @staticmethod
    def _assign_status_to_rest_match(rest: pd.DataFrame):
        """
        Assign new status to remaining matched spans.
        For FN (false negative) and FP (false positive) we solely look at the id_L and id columns.
        For the remaining matches, we compare their start and end positions to determine whether a span is
        shorter (sub) or longer (super) than the other one.
        :param rest: Spans dataframe
        """
        status = ["overlap"] * rest.shape[0]
        for i in range(rest.shape[0]):
            if pd.isna(rest.id_L.iloc[i]):
                status[i] = "FN"
            elif pd.isna(rest.id.iloc[i]):
                status[i] = "FP"
            else:
                # Subset: start left is higher than or equals right and end left is smaller than or equals right
                if status[i] == "overlap" and rest.start.iloc[i] >= rest.start_Y.iloc[i] and rest.end.iloc[i] <= rest.end_Y.iloc[i]:
                    status[i] = "sub"
                # Super: start left is smaller than or equals right and end left higher than or equals right
                elif status[i] == "overlap" and rest.start.iloc[i] <= rest.start_Y.iloc[i] and rest.end.iloc[i] >= rest.end_Y.iloc[i]:
                    status[i] = "super"
        rest = rest.assign(status=status)
        return rest


class JoinMultitaskSpans(Join):
    def __init__(self, anon: pd.DataFrame, entity: pd.DataFrame, risk: pd.DataFrame):
        """
        Join anon, entities and risk spans tables into one unified dataframe.
        :param anon: Anon spans
        :param entity: Entity spans
        :param risk: risk spans
        """
        super().__init__(anon, entity)
        self.risk = risk

    def __call__(self, on: str | List[str], how: str = "inner", suffixes: tuple = ("", "_Y")):
        """
        :param on: Column or columns for joining
        :param how: Join strategy. Options: 'left', 'inner'. Default: 'inner'
        :param suffixes: Columns suffixes after joining tables
        """
        # Firstly, predictions for anon and information classes are joined
        anon_cat = super().__call__(on=on, how=how, suffixes=suffixes)
        # Get partial matches for anon_cat
        anon_cat[["start", "end", "text"]] = anon_cat[[f"start{suffixes[1]}", f"end{suffixes[1]}", f"text{suffixes[1]}"]].where(
            anon_cat.status == "sub", anon_cat[["start", "end", "text"]].values
        )
        # Fill in some information for spans found by only one of the models
        # Assign "00_ANY" as new category to anon_cat table. This class indicates that an anonymisation span has
        # been predicted but not information class.
        cols = ["start", "end", "text", "id", "anon"]
        source_cols = [f"start{suffixes[1]}", f"end{suffixes[1]}", f"text{suffixes[1]}", f"id{suffixes[1]}", "anon"]

        for col, source_col in zip(cols, source_cols):
            anon_cat[col] = np.where(anon_cat.status == "FP", anon_cat[source_col], anon_cat[col])

        # anon_cat["cat"].loc[anon_cat.status == "FN"] = "00_Any"
        anon_cat.loc[anon_cat.status == "FN", "cat"] = "00_Any"
        # Reorder spans based on position ids and remove duplications if exist -> Remove nested spans and keep longer
        # spans
        anon_cat = anon_cat.sort_values(by=["start", "end"], ascending=[True, False])
        anon_cat = anon_cat.loc[~anon_cat.duplicated(subset="start")].loc[~anon_cat.duplicated(subset="end")]
        while any(anon_cat.end.shift() >= anon_cat.start):
            anon_cat = anon_cat.loc[~(anon_cat.end.shift(fill_value=True) >= anon_cat.start)].reset_index(drop=True)
        assert not any(anon_cat.end.shift() >= anon_cat.start)  # This assertion should check whether the spans are correctly sorted
        anon_cat.drop(columns=["status", f"start{suffixes[1]}", f"end{suffixes[1]}", f"id{suffixes[1]}", f"text{suffixes[1]}"], inplace=True)

        # Secondly, join Anon-Entity table with Risk spans
        super().__init__(anon_cat, self.risk)
        all_df = super().__call__(on=on, suffixes=suffixes)
        all_df = all_df.loc[all_df.status != "FP"]
        # Add 'Any' annotation to existing spans
        # all_df["risk"].loc[all_df.status == "FN"] = "00_Any"
        all_df.loc[all_df.status == "FN", "risk"] = "00_Any"
        all_df.drop(columns=[f"text{suffixes[1]}", f"id{suffixes[1]}", f"start{suffixes[1]}", f"end{suffixes[1]}"], inplace=True)
        return all_df
