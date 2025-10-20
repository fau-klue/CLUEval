from typing import List

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

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
        all_df[["domain"]] = all_df[[f"domain{suffixes[1]}"]].where(
            all_df["status"] == "FP", all_df[["domain"]].values
        )
        all_df["doc_id"] = all_df[f"doc_id{suffixes[1]}"].where(all_df["doc_id"].isna(), all_df["doc_id"].values)
        all_df.drop(columns=[f"domain{suffixes[1]}",
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

        # Handle ValueError when reference and prediction yield the same spans
        if not rest_x.empty and not rest_y.empty:
            rest_x, rest_y = self._get_overlaps(rest_x, rest_y)
            rest = rest_x.merge(rest_y, left_on="id_R", right_on="id", how="outer", suffixes=suffixes)
        else:
            # Merge empty dataframes -> no rest matching
            rest = rest_x.merge(rest_y, on="id", how="outer", suffixes=suffixes)
            rest[f"id{suffixes[1]}"] = None
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
            overlap = np.maximum(0, np.minimum(left.end.iloc[i], right.end) - np.maximum(left.start.iloc[i], right.start) + 1)
            # Get id for overlapping if exists
            if max(overlap) > 0:
                id_max_overlap = overlap.idxmax()
                # Assign span id from right to corresponded span from left
                left.loc[i, "id_R"] = right.id.iloc[id_max_overlap]
                # Assign span id from left to corresponded span from right
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
                # Subset: start left >= start right and end left <= right
                if status[i] == "overlap" and rest.start.iloc[i] >= rest.start_Y.iloc[i] and rest.end.iloc[i] <= rest.end_Y.iloc[i]:
                    status[i] = "sub"
                # Super: start left <= start right and end left >= right
                elif status[i] == "overlap" and rest.start.iloc[i] <= rest.start_Y.iloc[i] and rest.end.iloc[i] >= rest.end_Y.iloc[i]:
                    status[i] = "super"
        rest = rest.assign(status=status)
        return rest


class JoinAnnotationLayers(Join):
    def __int__(self, x: pd.DataFrame, y: pd.DataFrame):
        super().__init__(x, y)

    def __call__(self, lcat: str, rcat: str, on: str| List[str], how="inner", suffixes: tuple=("", "_Y")):
        exact = self.get_exact_match(on=on, how=how, suffixes=suffixes)
        rest = self.match_rest(exact)
        # Get partial matches for anon_cat
        rest[["start", "end", "text"]] = rest[
            [f"start{suffixes[1]}", f"end{suffixes[1]}", f"text{suffixes[1]}"]].where(
            rest.status == "sub", rest[["start", "end", "text"]].values
        )
        # Drop NaN by start and end ids
        # rest = rest.dropna(subset=["start", "end"])
        # Fill category columns with information if NaN
        # rest[lcat] = rest[lcat].fillna(lcat)

        # Concatenate both dataframes to a single one
        concatenated = pd.concat([exact, rest])

        # Fill in some information for spans found by only one of the classification header.
        # If status is FP (exists only in the right table), copy values from right column to corresponding left column
        # only if left value is NaN and right value doesn't exist in the left column
        left_columns = ["start", "end", "text", "id", lcat]
        right_columns = [f"start{suffixes[1]}", f"end{suffixes[1]}", f"text{suffixes[1]}", f"id{suffixes[1]}"]
        for lcol, rcol in zip(left_columns, right_columns):
            concatenated[lcol] = np.where(concatenated.status == "FP", concatenated[rcol], concatenated[lcol])
        concatenated.loc[concatenated.status == "FN", rcat] = "Any"

        all_df = (concatenated.sort_values(by=["start", "end"], ascending=[True, False])).reset_index(drop=True)
        all_df = all_df.loc[~all_df.duplicated(subset="start")].loc[~all_df.duplicated(subset="end")]

        all_df.reset_index(drop=True, inplace=True)
        all_df.loc[all_df.status == "FN", rcat] = "Any"
        all_df = all_df.drop(columns=["status",
                                  f"start{suffixes[1]}",
                                  f"end{suffixes[1]}",
                                  f"id{suffixes[1]}",
                                  f"text{suffixes[1]}",
                                  f"domain{suffixes[1]}",
                                  f"doc_id{suffixes[1]}",
                                  "id_L",
                                  "id_R",
                                  "status"]
                         )
        return all_df
