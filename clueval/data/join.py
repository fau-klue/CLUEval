from typing import List

import numpy as np
import pandas as pd
from numpy.ma.core import concatenate

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
        # Post-processing of rest_match_df. # Handle incorrect FPs after joining candidate with reference.
        filtered_rest_df = rest_match_df.loc[~(rest_match_df["id_L"].isna())]
        if not filtered_rest_df.empty:
            filtered_rest_df = filtered_rest_df[filtered_rest_df["id_L"].str.contains("gold")]
            filtered_rest_df = filtered_rest_df[~(filtered_rest_df["status"].isin(["sub"]))]
            id_count = filtered_rest_df["id_L"].value_counts()
            filtered_rest_df = filtered_rest_df.loc[filtered_rest_df["id_L"].map(id_count) > 1].sort_values(by=["id_L", "start_Y"])

            # TODO: Make a method from the following code block...
            # TODO:
            columns_to_concat = ["doc_token_id_start_Y", "doc_token_id_end_Y", "text_Y", "id_Y"]
            agg_func = {}
            for col in filtered_rest_df.columns:
                if col == "id_L":
                    continue
                elif col == f"start{suffixes[1]}":
                    agg_func[col] = "min"
                elif col == f"end{suffixes[1]}":
                    agg_func[col] = "max"
                elif col in columns_to_concat:
                    agg_func[col] = lambda x: " ".join(x.dropna().astype(str))
                else:
                    agg_func[col] = "last"
            # Aggregate FPs + overlap and FP + super to a unified span and insert them to rest_match_df (no incorrect FP cases should exist after this).
            agg_rest_df = filtered_rest_df.groupby("id_L", as_index=False).agg(agg_func)
            rest_match_df = rest_match_df[~(rest_match_df["id_L"].isin(agg_rest_df["id_L"]) & ~(rest_match_df["status"] == "sub"))] # Remove previous entries for those FPs cases
            rest_match_df = pd.concat([rest_match_df, agg_rest_df]).sort_values(by=["start", "end"])
        # Concatenate both dataframes to a single one and drop redundant columns
        all_df = pd.concat([exact_match_df, rest_match_df])
        # Extend TPs: A predicted span should be counted as TP if it equals a gold annotation after the merging step
        all_df.loc[(all_df["text"] == all_df["text_Y"]) & (all_df["status"] != "TP"), "status"] = "TP"
        all_df.loc[all_df["status"] == "TP", [f"start{suffixes[1]}", f"end{suffixes[1]}"]] = all_df.loc[all_df["status"] == "TP", ["start", "end"]].values
        all_df.loc[all_df["status"] == "FP", "domain"] = all_df.loc[all_df["status"] == "FP", f"domain{suffixes[1]}"]
        all_df.loc[all_df["doc_id"].isna(), "doc_id"] = all_df.loc[all_df["doc_id"].isna(), f"doc_id{suffixes[1]}"]

        # Drop redundant columns
        all_df.drop(columns=[f"domain{suffixes[1]}",
                             "id_L",
                             "id_R"], inplace=True)
        # Sort concatenated table
        all_df = all_df.iloc[all_df[["start", f"start{suffixes[1]}"]].min(axis=1).argsort()].reset_index(drop=True)
        return all_df

    def get_exact_match(self, on: str | List[str], how: str = "inner", suffixes: tuple = ("", "_Y")):
        """
        Inner join x and y on given columns to get exact matches between x and y.
        :param x: Input dataframe left
        :param y: Input dataframe right
        :param on: List of columns for joining
        :param how: Join strategy. Default: 'inner'. Option: 'left'
        :paraprint(f"Reference: {annotation_layer[i]}")m suffixes: Tuple of columns suffixes after merging tables
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
            if overlap.max() > 0:
                id_max_overlap = overlap.argmax()
                # Assign span id from right to corresponded span from left
                # left.loc[i, "id_R"] = right.id.iloc[id_max_overlap]
                # Assign span id from left to corresponded span from right
                # right.loc[id_max_overlap, "id_L"] = left.id.iloc[i]
                # Alternative solution: assign all predicted spans to gold. We might need a post processing step to join these overlapping cases
                for j, o in enumerate(overlap):
                    if o != 0:
                        # Assign span id from right to corresponded span from left
                        left.loc[i, "id_R"] = right.id.iloc[j]
                        # Assign span id from left to corresponded span from right
                        if right.loc[j, "id_L"] == "":
                            right.loc[j, "id_L"] = left.id.iloc[i]
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
        rest["status"] = "overlap"
        rest.loc[rest["id_L"].isna(), "status"] = "FN"
        rest.loc[rest["id"].isna(), "status"] = "FP"
        # Subset: start left >= start right and end left <= right
        rest.loc[(rest["status"] == "overlap") & (rest["start"] >= rest[f"start_Y"]) & (rest["end"] <= rest["end_Y"]), "status"] = "sub"
        # Super: start left <= start right and end left >= right
        rest.loc[(rest["status"] == "overlap") & (rest["start"] <= rest[f"start_Y"]) & (rest["end"] >= rest["end_Y"]), "status"] = "super"
        return rest


class JoinAnnotationLayers(Join):
    def __int__(self, x: pd.DataFrame, y: pd.DataFrame):
        super().__init__(x, y)

    def __call__(self, lcat: str, rcat: str, on: str| List[str], how="inner", suffixes: tuple=("", "_Y")):
        exact = self.get_exact_match(on=on, how=how, suffixes=suffixes)
        rest = self.match_rest(exact)
        # Handle sub cases
        rest.loc[rest["status"] == "sub", ["start",
                                           "end",
                                           "doc_token_id_start",
                                           "doc_token_id_end",
                                           "text"]
        ] = rest.loc[rest["status"] == "sub", [f"start{suffixes[1]}",
                                               f"end{suffixes[1]}",
                                               f"doc_token_id_start{suffixes[1]}",
                                               f"doc_token_id_end{suffixes[1]}",
                                               f"text{suffixes[1]}"
                                               ]
        ].values
        # Handle overlapping cases for predicted spans: we want to extend those spans by joining predicted tokens
        mask_overlap = rest["status"] == "overlap"
        rest.loc[mask_overlap, "start"] = rest.loc[mask_overlap, ["start", f"start{suffixes[1]}"]].min(axis=1)
        rest.loc[mask_overlap, "end"] = rest.loc[mask_overlap, ["end", f"end{suffixes[1]}"]].max(axis=1)
        # We also need to adjust doc_token_ids for further analysis
        rest.loc[mask_overlap, "doc_token_id_start"] = rest.loc[mask_overlap, ["doc_token_id_start", f"doc_token_id_start{suffixes[1]}"]].min(axis=1)
        rest.loc[mask_overlap, "doc_token_id_end"] = rest.loc[mask_overlap, ["doc_token_id_end", f"doc_token_id_end{suffixes[1]}"]].max(axis=1)
        # Unified span: Do not keep both spans like in the previous version from the R code: paste(text, "+", text.R).
        # Try to concatenate partially matched spans that are neither 'sub' or 'super'. Partially matching cases:
        # Case 1: text from x starts earlier
        rest.loc[mask_overlap & (rest["start"] < rest[f"start{suffixes[1]}"]), "text"] = [self._concatenate_overlapping(text, text_Y)
                                                                                          for text, text_Y in zip(rest.loc[mask_overlap & (rest["start"] < rest[f"start{suffixes[1]}"]), "text"],
                                                                  rest.loc[mask_overlap & (rest["start"] < rest[f"start{suffixes[1]}"]), f"text{suffixes[1]}"]
                                                                  )
                                                                                          ]
        # Case 2: text from y starts earlier
        rest.loc[mask_overlap & (rest["start"] == rest[f"start{suffixes[1]}"]), "text"] = [
            self._concatenate_overlapping(text_Y, text)
            for text, text_Y in zip(rest.loc[mask_overlap & (rest["start"] == rest[f"start{suffixes[1]}"]), f"text{suffixes[1]}"],
                                    rest.loc[mask_overlap & (rest["start"] == rest[f"start{suffixes[1]}"]), f"text"]
                                    )
            ]

        # Drop NaN by start and end ids
        # rest = rest.dropna(subset=["start", "end"])
        # Fill category columns with information if NaN
        # rest[lcat] = rest[lcat].fillna(lcat)

        # Concatenate both dataframes to a single one if rest is not empty (joining gold annotation layers)
        concatenated = pd.concat([exact, rest])
        # Convert dtypes for start and end to floating points
        concatenated[["start", "end"]] = concatenated[["start", "end"]].astype(float)

        # Fill in some information for spans found by only one of the classification header.
        # If status is FP (exists only in the right table), copy values from right column to corresponding left column
        # only if left value is NaN and right value doesn't exist in the left column
        left_columns = ["start", "end", "doc_token_id_start", "doc_token_id_end", "text", "id", lcat]
        right_columns = [f"start{suffixes[1]}", f"end{suffixes[1]}",
                         f"doc_token_id_start{suffixes[1]}", f"doc_token_id_end{suffixes[1]}",
                         f"text{suffixes[1]}", f"id{suffixes[1]}"
                         ]
        for lcol, rcol in zip(left_columns, right_columns):
            concatenated[lcol] = np.where(concatenated.status == "FP", concatenated[rcol], concatenated[lcol])
        concatenated.loc[concatenated.status == "FN", rcat] = "Any"

        # Sort by asc start and desc end
        all_df = (concatenated.sort_values(by=["start", "end"], ascending=[True, False])).reset_index(drop=True)
        # Drop duplicated start and end
        all_df = all_df.loc[~all_df.duplicated(subset="start")]
        all_df = all_df.loc[~all_df.duplicated(subset="end")]
        # Remove nested spans: If end.shift(1) >= start and wrong FPs (tokens inside longer spans that were not considered as 'sub' error).
        # Use while loop to ensure, that all incorrect FPs will be removed (tokens inside an overlap)
        list_of_nested_id = []
        while True:
            inside = all_df["end"].shift(1) >= all_df["start"]
            if not inside.any():
                break
            list_of_nested_id.extend(all_df[inside].index.tolist())
            all_df = all_df[~inside]# .reset_index(drop=True)
        # all_df = all_df.loc[~(all_df["end"].shift(1) >= all_df["start"])] # R code: All[!(shift(end) >= start)])

        all_df = all_df.sort_values(by=["start", "end"], ascending=[True, False]).reset_index(drop=True)

        # Replicate assertion from R code: stopifnot(all(shift(All$end) < All$start, na.rm=TRUE))
        assertion_cond = (all_df["end"].shift(1) < all_df["start"])
        assertion_cond[0] = True # NaN compare to everything is always False so assertion all() will be False
        assert all(assertion_cond)
        # print(all_df.loc[all_df["end"].shift(1) > all_df["start"]])

        all_df["doc_id"] = all_df[f"doc_id{suffixes[1]}"].where(all_df["doc_id"].isna(), all_df["doc_id"].values)

        all_df = all_df.drop(columns=["status",
                                      f"start{suffixes[1]}",
                                      f"end{suffixes[1]}",
                                      f"doc_token_id_start{suffixes[1]}",
                                      f"doc_token_id_end{suffixes[1]}",
                                      f"id{suffixes[1]}",
                                      f"text{suffixes[1]}",
                                      f"domain{suffixes[1]}",
                                      f"doc_id{suffixes[1]}",
                                      "id_L",
                                      "id_R",]
                         )
        return all_df

    @staticmethod
    def _concatenate_overlapping(text_x, text_y):
        n_overlap = 0
        split_text_x = text_x.split()
        split_text_y = text_y.split()
        for i in range(1, min(len(split_text_x), len(split_text_y)) + 1):
            if split_text_x[-i:] == split_text_y[:i]:
                n_overlap = i
        return " ".join(split_text_x + split_text_y[n_overlap:])


