import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) # Suppress pandas FutureWarnings for the groupby() function for now

import pandas as pd

class Match:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = x
        self.y = y

    def __call__(self, on:str|list[str], suffixes:list[str]=("", "_Y")):
        exact = self.exact_match(self.x, self.y, on=on, suffixes=suffixes)
        rest = self.rest_match(exact, suffixes=suffixes)
        match_df = pd.concat([exact, rest], ignore_index=True).sort_values(by=["start", "end"])
        match_df.drop(columns=[f"start{suffixes[1]}",
                            f"end{suffixes[1]}",
                            "id",
                            "id_y",
                            f"id{suffixes[1]}",
                            f"domain{suffixes[1]}",
                            f"doc_id{suffixes[1]}",
                            f"status{suffixes[1]}"
                             ], inplace=True)
        # Fill Nan values in label columns with "FN"
        for column in [col for col in match_df.columns if col.startswith("head_")]:
            match_df.fillna({column: "FN"}, inplace=True)
        return match_df

    @staticmethod
    def exact_match(x:pd.DataFrame, y: pd.DataFrame, on:str|list[str], suffixes:list[str]=("", "_Y")):
        """ x.s0 == y.s1 & x.e0 == y.e1 """
        return x.merge(y, on=on, suffixes=suffixes, how="inner").assign(status="exact")

    def rest_match(self, exact_df: pd.DataFrame, suffixes:list[str] = ("", "_Y")):
        """

        :param exact_df:
        :param suffixes:
        :return:
        """
        # Temporarily remove exact matches from x and y
        x_rest = self.x.loc[~self.x["id"].isin(exact_df["id"])].reset_index(drop=True).assign(status="rest")
        x_rest = x_rest.assign(id_y="")
        y_rest = self.y.loc[~self.y["id"].isin(exact_df[f"id{suffixes[1]}"])].reset_index(drop=True).assign(status="rest")
        y_rest = y_rest.assign(id_x="")

        # Case 2: x is a subset of y
        x_rest = self.subset(x_rest, y_rest)

        # Check overlaps between x and y. Assign status according to following conditions:
        # 1. Overlap: If spans in y overlap with x and belong to an adjacent span. We can use this case for determining tiling and remaining overlap cases.
        # 2. Assign span to case 5: FN - if spans in y overlap with x but do not belong to any adjacent span
        x_rest, y_rest = self.overlap(x_rest, y_rest)

        # Case 3: tiling - exact match between spans in x and overlapping adjacent spans from y
        x_rest, y_rest = self.match_tiled_spans(x_rest, y_rest)

        # Case 4: overlap - remaining spans in x are contained in adjacent overlapping spans in y
        x_rest, y_rest = self.match_overlap_containment(x_rest, y_rest)

        # Case 5: All remaining rows in x are considered as "mismatching" between x_rest and y_rest
        x_rest.loc[x_rest["status"] == "rest", "status"] = "mismatch"

        # Merge x_rest with y_rest by x_rest.id_y and y_rest.id to receive spans information from y
        x_rest = self.merge_y_to_x(x_rest, y_rest, suffixes=suffixes)
        # x_rest = x_rest.merge(y_rest, how="left", left_on="id_y", right_on="id", suffixes=suffixes)
        return x_rest

    @staticmethod
    def subset(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches:
        y.s1 <= x.s0 & y.e1 >= x.e0
        """
        for i, row in x.iterrows():
            superset_y = y[(y["start"] <= row["start"]) & (y["end"] >= row["end"])]
            if not superset_y.empty:
                matched_id = superset_y.id.values[0]
                x.at[i, "id_y"] = matched_id
                x.at[i, "status"] = "subset"
        return x

    @staticmethod
    def match_tiled_spans(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches and combining adjacent spans:
        x.s1 == y.s0 && x.e0 == y.e1
        """
        # Use an inner join to select all spans from x that exactly match spans in y.
        # Since x and y are rest tables, only adjacent overlapping spans should be considered.
        subset_join = x.merge(y, on=["start", "end"], how="inner", suffixes=("", "_Y"))
        # Reassign status in x and y based on id column from subset_join
        x.loc[x["id"].isin(subset_join["id"]), "status"] = "tiling"
        y.loc[y["id"].isin(subset_join["id_Y"]), "status"] = "tiling"
        return x, y

    @staticmethod
    def match_overlap_containment(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches and overlapping adjacent spans:
        y.s1 <= x.s0 && y.e1 >= x.e0
        """
        x_overlap = x[x["status"] == "overlap"]
        y_overlap = y[y["status"] == "overlap"]
        for i, row in x_overlap.iterrows():
            superset_y = y_overlap[(y_overlap["start"] <= row["start"]) & (y_overlap["end"] >= row["end"])]
            if superset_y.empty:
                x.at[i, "status"] = "mismatch"
                y.loc[y["id_x"].str.contains(x.iloc[i]["id"]), "status"] = "mismatch"
        return x, y

    def overlap(self, x: pd.DataFrame, y: pd.DataFrame):
        for i, row in x.iterrows():
            overlap = y[~((y["end"] < row["start"]) | (row["end"] < y["start"]))]
            if not overlap.empty and row["status"] == "rest":
                # Assign status and id from x to y
                y.loc[overlap.index, "status"] = "overlap"
                y.loc[overlap.index, "id_x"] = row["id"]
                # Determine the number of overlapping tokens between x_row and y_overlap for later use
                y.loc[overlap.index, "number_overlapping_tokens_with_x"] = len(row["text"].split()) - len(overlap["text"].str.split())
                x.at[i, "id_y"] = overlap.id.values[0]
                x.at[i, "status"] = "overlap"
        # Determine adjacent spans in y and keep them as overlap. All remaining overlaps with x should be labelled as FN
        y.fillna(value={"number_overlapping_tokens_with_x": 0}, inplace=True)
        x, y = self.adjacent_spans(x, y)
        y = y[[column for column in y.columns if column not in ["status", "id_x"]] + ["status", "id_x"]] # Rearrange columns
        return x, y

    def adjacent_spans(self, x, overlap_df):
        """

        :param overlap_df:
        :return:
        """
        adjacency_df = overlap_df.copy()
        adjacency_cond = ((adjacency_df["status"] == "overlap") & (adjacency_df["start"] == adjacency_df["end"].shift(1) + 1))
        adjacency_df["adjacent"] = adjacency_cond
        adjacency_df["adjacency"] = (~adjacency_df["adjacent"]).cumsum()

        # Compute number of span for each adjacency group
        number_of_spans_per_group = adjacency_df.groupby("adjacency")["adjacent"].count()
        # Map count values to adjacency dataframe
        adjacency_df["adjacency_count"] = adjacency_df["adjacency"].map(number_of_spans_per_group)
        # Assign case 5 "mismatch" to rows that are marked as overlap but not part of adjacent spans in adjacency_df and in x
        adjacency_df.loc[(adjacency_df["status"] == "overlap") & (adjacency_df["adjacency_count"] == 1), "status"] = "mismatch"
        x.loc[x["id"].isin(adjacency_df.loc[adjacency_df["status"] == "mismatch", "id_x"]), "status"] = "mismatch"

        # Group adjacency_df to unify overlapping spans
        headers_column = [column for column in adjacency_df.columns if column.startswith("head_")]
        combined_adjacency_df = adjacency_df.groupby("adjacency").apply(self.unify_adjacent_spans, headers_column=headers_column)
        return x, combined_adjacency_df

    @staticmethod
    def unify_adjacent_spans(grouped_df, headers_column):
        # Select id from span with based on the total number of tokens (longest span)
        combined_spans = {"start": grouped_df["start"].iloc[0],
                          "end": grouped_df["end"].iloc[-1],
                          "doc_token_id_start": grouped_df["doc_token_id_start"].iloc[0],
                          "doc_token_id_end": grouped_df["doc_token_id_end"].iloc[-1],
                          "doc_id": grouped_df["doc_id"].iloc[0],
                          "domain": grouped_df["domain"].iloc[0],
                          "text": " | ".join(grouped_df["text"]),
                          "id": " | ".join(grouped_df["id"]),
                          "status": grouped_df["status"].iloc[0],
                          "id_x": " | ".join(grouped_df["id_x"])
                          }
        try:
            longest_overlap = grouped_df.loc[grouped_df["number_overlapping_tokens_with_x"].idxmax()]
        except KeyError:
            longest_overlap = grouped_df

        for column in headers_column:
            combined_spans[column] = longest_overlap[column]
        return pd.Series(combined_spans)

    @staticmethod
    def merge_y_to_x(x:pd.DataFrame, y: pd.DataFrame, suffixes=("", "_Y")):
        y_transformed = y.assign(id=y.id.str.split(" | ")
                                 ).explode("id")[["id",
                                                  "start",
                                                  "end",
                                                  "doc_token_id_start",
                                                  "doc_token_id_end",
                                                  "text",
                                                  "status"] + [col for col in y if col.startswith( "head_")]]
        y_transformed = y_transformed[y_transformed["id"] != "|"]
        return x.merge(y_transformed, left_on="id_y", right_on="id", how="left", suffixes=suffixes)