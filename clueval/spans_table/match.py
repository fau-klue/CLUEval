import warnings
warnings.simplefilter(action="ignore", category=FutureWarning) # Suppress pandas FutureWarnings for the groupby() function for now

import pandas as pd

class Match:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, annotation_layer:str|list[str]):
        self.x = x
        self.y = y
        self.annotation_layer = annotation_layer if isinstance(annotation_layer, list) else [annotation_layer]

    def __call__(self, on:str|list[str]):
        exact = self.exact_match(self.x, self.y, on=on)
        rest = self.rest_match(exact)
        match_df = pd.concat([exact, rest], ignore_index=True).sort_values(by=["start", "end"])
        exact_mask = match_df["status"] == "exact"
        match_df.loc[exact_mask, ["start_Y", "end_Y"]] = match_df.loc[exact_mask][["start", "end"]].values
        match_df.drop(columns=["id",
                            "id_y",
                            "id_Y",
                            "doc_id_Y",
                            "domain_Y"
                            ], inplace=True)
        # Fill Nan values in label columns with "FN"
        for column in self.annotation_layer:
            match_df.fillna({column + "_Y": "FN"}, inplace=True)
        match_df.loc[match_df[["start_Y", "end_Y"]].isna().any(axis=1), ["start_Y", "end_Y"]] = -100
        match_df[["start_Y", "end_Y"]] = match_df[["start_Y", "end_Y"]].astype("Int64")
        return match_df.reset_index(drop=True)

    @staticmethod
    def exact_match(x:pd.DataFrame, y: pd.DataFrame, on:str|list[str]):
        """ x.s0 == y.s1 & x.e0 == y.e1 """
        return x.merge(y, on=on, suffixes=("", "_Y"), how="inner").assign(status="exact")

    def rest_match(self, exact_df: pd.DataFrame):
        """
        :param exact_df:
        :return:
        """
        # Temporarily remove exact matches from x and y
        x_rest = self.x.loc[~self.x["id"].isin(exact_df[f"id"])].reset_index(drop=True).assign(status="rest")
        x_rest = x_rest.assign(id_y="")
        y_rest = self.y.loc[~self.y["id"].isin(exact_df[f"id_Y"])].reset_index(drop=True).assign(status="rest")
        y_rest = y_rest.assign(id_x="")

        # Case 2: x is a subset of y
        x_rest = self.subset(x_rest, y_rest)

        # Check overlaps between x and y. Assign status according to following conditions:
        # 1. Overlap: If spans in y overlap with x and belong to an adjacent span. We can use this case for determining tiling (case 3) and remaining overlaps (case 4).
        # 2. Assign span to case 5: FN - if spans in y overlap with x but do not belong to any adjacent span
        x_rest = self.overlap(x_rest, y_rest)
        # Case 5: All remaining rows in x are considered as "unmatching" between x_rest and y_rest
        x_rest.loc[x_rest["status"] == "rest", "status"] = "unmatch"
        return x_rest

    def subset(self, x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches:
        y.s1 <= x.s0 & y.e1 >= x.e0
        """
        y_columns = ["start_Y", "end_Y", "token_id_start_Y", "token_id_end_Y", "text_Y"] + [col + "_Y" for col in self.annotation_layer]
        x[y_columns] = None
        for i, row in x.iterrows():
            superset_y = y[(y["start"] <= row["start"]) & (y["end"] >= row["end"])]
            if not superset_y.empty:
                matched_id = superset_y.id.values[0]
                x.at[i, "id_y"] = matched_id
                x.at[i, "status"] = "subset"
                x.loc[i, y_columns] = superset_y.loc[:, [column.strip("_Y") for column in y_columns]].iloc[0].values
        return x

    def overlap(self, x: pd.DataFrame, y: pd.DataFrame):
        """ Consider overlap cases, where x could be covered by y spans in two different ways:
        1. Tiling: x has the same start and end positions as adjacent spans in y
        2. Overlap: x is covered by longer adjacent spans in y
        :param x: Rest x dataframe
        :param y: Rest y dataframe
        :return:
        """
        _x = x.copy()
        y_columns = ["start_Y", "end_Y", "token_id_start_Y", "token_id_end_Y", "text_Y"] + [col + "_Y" for col in self.annotation_layer]
        for i, row in _x.iterrows():
            if row["status"] == "rest":
                overlap = y[~((row["end"] < y["start"]) | (y["end"] < row["start"]))]
                overlap = overlap.sort_values(by=["start", "end"])
                if not overlap.empty:
                    adjacent = True
                    for j in range(overlap.shape[0] - 1):
                        if overlap.iloc[j]["end"] + 1 != overlap.iloc[j+1]["start"]:
                            adjacent = False
                    if adjacent:
                        # Flag adjacency and generate adjacency dataframe for overlap
                        overlap["adjacent"] = 1
                        overlap["number_overlapping_tokens_with_x"] = overlap.apply(lambda r: len([token for token in r["text"].split() if token in row["text"].split()]), axis=1)
                        overlap["id_x"] = row["id"]
                        adjacent_overlap = overlap.groupby("adjacent").apply(self.unify_adjacent_spans, headers_column=self.annotation_layer)
                        _x.at[i, "id_y"] = adjacent_overlap["id"]
                        # Insert spans information from overlap to _x
                        _x.loc[i, y_columns] = adjacent_overlap.loc[:, [column.strip("_Y") for column in y_columns]].iloc[0].values
                        # Check whether x.s1 == y.s0 && x.e0 == y.e1 (tiling) or y.s1 <= x.s0 && y.e1 >= x.e0 (overlap)
                        if adjacent_overlap["start"].item() == row["start"] and adjacent_overlap["end"].item() == row["end"]:
                            _x.at[i, "status"] = "tiling"
                        elif adjacent_overlap["start"].item() <= row["start"] and adjacent_overlap["end"].item() >= row["end"]:
                            _x.at[i, "status"] = "overlap"
                    else:
                        # Assign 'unmatch' to status in _x
                        _x.at[i, "status"] = "unmatch"
        return _x

    @staticmethod
    def unify_adjacent_spans(adjacent_df, headers_column):
        """ Combine spans information fr
        :param adjacent_df: Grouped dataframe with adjacent span information
        :param headers_column: Prediction head columns
        :return:
        """
        # Select id from span with based on the total number of tokens (longest span)
        combined_spans = {"start": adjacent_df["start"].iloc[0],
                          "end": adjacent_df["end"].iloc[-1],
                          "token_id_start": adjacent_df["token_id_start"].iloc[0],
                          "token_id_end": adjacent_df["token_id_end"].iloc[-1],
                          "doc_id": adjacent_df["doc_id"].iloc[0],
                          "domain": adjacent_df["domain"].iloc[0],
                          "text": " | ".join(adjacent_df["text"]),
                          "id": " | ".join(adjacent_df["id"]),
                          "status": adjacent_df["status"].iloc[0],
                          "id_x": " | ".join(adjacent_df["id_x"])
                          }
        try:
            # Select label according to number of overlapping tokens
            longest_overlap = adjacent_df.loc[adjacent_df["number_overlapping_tokens_with_x"].idxmax()]
        except KeyError:
            longest_overlap = adjacent_df

        for column in headers_column:
            combined_spans[column] = longest_overlap[column]
        return pd.Series(combined_spans)