import numpy as np
import pandas as pd

class Match:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.x = x
        self.y = y

    def __call__(self):
        pass

    @staticmethod
    def exact(x:pd.DataFrame, y: pd.DataFrame, on:str, suffixes: tuple=("", "_Y")):
        """ x.s0 == y.s1 & x.e0 == y.e1 """
        pass

    @staticmethod
    def superset(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches:
        y.s1 <= x.s0 & y.e1 >= x.e0
        """
        pass

    @staticmethod
    def tiling(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches and combining adjacent spans:
        x.s1 == y.s0 && x.e0 == y.e1
        """
        pass

    @staticmethod
    def overlap(x: pd.DataFrame, y: pd.DataFrame):
        """ Remaining rows after omitting exact matches and combining adjacent spans:
        x.s1 <= y.s0 && x.e0 >= y.e1
        """
        pass

    @staticmethod
    def join_adjacent(overlap_df):
        pass

    @staticmethod
    def _overlap(x: pd.DataFrame, y: pd.DataFrame):
        pass

