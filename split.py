import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut
from pathlib import Path
from helpers import to_json, pickler

def get_KFold(feat_df: pd.DataFrame, target_df: pd.DataFrame, output: str | Path, k: int=3) -> tuple:
    '''splits = dict of dict of lists, s.t. folds contains "test" and "train" keys that contain list catalyst names'''
    kf = KFold(k, shuffle=True, random_state=42)
    
    splits = {}
    for i, (train, test) in enumerate(kf.split(feat_df, target_df)):
        train_names = feat_df.index[train].to_list()
        test_names = feat_df.index[test].to_list()
        splits["Fold_" + str(i)] = {"train": train_names, "test": test_names}

    to_json(splits, output + "splits.json")
    pickler(kf, output + "kfold.pickle")

    return kf, splits


def LOO(feat_df: pd.DataFrame, target_df: pd.DataFrame, output: str | Path) -> tuple:
    '''Same formatting as get_KFold for train/test split output'''
    loo = LeaveOneOut()
    left_out = {}
    for i, (train, test) in enumerate(loo.split(feat_df, target_df)):
        train_names = feat_df.index[train].to_list()
        test_names = feat_df.index[test].to_list()
        left_out["Fold_" + str(i)] = {"train": train_names, "test": test_names} 
        # dumb way to format LOO, but consistent w/KFolds

    to_json(left_out, output + "left_out.json")
    pickler(loo, output + "LOO.pickle")

    return loo, left_out