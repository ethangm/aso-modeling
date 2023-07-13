from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from pathlib import Path
from helpers import to_json


models = {  # TO DO: initialize these correctly
    "Ridge" : Ridge(random_state=42), # alpha, solver?
    "Lasso" : Lasso(), # alpha?
    "Random Forest" : RandomForestRegressor(random_state=42), # n_estimators, max_depth, etc?
    "Gradient Boosting" : GradientBoostingRegressor(),
    "Support Vector Machine" : SVR(),
    "Partial Least Squares" : PLSRegression()
}

metrics = {
    "MAE" : mean_absolute_error,
    "MSE" : mean_squared_error,
    "R**2" : r2_score
}


def all_models(features: pd.DataFrame, target: pd.DataFrame, cv, output: str | Path) -> dict:

    if cv.get_n_splits(features) == len(features):
        r2_error = True
    else:
        r2_error = False

    all_results = {}
    for name, model in models.items():
        model_results = {}
        for metric_name, metric in metrics.items():
            if not r2_error or metric_name != "R**2":  # as long as test array > 1 or its not R**2 scoring
                scores = cross_val_score(model, features, target.values.ravel(), cv=cv, scoring=make_scorer(metric))
                model_results[metric_name] = np.mean(scores)

        all_results[name] = model_results
        
        if r2_error:    # this takes a little bit
            ytests = []
            ypreds = []
            for _, (train, test) in enumerate(cv.split(features)):
                model.fit(features.iloc[train, :], target.iloc[train, :].values.ravel())
                pred = model.predict(features.iloc[test, :])

                ytests += target.iloc[test, :].values.ravel().tolist()
                ypreds += pred.tolist()
            all_results[name]["R**2"] = r2_score(ytests, ypreds)

    to_json(all_results, output + "regression_scores.json")

    return all_results


'''         Need fix for LOO r^2 if I want to go this way, easier to just use sklearn cross_val_score
def all_models(features: pd.DataFrame, target: pd.DataFrame, split: dict) -> dict:

    X_train = features.loc[split["train"]]
    X_test = features.loc[split["test"]]
    y_train = target.loc[split["train"]]
    y_test = target.loc[split["test"]]

    all_results = {}
    for name, model in models.items():
        model_results = {}

        model.fit(X_train, y_train.values.ravel())
        pred = model.predict(X_test)

        for metric_name, metric in metrics.items():
            model_results[metric_name] = metric(y_test, pred)

        all_results[name] = model_results

    return all_results


def all_splits(features: pd.DataFrame, target: pd.DataFrame, splits: dict, output: str | Path) -> dict:
    mean_results = {}
    temp = {}

    for fold, split in splits.items():
        fold_scores = all_models(features, target, split)
        to_json(fold_scores, output + fold + "_scores.json")
        temp[fold] = fold_scores

    for name in models.keys():
        mean_results[name] = {}
        for metric in metrics.keys():
            avg = 0
            i = 0
            for fold2, _ in temp.items():
                avg += temp[fold2][name][metric]
                i += 1
            mean_results[name][metric] = avg / i

    to_json(mean_results, output + "mean_scores.json")

    return mean_results
'''