from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, RFE, SequentialFeatureSelector
import pandas as pd
from pathlib import Path

def ANOVA_one_way(data: pd.DataFrame, target: pd.DataFrame, output: str | Path) -> pd.DataFrame:
    best_feats = SelectKBest(score_func=f_regression, k='all') # just get all features

#    for col in data.columns:
#        F, p = f_oneway(data.loc[:, col])
#        print("Scores for " + str(col) + ":")
#        for i, cat in enumerate(data.index):
#            print(cat + ": " + F[i])

    print("\nDOUBLE CHECKING TARGET.VALUES.RAVEL()\n" + str(target.values.ravel()) + "\n")
    fit = best_feats.fit(data, target.values.ravel())
#    print(fit.get_feature_names_out())
#    print(fit.scores_)
#    print(fit.pvalues_)

    temp = [fit.scores_, fit.pvalues_]
    score_df = pd.DataFrame(temp, index=["f score", "p value"], columns=fit.get_feature_names_out())

    try:
        with open(output, "w") as f:
            score_df.to_json(f, indent=2)
    except Exception as exp:
        print(f"issue with saving json file: {exp}")

    return score_df


def select_features(score_df: pd.DataFrame, threshold: float=0.05) -> list:
    '''List of column names from score_df whose p-values are <= threshold'''
    features = []
    for col_name, values in score_df.items():
        if values[1] <= threshold:
            features.append(col_name)

    return features


