import shap
from os import mkdir, listdir
from os.path import isdir
from pathlib import Path
from sklearn.model_selection import BaseCrossValidator, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
import sys
import pickle
import json
import helpers
import csv




def shap_analysis(model, name: str, cv: BaseCrossValidator, steps: list, features: pd.DataFrame, target: pd.DataFrame, selected: list, outliers: list, save_path: str | Path):
    """
    Change implementation later, when saving all parts correctly. Also add in outlier detection part too.
    """
    # For some of these I forgot to save entire pipeline, but I have the preprocessing steps,
    # train/test splitter, and selected features saved and can recreate estimator
    steps.append((name, model))
    pipe = Pipeline(steps)

    df = features.loc[:, selected]
    df = df.drop(outliers)
    target = target.drop(outliers)
    t_df = ee_transform(target)

    print(df)
    print(t_df)
    print(cv)
    all_values = []
    ix_test = []
    for train, test in cv.split(df, t_df):    # gives out train/test indices
        ix_test.append(test)

        train_X = df.iloc[train, :]
        train_y = t_df.iloc[train, :]
        test_X = df.iloc[test, :]
        test_y = t_df.iloc[test, :]

        pipe.fit(train_X, train_y)
        explainer = shap.Explainer(pipe.predict, train_X)
        shap_values = explainer.shap_values(test_X)
        print(shap_values)
        for val in shap_values:
            all_values.append(val)

    mean_values = np.mean(np.array(all_values), 0)
    print(all_values)
    print(mean_values)

    shap_scores = {}
    i = 0
    for col in df.columns:
        temp = []
        for j in all_values:
            temp.append(j[i])
        shap_scores[col] = temp
        i += 1


    scores = save_path.split(".")[0]
    scores += "_scores.json"
    with open(scores, 'w') as f:   #TEMPORARY
        json.dump(shap_scores, f, indent=2)

    #new_index = [i for i_fold in ix_test for i in i_fold]
    #print(new_index)
    shap.summary_plot(np.array(all_values), df) # df.reindex(new_index)
    plt.savefig(save_path)
    plt.close()



# taken from pipes
def ee_transform(target: pd.DataFrame) -> None: # assuming temp is room temp, 293 K
    # delta delta G = -RTln(abs(ee))
    # modifies in place
    for idx, i in target.iterrows():
        temp = log(abs(i[0])) * 293 * 0.0019872 # took the (-) out for the plotter
        target.loc[idx] = temp                     # ^ So that positive SHAP values mean positive boost to enantioselectivity

    target.rename(columns={"ee": "-ddGTS"}, inplace=True)
    return target


def unpickler(directory: str | Path) -> list:
    temp = []
    for file in listdir(directory):
        name = file.split(".")[0]
        if name != "pipes": # skip pipes subdirectory
            with open(directory + file, "rb") as f:
                temp.append((name, pickle.load(f)))
    return temp


def selected_features(filepath: str| Path) -> list:
    with open(filepath, "r") as f:
        return json.load(f)
    

def detected_outliers(filepath: str | Path) -> list:
    results = []
    with open(filepath, "r") as f:
        for row in csv.reader(f):
            for i in row:
                results.append(i.replace("'", ""))
    return results
    

def pickled_unpack(pickled: list, selected_features: str):
    """Only necessary because I didn't save final pipeline correctly, in the future deprecate this"""
    # Utilize the selected features formatting to take correct model/cv/feat_selector from pickled folder
    # Format example: Ridge_3-Fold_SFS_selected_features.json (split on the _, model-cv-featselect)
    temp = selected_features.split("/")[-1]
    split = temp.split("_")[:-2]

    match split[0]:
        case "Ridge":
            model = Ridge(random_state=42)
        case "Lasso":
            model = Lasso(0.005, random_state=42)
        case "Elastic Net":
            model = ElasticNet(0.0001, random_state=42)
        case _:
            raise TypeError("Unrecognized model type")
        
    cv = None
    for i in pickled:
        if i[0] == split[1]:
            cv = i[1]
            break
    if cv is None:
        raise ValueError("CV not found in pickle folder")
    
    steps = [i for i in pickled if i[0] == "MinMax_Scaler"]
    steps.extend([i for i in pickled if i[0] == "Power_Transformer"])

    return model, cv, steps, split[0]


def correct_pickled_unpack(pickled: list, selected_features: str):
    temp = selected_features.split("/")[-1]
    split = temp.split("_")[:-2]    # lazy way to find correct model from name

    model = None
    for name, obj in pickled:
        name = name.replace("_", " ")
        if name == split[1]:
            model = obj
            break
    if model is None:
        raise ValueError("Model not found in pickled data")
    
    cv = None
    for i in pickled:
        if i[0] == split[2]:
            cv = i[1]
            break
    if cv is None:
        raise ValueError("CV not found in pickle folder")
    
    '''
    outlier = None
    if split[0] != "":
        for i in pickled:
            print(i[0])
            if i[0] == split[0]:
                outlier = i[1]
                break
        if outlier is None:
            raise ValueError("Outlier detection not found in pickle folder")
    '''
        
    steps = [i for i in pickled if i[0] == "MinMax_Scaler"]
    steps.extend([i for i in pickled if i[0] == "Power_Transformer"])
    
    name = split[1]

    return model, cv, steps, name
    


if __name__ == "__main__":  # directory of pickled, filepath of selected features, fp of outliers, filepath of aso and ee data, savepath of png
#    print(pd.read_csv("real_output/ensembles_redo/all_results.csv"))
    


    pickled = unpickler(sys.argv[1])
    selected = selected_features(sys.argv[2])
    outliers = detected_outliers(sys.argv[3])
    model, cv, steps, name = correct_pickled_unpack(pickled, sys.argv[2])
    

    features, target = helpers.unpack_data(sys.argv[4], sys.argv[5])
    features = features.astype(float)

    shap_analysis(model, name, cv, steps, features, target, selected, outliers, sys.argv[6])

