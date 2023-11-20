import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from os import makedirs, listdir
from os.path import isdir, join


def unpack_data(aso_path: str | Path, ee_path: str | Path) -> tuple: # returns target, then feature dataframes
    edf, catalysts = _unpack_ee(ee_path)
    edf = rename_target(edf) # specific to this csv formatting
    adf = _unpack_aso(aso_path, catalysts)

    edf = edf.sort_index(axis=0)
    adf = adf.sort_index(axis=0)

    return adf, edf

        
def _unpack_aso(aso_path: str | Path, catalysts: list) -> pd.DataFrame:
    try:
        with open(aso_path, 'r') as f:
            for i, line in enumerate(f):    # obtuse way versus pd.read_csv to avoid loading massive df into memory. check if necessary
                line = line.strip()
                if i == 0:
                    df = pd.DataFrame(columns=line.split(',')[1:])

                temp = line.split(',')
                if temp[0] in catalysts:
                    df.loc[temp[0]] = temp[1:]

    except Exception as exp:
        print(f"Error with unpacking aso csv data: {exp}")
    else:
        return df


def _unpack_ee(ee_path: str | Path) -> tuple:
    try:
        with open(ee_path, 'r') as f:
            df = pd.read_csv(f)
        df.set_index('reaction handle', inplace=True)
        df.drop(columns=["dr"], inplace=True) # only using ee as target

        catalysts = get_relevant_catalysts(df)
    except Exception as exp:
        print(f"Error with unpacking ee csv: {exp}")
    else:
        return df, catalysts
    

def get_relevant_catalysts(df: pd.DataFrame) -> list:
    catalysts = []
    for i in df.index.to_list():
        temp = i.replace('_1B', "")
        catalysts.append(temp)

    return catalysts


def pickler(input, output: str | Path, filename: str | Path):
    try:
        if not isdir(output):
            makedirs(output)
        with open(output + filename, 'wb') as f:
            pickle.dump(input, f)
    except Exception as exp:
        print("Error with pickling " + str(input) + f": {exp}")


def to_json(input, filepath: str | Path):
    try:
        with open(filepath, 'w') as f:
            json.dump(input, f, indent=2)
    except Exception as exp:
        print("Error with outputting " + str(input) + f" to json file: {exp}")


def rename_target(target: pd.DataFrame) -> pd.DataFrame:
    new_names = {}
    for name in target.index:                                              # fix naming of target dataframe
        temp = name.split('_')
        if temp[0] == 'aa':                                             # outlier in naming scheme
            new_names[name] = temp[0] + "_" + temp[1]
        else:
            new_names[name] = temp[0] + "_" + temp[1] + "_" + temp[2] + "_" + temp[3]                # remove ending 
    
    df = target.rename(index=new_names)

    return df


def scrape_selected_model(directory: str | Path, model_name: str) -> np.ndarray:
    # returns an array of shape (all selected features, 2), where second entry is frequency of selection

    selected = []
    for file in listdir(directory):
        split = file.split(".")
        if split[-1] == "json":
            underscore = split[0].split("_")
            if(underscore[-1] != "scores" and underscore[1] == model_name):  # only json files in directory should be feat_select_mae_scores and all selected feat
                path = join(directory, file)
                with open(path, "r") as f:
                    temp = json.load(f)
                if len(selected) == 0:
                    for mol in temp:
                        selected.append([mol, 1])
                else:
                    for mol in temp:
                        for set in selected:
                            add = True
                            if mol == set[0]:
                                set[1] += 1
                                add = False
                                break
                        if add:
                            selected.append([mol, 1])
 
    selected = sorted(selected, key=lambda x: int(x[0]))


    result = np.array(selected, np.int32)
    #np.save(directory + "all_features.npy", result)
    return result


def filter_feats(feats: np.ndarray, threshold: int=2) -> np.ndarray:
    return np.array([i for i in feats if i[1] >= threshold])