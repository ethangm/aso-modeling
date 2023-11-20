from os import mkdir, getcwd, listdir
from os.path import isdir, join
from math import sqrt, pow
from matplotlib import pyplot as plt
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

def scrape_selected(directory: str | Path) -> np.ndarray:
    # returns an array of shape (all selected features, 2), where second entry is frequency of selection

    selected = []
    for file in listdir(directory):
        if file.split("/")[-1][0] != "_": # ONLY IF LOF DOESNT DETECT OUTLIERS
            split = file.split(".")
            if split[-1] == "json":
                underscore = split[0].split("_")
                if(underscore[-1] != "scores"):  # only json files in directory should be feat_select_mae_scores and all selected feat
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
    np.save(directory + "all_features_skip.npy", result)    # _SKIP FOR SKIPPING NO OUTLIER RUN
    return result


def bar_chart(input: np.ndarray, directory: str | Path, save_name: str = "feature_selection", cutoff: int = 2) -> None:
    subset = [i for i in input if i[1] > cutoff]
    labels = [i[0].astype(int).astype(str) for i in subset]
    count = [i[1] for i in subset]

    plt.bar(labels, count)
    plt.xlabel("ASO gridpoints")
    plt.ylabel("# times selected")
    plt.title(f"All features selected more than {cutoff} times ({len(input)} selected total)")

    plt.savefig(directory + f"{save_name}.png")
    plt.close()


def feature_region_selectivity(features: np.ndarray, grid: np.ndarray, directory: str | Path, power_factor: int = 2) -> np.ndarray:
    # Vp = (Vi/Di^P)/(1/Di^P) summed for all neighbors 1 -> n
    # the higher the power_factor (P), the less that further points matter
    regional_feats = np.array(features, copy=True, dtype=np.float32)
    for i, feat_select in enumerate(regional_feats):
        grid_num = feat_select[0].astype(np.int32)
        weight = feat_select[1]
        numer = 0
        denom = 0
        for other_grid, freq in features:
            if grid_num != other_grid:
                numer += IDW(grid[grid_num], grid[other_grid], freq, power_factor)
                denom += IDW(grid[grid_num], grid[other_grid], 1, power_factor)
        temp = numer / denom
        regional_feats[i][1] = weight + temp

    np.save(directory + f"all_features_IDW_{power_factor}_skip.npy", regional_feats) # # _SKIP FOR SKIPPING NO OUTLIER RUN
        
    return regional_feats


def IDW(coord1: np.ndarray, coord2: np.ndarray, freq: int, power_factor: int) -> np.float32:
    sq_dist = pow(coord1[0] - coord2[0], 2) + pow(coord1[1] - coord2[1], 2) + pow(coord1[2] - coord2[2], 2)
    distance = sqrt(sq_dist)
    result = (freq / pow(distance, power_factor))
    return np.float32(result)

    
def clustering(features: np.ndarray, grid: np.ndarray, directory: str | Path):
    data = {}
    data["key"] = []
    data["x"] = []
    data["y"] = []
    data["z"] = []
    for feat in features:
        i = 1
        while i <= feat[1].astype(np.int64):
            data["key"].append(feat[0].astype(str))
            data["x"].append(grid[feat[0].astype(int)][0])
            data["y"].append(grid[feat[0].astype(int)][1])
            data["z"].append(grid[feat[0].astype(int)][2])
            i += 1
    df = pd.DataFrame(data)
    df.set_index("key", inplace=True)
    print(df)
    def elbow(distortions: list, clusters: int=11) -> int:
        kl = KneeLocator(list(range(1, clusters)), distortions)
        return kl.knee
    
    def distort(space: pd.DataFrame, sample: pd.DataFrame) -> tuple:
        assign, distances = pairwise_distances_argmin_min(space, sample)
        distortion = np.sum([i**2 for i in distances])
        return assign, distortion
    
    distortions = []
    assignments = []
    upper_k = 11 #len(df.index) + 1
    for i in range(1, upper_k):
        kmeans = KMeans(i, random_state=42, n_init=10)
        kmeans.fit(df)
        temp1, temp2 = distort(df, pd.DataFrame(kmeans.cluster_centers_))
        assignments.append(temp1)
        distortions.append(temp2)

    knee = elbow(distortions, upper_k)
    plot_elbow(distortions, knee + 1, directory) # knee+1 b.c. knee represents index in assignments list
    
    print(len(assignments[knee]))
    result = pd.DataFrame(assignments[knee], columns=["assignment"], index=df.index)


    return result


def plot_elbow(distortions: list, elbow: int, directory: str | Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid(True, axis="x", linestyle="--")

    plt.title(f"kmeans elbow plot (distortion)")
    plt.xlabel("k clusters")
    plt.xticks(range(1, len(distortions)))
    plt.ylabel("distortion")

    ax.plot(range(1, len(distortions) + 1), distortions)
    plt.axvline(elbow, 0, 1, color="r")
    plt.text(elbow + 0.1, 0.5, f"knee is {elbow}")
    fig.savefig(directory + f"kmeans_elbow.png", bbox_inches="tight")

    

if __name__ == "__main__":  # idw/kmeans, directory of pipes.py output, filepath to grid.np
    features = scrape_selected(sys.argv[2])
    print(features)
    print(len(features))
    grid = np.load(sys.argv[3])
    match sys.argv[1]:
        case "idw":
            bar_chart(features, sys.argv[2], cutoff=5)

            region_1 = feature_region_selectivity(features, grid, sys.argv[2], 1)
            bar_chart(region_1, sys.argv[2], "feature_selection_IDW_1_skip", 7)  # _SKIP FOR SKIPPING NO OUTLIER RUN
            print(region_1)

            region_2 = feature_region_selectivity(features, grid, sys.argv[2])
            bar_chart(region_2, sys.argv[2], "feature_selection_IDW_2_skip", 7)  # _SKIP FOR SKIPPING NO OUTLIER RUN
            print(region_2)

            region_3 = feature_region_selectivity(features, grid, sys.argv[2], 3)
            bar_chart(region_3, sys.argv[2], "feature_selection_IDW_3_skip", 7)  # _SKIP FOR SKIPPING NO OUTLIER RUN
            print(region_3)

            differences = []
            for i, value in enumerate(region_1):
                    differences.append((value[0].astype(int), value[1] - region_2[i][1], value[1] - region_3[i][1]))

            print("gridpoint | p=1 - p=2 | p=1 - p=3")
            for j in differences:
                    print(j)
        case "kmeans":
            cluster_dir = f"{getcwd()}/{sys.argv[2]}/_cluster"
            if not isdir(cluster_dir):
                mkdir(cluster_dir)
            assignments = clustering(features, grid, sys.argv[2])
            print(assignments)
            with open(f"{cluster_dir}/cluster_assignments.json", "w") as f:
                assignments.to_json(f, orient="split")
        case _:
            print("Please input idw or kmeans")