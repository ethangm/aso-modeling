import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
import json
from scipy.stats import t as t_test


output_directory = "../images/chp_2/"
input_files = ["/home/ethangm2/NBO-Modeling/aso-modeling-out/_target_fix_reruns/full_regularized/all_results.csv",
            "/home/ethangm2/NBO-Modeling/aso-modeling-out/_target_fix_reruns/half_regularized/all_results.csv",
            "/home/ethangm2/NBO-Modeling/aso-modeling-out/_target_fix_reruns/full_svr/all_results.csv",
            "/home/ethangm2/NBO-Modeling/aso-modeling-out/_target_fix_reruns/half_svr/all_results.csv"
            ]
names = ["full_reg", "half_reg", "full_svr", "half_svr"]

def ddG_plotter(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame):
    '''
    Compares actual vs. predicted ddG by training pipe on every split of LOO's
    splits, plotting the left out catalyst as a test 
    '''


# should plot average MAE scores of RFECV, SFS, and no feat select options, using std. error.
# as error bars

def compare_selectors(file_path: str, directory: str):
    data = {}
    samples = [0, 0, 0]
    names = ["SFS", "RFECV", "no_select"]

    rfecv = []
    sfs = []
    no_select = []

    with open(file_path) as f:
        temp = pd.read_csv(f, index_col=0)
    print(temp)

    for index, row in temp.iterrows():
        if index.split("_")[-1] == "select":
            no_select.append(row[0])
            samples[2] += 1
        elif index.split("_")[-1] == "RFECV":
            rfecv.append(row[0])
            samples[1] += 1
        elif index.split("_")[-1] == "SFS":
            sfs.append(row[0])
            samples[0] += 1
        else:
            print("Error!\nError!\nError!")

    mae = np.mean([sfs, rfecv, no_select], axis=1)
    dev = np.std([sfs, rfecv, no_select], axis=1)
    errors = dev / np.sqrt(samples)

    data["No Selector"] = [mae[2], dev[2]]
    data["RFECV"] = [mae[1], dev[1]]
    data["SFS"] = [mae[0], dev[0]]

    p_select = two_sample_stats(mae[0], errors[0], mae[1], errors[1], samples[0] + samples[1])
    p_no_select = two_sample_stats(mae[1], errors[1], mae[2], errors[2], samples[1] + samples[2])

    data["RFECV v. SFS"] = p_select
    data["RFECV v. No Selector"] = p_no_select

    with open(directory + "maes_selectors.json", "w") as o:
        json.dump(data, o, indent=2)

    fig, ax = plt.subplots()
    bars = ax.bar(names, mae, yerr=errors, capsize=5, color=["m", "c", "r"])

    ax.set_ylabel('Mean MAE (kcal/mol)')
    ax.set_title('Mean MAE by Feature Selector')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(directory + "MAE_selectors.png")


def compare_detectors(file_path: str, directory: str):
    data = {}
    samples = [0, 0, 0]
    names = ["Isolation Forest", "LOF", ""]

    isof = []
    lof = []
    no_detect = []

    with open(file_path) as f:
        temp = pd.read_csv(f, index_col=0)
    print(temp)

    for index, row in temp.iterrows():
        if index.split("_")[0] == names[0]:
            isof.append(row[0])
            samples[0] += 1
        elif index.split("_")[0] == names[1]:
            lof.append(row[0])
            samples[1] += 1
        elif index.split("_")[0] == names[2]:
            no_detect.append(row[0])
            samples[2] += 1
        else:
            print("Error!\nError!\nError!")

    mae = np.mean([isof, lof, no_detect], axis=1)
    dev = np.std([isof, lof, no_detect], axis=1)
    errors = dev / np.sqrt(samples)

    data["No Detector"] = [mae[2], dev[2]]
    data["LOF"] = [mae[1], dev[1]]
    data["IF"] = [mae[0], dev[0]]

    with open(directory + "maes_detectors.json", "w") as o:
        json.dump(data, o, indent=2)

    fig, ax = plt.subplots()
    bars = ax.bar(names, mae, yerr=errors, capsize=5, color=["m", "c", "r"])

    ax.set_ylabel('Mean MAE (kcal/mol)')
    ax.set_title('Mean MAE by Outlier Detector')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(directory + "MAE_detectors.png")


def compare_ASO_models(file_paths: list, directory: str):
    # order is full reg, half reg, full svr, half svr
    data = {}
    names = ["Ridge", "Lasso", "Elastic Net", "SVR"]
    full = True
    fad = [0, 0, 0, 0]
    had = [0, 0, 0, 0]
    fad_errors = [0] * 4
    had_errors = [0] * 4

    for path in file_paths:
        with open(path) as f:
            temp = pd.read_csv(f, index_col=0)
        print(temp)

        this_MAE = [[], [], [], []]
        this_num = [0, 0, 0, 0]
        for index, row in temp.iterrows():
            if index.split("_")[-1] != "select":    # skip no feature select
                if index.split("_")[1] == "Ridge":
                    this_MAE[0].append(row[0])
                    this_num[0] += 1
                elif index.split("_")[1] == "Lasso":
                    this_MAE[1].append(row[0])
                    this_num[1] += 1
                elif index.split("_")[1] == "Elastic Net":
                    this_MAE[2].append(row[0])
                    this_num[2] += 1
                elif index.split("_")[1] == "Support Vector Machine":
                    this_MAE[3].append(row[0])
                    this_num[3] += 1
                else:
                    print("Error!\nError!\nError!")


        data["FAD"] = {}
        data["HAD"] = {}
        for i, scores in enumerate(this_MAE):
            if this_num[i] != 0:
                if full:
                    fad[i] = np.mean(scores)
                    fad_errors[i] = np.std(scores) / np.sqrt(this_num[i])
                    data["FAD"][names[i]] = [fad[i], fad_errors[i]]
                else:
                    had[i] = np.mean(scores)
                    had_errors[i] = np.std(scores) / np.sqrt(this_num[i])
                    data["HAD"][names[i]] = [had[i], had_errors[i]]

        if full:
            full = False
        else:
            full = True

    p_en = two_sample_stats(fad[2], fad_errors[2], had[2], had_errors[2], 16) # manually counted samples
    data["p_en"] = p_en

    with open(directory + "maes_models.json", "w") as o:
        json.dump(data, o, indent=2)


    fig, ax = plt.subplots()
    ax.bar(names, fad, yerr=fad_errors, capsize=5, width=-0.4, color="blue", label="FAD", align="edge")
    ax.bar(names, had, yerr=had_errors, capsize=5, width=0.4,  color="orange", label="HAD", align="edge")

    ax.set_ylabel('Mean MAE (kcal/mol)')
    ax.set_title('Mean MAE by Model for FAD and HAD Runs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + "MAE_models.png")


def compare_runs(file_paths: list, names: list, directory: str):
    data = {}
    mae = []
    errors = []
    samples = []
    i = 0

    for path in file_paths:
        with open(path) as f:
            temp = pd.read_csv(f, index_col=0)
        print(temp)

        this_MAE = []
        num = 0
        for index, row in temp.iterrows():
            if index.split("_")[-1] != "select":    # skip no feature select
                this_MAE.append(row[0])
                num += 1

        samples.append(num)
        avg = np.mean(this_MAE)
        dev = np.std(this_MAE)
        mae.append(avg)
        errors.append(dev / np.sqrt(num))
        data[names[i]] = [avg, dev]
        i += 1

    p_reg = two_sample_stats(mae[0], errors[0], mae[1], errors[1], samples[0] + samples[1])
    p_svr = two_sample_stats(mae[2], errors[2], mae[3], errors[3], samples[2] + samples[3])

    data["reg_p"] = p_reg
    data["reg_svr"] = p_svr

    with open(directory + "maes.json", "w") as o:
        json.dump(data, o)

    fig, ax = plt.subplots()
    bars = ax.bar(names, mae, yerr=errors, capsize=5, color=["blue", "orange", "blue", "orange"])

    ax.set_ylabel('Mean MAE (kcal/mol)')
    ax.set_title('Bar Chart with Std. Error Bars')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(directory + "MAE_bar_chart.png")


def two_sample_stats(mean1: float, stderror1: float, mean2: float, stderror2: float, tot_samples: int):
    diff = mean1 - mean2
    pooled = np.sqrt(np.power(stderror1, 2) + np.power(stderror2, 2))

    t = diff / pooled
    df = tot_samples - 2

    return t_test.sf(abs(t), df) * 2

if __name__ == "__main__":
    #compare_runs(input_files, names, output_directory)
    #compare_selectors("../aso-modeling-out/_target_fix_reruns/full_regularized/all_results.csv", "../images/chp_1_selectors/")
    #compare_ASO_models(input_files, output_directory)
    compare_detectors("../aso-modeling-out/halved-EN-0.001-0.75/all_results.csv", "../images/c1_outliers_en/")