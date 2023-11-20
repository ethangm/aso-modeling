import sys
import numpy as np
from pathlib import Path
from helpers import unpack_data #, scrape_selected, bar_chart, feature_region_selectivity
from time import time
import random

from sklearn.preprocessing import PowerTransformer, MinMaxScaler

from pipes import Pipes


def model(feature_data: str | Path, target_data: str | Path, output: str | Path):
    '''
    feature and target data as filepaths to csv files, output as filepath to directory for output.
    This script will automatically unpack aso and ee data, run minmax scaling and yeo-johnson
    transformation, ANOVA f testing for feature selection, train/test splits and LOO cross validation
    selection, then linear regression modelling with lasso, ridge, randomforest, svm, gradient boosting,
    and pls methods.
    '''
    
    # unpacking feature and target data into dataframes
    aso_df, ee_df = unpack_data(feature_data, target_data)
    print("ee data (" + str(len(ee_df.index.to_list())) + " catalysts)")
    print(ee_df)
    print("\naso data (" + str(len(aso_df.index.to_list())) + " catalysts)")
    print(aso_df)
    Pipes.ee_transform(ee_df)
    print("ddG data (" + str(len(ee_df.index.to_list())) + " catalysts)")
    print(ee_df)
#    ee_df.to_csv("aso-ddg-data.csv")

    steps = [("Power Transformer", PowerTransformer()), ("MinMax Scaler", MinMaxScaler())]
    all_pipes = Pipes(aso_df, ee_df, output, steps, n_jobs = 20)

    start = time()
    print(all_pipes.run_all())                 
#    print(all_pipes.hyperparam_tuning())
    end = time()
    total = end - start

    minutes = total // 60

    hours = minutes // 60
    minutes = minutes % 60

    seconds = total % 60

    time_result = f"{hours} h, {minutes} m, and {seconds} s elapsed"
    print(time_result)
    
    with open(output + "time.txt", "w") as f:
          f.write(time_result)



if __name__ == "__main__":
#    try:
        #temp = str(Path.cwd())
        #model(temp + "/nbo-cu-box-clean/halve_testing/halved_aso_vt0_nocorr950_redone_grid.csv", temp + "/aso-modeling/aso-ee-data.csv", temp + "/aso-modeling-out/grid-search-2-test/")
        model(sys.argv[1], sys.argv[2], sys.argv[3])
#    except IndexError as exp:
#        print("Enter three inputs: feature data csv filepath, target data csv filepath, and desired directory for output (include /)"
#              f": {exp}")
'''
        features = scrape_selected(sys.argv[1])
        print(features)
        print(len(features))
        bar_chart(features, sys.argv[1], cutoff=5)

        grid = np.load(sys.argv[2])

        region_1 = feature_region_selectivity(features, grid, sys.argv[1], 1)
        bar_chart(region_1, sys.argv[1], "feature_selection_IDW_1", 7)
        print(region_1)

        region_2 = feature_region_selectivity(features, grid, sys.argv[1])
        bar_chart(region_2, sys.argv[1], "feature_selection_IDW_2", 7)
        print(region_2)

        region_3 = feature_region_selectivity(features, grid, sys.argv[1], 3)
        bar_chart(region_3, sys.argv[1], "feature_selection_IDW_3", 7)
        print(region_3)

        differences = []
        for i, value in enumerate(region_1):
              differences.append((value[0].astype(int), value[1] - region_2[i][1], value[1] - region_3[i][1]))

        print("gridpoint | p=1 - p=2 | p=1 - p=3")
        for j in differences:
              print(j)
'''