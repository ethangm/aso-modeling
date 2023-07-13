import sys
from pathlib import Path
from helpers import unpack_data
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
    ee_df, aso_df = unpack_data(feature_data, target_data)
    print("ee data (" + str(len(ee_df.index.to_list())) + " catalysts)")
    print(ee_df)
    print("\naso data (" + str(len(aso_df.index.to_list())) + " catalysts) (PCA projection)")
    print(aso_df)

    steps = [("Power Transformer", PowerTransformer()), ("MinMax Scaler", MinMaxScaler())]
    all_pipes = Pipes(aso_df, ee_df, output, steps, n_jobs = 20)
    
    start = time()
    print(all_pipes.run_all())
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
        #model(temp + "/aso-modeling/pca_projections/aso_vt0_nocorr950pca200.csv", temp + "/aso-modeling/aso-ee-data.csv", temp + "/aso-modeling/output/OOP_rfecv_all/")
        model(sys.argv[1], sys.argv[2], sys.argv[3])
#    except IndexError as exp:
#        print("Enter three inputs: feature data csv filepath, target data csv filepath, and desired directory for output (include /)"
#              f": {exp}")