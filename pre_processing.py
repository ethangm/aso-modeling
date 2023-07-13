import pandas as pd
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from pathlib import Path
from helpers import pickler

# min-max scaling and yeo-johnson transformation

def scale(df: pd.DataFrame, output: str | Path) -> pd.DataFrame:
    '''MinMaxScaler is saved as json file to given filepath'''
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        sdf = df
        sdf.iloc[:, :] = scaled


        pickler(scaler, output)

        return sdf
    except Exception as exp:
        print(f"Error with MinMaxScaler: {exp}")


def transform(target: pd.DataFrame, output: str | Path) -> pd.DataFrame:
    '''PowerTransformer is saved as json file to given filepath'''
    try:
        transformer = PowerTransformer()
        values = transformer.fit_transform(target)

        tdf = target
        tdf.iloc[:, 0] = values
        tdf.rename(columns={"ee": "ee transformed"}, inplace=True)

        pickler(transformer, output)

        return tdf
    except Exception as exp:
        print(f"Error with PowerTransformer: {exp}")
