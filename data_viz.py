import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
import pandas as pd

def ddG_plotter(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame):
    '''
    Compares actual vs. predicted ddG by training pipe on every split of LOO's
    splits, plotting the left out catalyst as a test 
    '''