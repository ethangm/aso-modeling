from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, BaseCrossValidator
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from helpers import pickler, to_json
from copy import deepcopy
from math import log



default_models = [  # TO DO: initialize these correctly
    ("Ridge", Ridge(random_state=42)), # alpha, solver?
    ("Lasso", Lasso(random_state=42)), # alpha?
#    ("Random Forest", RandomForestRegressor(random_state=42)), # n_estimators, max_depth, etc?
#    ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    ("Support Vector Machine", SVR(kernel="linear")),
    ("Partial Least Squares", PLSRegression(1, scale=False)) # already scaled
]

default_metrics = [
    ("MAE", mean_absolute_error, {}),
    ("mae", mean_squared_error, {"squared": False}),
    ("R**2", r2_score, {})
]

default_cv = [
    ("3-Fold", KFold(n_splits=3, shuffle=True, random_state=42)),
    ("LOO", LeaveOneOut())
]

default_selectors = [
    ("RFECV", RFECV, {}),
    ("SFS", SFS, {}), # select best subset of features between 1 and 50 features (inclusive) (necessary?) k_features: best
]


class Pipes:    # ADD PARALLELIZATION
    '''
    Please set random states for any applicable steps before passing to Pipes
    '''


    # Look into memory caching between different pipelines
    # Check validity of inputs. e.g. make sure features and target indexes are the same, and sort them both
    def __init__(self,
                 features: pd.DataFrame,
                 target: pd.DataFrame,
                 output: str,   
                 steps: list,   # (ALL STEPS EXCEPT FOR END MODEL)
                 metrics: list = default_metrics,
                 cvs: list = default_cv,
                 models: list = default_models,
                 selectors: list = default_selectors,
                 n_features: int | None = None, # defaults to half of features
                 init_params: dict = None, # keys must be in format *model name*__*parameter name*
                 fit_params: dict = None, # to be passed during the fit step. Format is {"step_name":{"param_name":value}} (DONT INCLUDE FINAL ESTIMATOR STEP)
                 n_jobs: int = 1
                ) -> None:
        
        self._init_data(features, target)
        self._output = output
        self._steps = steps
        self._metrics = metrics
        self._cvs = cvs
        self._models = models
        self._selectors = selectors
        self._n_features = n_features
        self._init_params = init_params
        self._fit_params = fit_params
        self._n_jobs = n_jobs
        
        try:
            self._all_pipes = self._init_pipelines(steps, models)
            self._save_steps_or_models(steps, output)
            self._save_steps_or_models(models, output)
            self._save_steps_or_models(cvs, output)
            self._init_jobs(n_jobs)
        except Exception as exp:
            print(f"Error with Pipe creation, please try again: {exp}")


    def _init_pipelines(self, steps: list, models: list) -> dict: # {"Name of end estimator":pipeline including that estimator}
        results = {}
        for name, model in models:
            temp = steps.copy()
            temp.append((name, model))
            results[name] = Pipeline(temp)

        return results
    

    def _init_data(self, features: pd.DataFrame, target: pd.DataFrame):
        try:
            features = features.sort_index(axis=0)
            target = target.sort_index(axis=0)
            if np.array_equal(features.index, target.index):
                self._features = features
                self._target = target
            else:
                raise IndexError
        except Exception as exp:
            print(f"Invalid features/target dataframe (make sure that their samples match): {exp}")


    def _init_jobs(self, n_jobs: int):
        for _, _, kwargs in self._selectors:
            kwargs["n_jobs"] = n_jobs


    @staticmethod
    def _save_steps_or_models(steps: list, output: str) -> None:    # to "pickled" subdirectory in output directory
        for name, ob in steps:
            name = name.replace(" ", "_")
            pickler(ob, output + "pickled/", name + ".pickle")


    @staticmethod
    def ee_transform(target: pd.DataFrame) -> None: # assuming temp is room temp, 293 K
        # delta delta G = -RTln(abs(ee))
        # modifies in place
        for idx, i in target.iterrows():
            temp = log(abs(i[0])) * 293 * -0.0019872
            target.loc[idx] = temp

        target.rename(columns={"ee": "ddGTS"}, inplace=True)
        return target

    
    @property
    def all_pipes(self) -> dict:
        return deepcopy(self._all_pipes)
    
    @all_pipes.setter
    def all_pipes(self, new_dict: dict) -> None:
        try:
            for _, i in new_dict.keys():
                if not isinstance(i, Pipeline):
                    raise Exception
        except Exception:
            print("Invalid new all_pipes. Must be dict of sklearn Pipelines")
        else:
            self._all_pipes = new_dict

    
    @property
    def steps(self) -> list:
        return self._steps
    
    @steps.setter
    def steps(self, new_steps: list) -> None:
        try:
            self._steps = new_steps
            self.all_pipes = self._init_pipelines(new_steps, self._models)
        except Exception as exp:
            print(f"Error with setting new steps, please try again: {exp}")
    

    @property
    def output(self) -> str:
        return self._output
    
    @output.setter
    def output(self, new_output: str) -> None:
        self._output = new_output


    @property
    def features(self) -> pd.DataFrame:
        return self._features
    
    @features.setter
    def features(self, new_feat: pd.DataFrame) -> None:
        self._features = new_feat


    @property
    def target(self) -> pd.DataFrame:
        return self._target
    
    @target.setter
    def target(self, new_target: pd.DataFrame) -> None:
        self._target = new_target

    
    @property
    def cv(self) -> list:
        return self._cvs
    
    @cv.setter
    def cv(self, new_cvs: list) -> None:
        self._cvs = new_cvs


    def run_all_feat_rank(self) -> pd.DataFrame: # UPDATE
        '''
        Selects selects features from best MSE score of all combinations of RFECV/SFS, CVs, and models
        (ONLY RFECV IMPLEMENTED)
        '''
        df = pd.DataFrame
        best_score = -float('inf')
        num = int
        for name, model in self._models:
            for cv_name, cv in self._cvs:
                new_name = name + "_RFECV_" + cv_name
                mae, n_feat, mean_scores, temp_df = self.feat_rank(model, new_name, cv, RFECV)
                if mae > best_score:
                    best_score = mae
                    num = n_feat
                    scores = mean_scores
                    df = temp_df
                    best_name = new_name

        plotter(scores.tolist(),
                num,
                best_name + " MAE scores vs n_features",
                self._output + "plot.png")
        return df
    

    def feat_rank_RFECV(self, features: pd.DataFrame, model, model_name: str, cv: BaseCrossValidator, selector, steps: int | float, min_feat: int, store: bool, kwargs) -> tuple:
        select_model = selector(model, cv=cv, scoring="neg_mean_absolute_error", step=steps, min_features_to_select=min_feat, **kwargs)
        temp = deepcopy(self._steps)
        temp.append((model_name, select_model))
        print(select_model) # REMOVE

        pipe = Pipeline(temp)
        pipe.fit(features, self._target.values.ravel())

        select_results = pipe.named_steps[model_name]
        df = features.loc[:, select_results.support_] # selected features dataframe
        if store:
            mean_scores = select_results.cv_results_["mean_test_score"]   # mae of all subsets (1 through n)
            n_feat = select_results.n_features_   # number of selected features
            mae = mean_scores[n_feat - 1]
        else:   # don't bother storing data from whittling features down to sample size step
            mean_scores, n_feat, mae = None, None, None

        return mae, n_feat, mean_scores, df
    

    def feat_rank_SFS(self, model, model_name: str, cv: BaseCrossValidator, selector, range: tuple, kwargs) -> tuple:
        select_model = selector(model, cv=cv, scoring="neg_mean_absolute_error", k_features=range, **kwargs)
        temp = deepcopy(self._steps)
        temp.append((model_name, select_model))
        print(select_model) # REMOVE

        pipe = Pipeline(temp)
        pipe.fit(self._features, self._target.values.ravel())

        select_results = pipe.named_steps[model_name]
        mean_scores = []
        for _, subset in select_results.subsets_.items():
            mean_scores.append(subset["avg_score"])
        df = self._features.loc[:, select_results.k_feature_names_]
        n_feat = len(select_results.k_feature_names_)
        mae = select_results.k_score_

        return mae, n_feat, mean_scores, df


    def feat_rank(self, model, model_name: str, cv: BaseCrossValidator, selector, kwargs) -> tuple():
        num_samples = len(self._features)

        match selector.__qualname__:
            case RFECV.__qualname__:   
                num_feats = len(self._features.columns) 
                if(num_feats > num_samples):
                    _, _, _, df = self.feat_rank_RFECV(self._features, model, model_name, cv, selector, 0.1, num_samples, False, kwargs) # cut down to features to sample size, 10% at a time
                else:
                    df = self._features

                mae, n_feat, mean_scores, subset = self.feat_rank_RFECV(df, model, model_name, cv, selector, 1, 1, True, kwargs)
            case SFS.__qualname__:
                range = (1, num_samples)
                mae, n_feat, mean_scores, subset = self.feat_rank_SFS(model, model_name, cv, selector, range, kwargs)
                mean_scores = np.array(mean_scores)
            case _:
                raise TypeError(f"This feature selection method is not yet implemented: {selector}")

#        df = self._features.loc[:, pipe.named_steps[model_name].support_] # selected features dataframe
#        n_feat = pipe.named_steps[model_name].n_features_   # number of selected features
#        mae = mean_scores[n_feat]   # mae of selected features

        mean_scores.tofile(self._output + model_name + "_mean_scores.csv", sep = ",")

        return mae, n_feat, mean_scores, subset


    def run_all(self) -> pd.DataFrame:
        cols = []
        for tup in self._metrics:
            cols.append(tup[0])
        cols.extend(("train R**2", "delta R**2 (test - train)"))
        results = pd.DataFrame(columns=cols)

        best_mae = {}

        for name, model in self._models:
            best_mae[name] = {}
            for cv_name, cv in self._cvs:
                best_mae[name][cv_name] = {}
                for select_name, selector, select_kwargs in self._selectors:
                    print(selector)     # REMOVE
                    long_name = f"{name}_{cv_name}_{select_name}"
                    mae, n_feat, mean_scores, df = self.feat_rank(model, long_name, cv, selector, select_kwargs)
                    best_mae[name][cv_name][select_name] = mae
                    plotter([-x for x in mean_scores], n_feat, "MAE scores vs n features", f"{self._output}{long_name}_plot.png")
                    to_json(df.columns.to_list(), f"{self._output}{long_name}_selected_features.json")
                    pipe = self.all_pipes[name]

                    if cv_name != "LOO":
                        temp_results = self.run_pipe(pipe, df, cv)
                    else:
                        temp_results = self.run_pipe_loo_r2(pipe, df, cv)
                   

                    results.loc[long_name] = temp_results
                    results.to_csv(self._output + "all_results.csv")    # continuously updating results

        to_json(best_mae, self._output + "feat_select_mae_scores.json")

        return results
    

    def run_all_best_selector(self) -> pd.DataFrame:    # TESTING IDEA (ehh, stick to run_all)
        cols = []
        for tup in self._metrics:
            cols.append(tup[0])
        cols.extend(("train R**2", "delta R**2 (test - train)"))
        results = pd.DataFrame(columns=cols)

        best_mae = {}

        for name, model in self._models:
            best_mae[name] = {}
            for cv_name, cv in self._cvs:
                best_mae[name][cv_name] = {}
                best_val = -float("inf")
                best_df = pd.DataFrame
                for select_name, selector, select_kwargs in self._selectors:
                    long_name = f"{name}_{cv_name}_{select_name}"
                    mae, n_feat, mean_scores, df = self.feat_rank(model, long_name, cv, selector, select_kwargs)
                    best_mae[name][cv_name][select_name] = mae
                    plotter([-x for x in mean_scores], n_feat, "MAE scores vs n features", f"{self._output}{long_name}_plot.png")
                    if mae > best_val:
                        best_val = mae
                        best_df = df

                pipe = self.all_pipes[name]

                if cv_name != "LOO":
                    temp_results = self.run_pipe(pipe, best_df, cv)
                else:
                    temp_results = self.run_pipe_loo_r2(pipe, best_df, cv)
                

                results.loc[long_name] = temp_results

        results.to_csv(self._output + "all_results.csv")
        to_json(best_mae, self._output + "feat_select_mae_scores.json")

        return results



    def run_pipe(self, pipe: Pipeline, features: pd.DataFrame, cv: BaseCrossValidator) -> dict:
        scorers = {}
        for metric_name, metric, kwargs in self._metrics:
            scorers[metric_name] = make_scorer(metric, **kwargs)

        scores = cross_validate(pipe, features, self._target.values.ravel(),
                                 cv=cv, scoring=scorers, return_train_score=True, n_jobs=self._n_jobs) # add fit_params later
        
        results = {}
        for metric_name, _ in scorers.items():
            results[metric_name] = np.mean(scores[f"test_{metric_name}"])
        results["train R**2"] = np.mean(scores["train_R**2"])
        results["delta R**2 (test - train)"] = results["R**2"] - results["train R**2"]

        return results

    
    def run_pipe_loo_r2(self, pipe: Pipeline, features: pd.DataFrame, cv: BaseCrossValidator) -> float:
        scorers = {}
        for metric_name, metric, kwargs in self._metrics:
            if metric_name != "R**2":
                scorers[metric_name] = make_scorer(metric, **kwargs)

        scores = cross_validate(pipe, features, self._target.values.ravel(),
                                 cv=cv, scoring=scorers, return_train_score=True, n_jobs=self._n_jobs) # add fit_params later
        
        results = {}
        for metric_name, _ in scorers.items():
            results[metric_name] = np.mean(scores[f"test_{metric_name}"])


        ytest = []
        ytest_preds = []
        ytrain = []
        ytrain_preds = []
        for _, (train, test) in enumerate(cv.split(features)): # this takes a while
            pipe.fit(features.iloc[train, :],
                         self._target.iloc[train, :].values.ravel())
            pred = pipe.predict(features.iloc[test, :])

            ytest += self._target.iloc[test, :].values.ravel().tolist()
            ytest_preds += pred.tolist()

            pred = pipe.predict(features.iloc[train, :])

            ytrain += self._target.iloc[train, :].values.ravel().tolist()
            ytrain_preds += pred.tolist()

        results["R**2"] = r2_score(ytest, ytest_preds)
        results["train R**2"] = r2_score(ytrain, ytrain_preds)
        results["delta R**2 (test - train)"] = results["R**2"] - results["train R**2"]

        return results

                    


def plotter(values: list, selected: int, name: str, savepath: str):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.grid(False)
    plt.xlabel("n_features")
    plt.ylabel("MAE scores")
    plt.title(name + f"({selected} features selected)")

    for i, score in enumerate(values):
        if i == selected - 1:
            ax.scatter([i], score, c="blue", alpha=1, zorder=2)
        else:
            ax.scatter([i], score, c="red", alpha=1, zorder=1)

    fig.savefig(savepath)


    # TO-DO: Functionality for sorting and cleaning feature/target data