from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, BaseCrossValidator, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet  # ElasticNetCV?
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.feature_selection import RFECV
from sklearn.neighbors import LocalOutlierFactor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from helpers import pickler, to_json, scrape_selected_model, filter_feats
from copy import deepcopy
from math import log
from itertools import product

# From Isolation Forest on full grid ASO
# this was used for the halved grid ASO runs for the thesis paper
full_aso_outliers = ['171_2_2_17','22_4_4_28','252_1_1_8','90_1_1_17']

# for ridge/lasso/elastic net
alpha_scores = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 10, 100]
l1_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]     # no need to test 0 or 1

# for SVR
epsilons = [0, 0.001, 0.01, 0.1, 0.5, 1, 2] # DEPENDS ON SCALE OF TARGET DATA, THIS IS HARDCODED TO ddG
Cs = [0.01, 0.1, 0.5, 1, 5, 10, 100]

# hyperparams determined manually, implementing automated search later
default_models = [  # TO DO: initialize these correctly
#    ("Ridge", Ridge(1, random_state=42)), # alpha, solver? , {"alpha": alpha_scores}   OLD VALUE: 1
#    ("Lasso", Lasso(0.005, random_state=42)), # alpha? , {"alpha": alpha_scores}   OLD VALUE: 0.005
#    ("Support Vector Machine", SVR(kernel="rbf", epsilon=0.01, C=5))
#    ("Partial Least Squares", PLSRegression(1, scale=False)), # already scaled
#    ("Partial Least Squares Scaled", PLSRegression(1, scale=True))  # REMOVED
    ("Elastic Net", ElasticNet(0.001, l1_ratio=0.75, random_state=42)) # alpha, L1 ratio? , {"alpha": alpha_scores, "l1_ratio": l1_ratios}    # OLD VALUES: 0.0001, 0.5
#    ("Random Forest", RandomForestRegressor(criterion='absolute_error', random_state=42)),  # more params?
#    ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
#    ("Gaussian Process", GaussianProcessRegressor(random_state=42))     # just using base state for now, look into other kernels
]

default_tunings = [
    ("Ridge", Ridge),
    ("Lasso", Lasso)
]

default_metrics = [
    ("MAE", mean_absolute_error, {}),
    ("RMSE", mean_squared_error, {"squared": False}),
    ("R**2", r2_score, {})
]

default_cv = [
#    ("3-Fold", KFold(n_splits=3, shuffle=True, random_state=42)),
    ("LOO", LeaveOneOut())
]

default_selectors = [
#    ("RFECV", RFECV, {}),
    ("SFS", SFS, {}), # select best subset of features between 1 and 50 features (inclusive) (necessary?) k_features: best
]

default_outliers = [
    ("Isolation Forest", IsolationForest, {"random_state": 42}),
#    ("LOF", LocalOutlierFactor, {}),
#    ("", None, {})
]

default_hyperparams = { # for hyperparameter tuning
    "epsilon": epsilons,
    "C": Cs
#    "alpha": alpha_scores,
#    "l1_ratio": l1_ratios
}

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
                 tuning_models: list = default_tunings,
                 selectors: list = default_selectors,
                 detectors: list = default_outliers,
                 hyperparams: dict = default_hyperparams,
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
        self._tuning_models = tuning_models
        self._selectors = selectors
        self._detectors = detectors
        self._n_features = n_features
        self._init_params = init_params     # delete these later, initializing in _init_pipelines now
        self._fit_params = fit_params
        self._n_jobs = n_jobs
        self._hyperparameters = hyperparams
        
        try:
            self._all_pipes = self._init_pipelines(steps, models)
            self._save_steps_or_models(steps, output + "pickled/")
            self._save_steps_or_models(models, output + "pickled/")
            self._save_steps_or_models(cvs, output + "pickled/")
            self._init_jobs(n_jobs)
        except Exception as exp:
            print(f"Error with Pipe creation, please try again: {exp}")


    def _init_pipelines(self, steps: list, models: list) -> tuple: # {"Name of end estimator":pipeline including that estimator}
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
        for _, _, kwargs in self._detectors:
            kwargs["n_jobs"] = n_jobs


    @staticmethod
    def _save_steps_or_models(steps: list, output: str) -> None:    # to "pickled" subdirectory in output directory
        for name, ob in steps:
            name = name.replace(" ", "_")
            pickler(ob, output, name + ".pickle")


    @staticmethod 
    def ee_transform(target: pd.DataFrame) -> None: # assuming temp is room temp, 293 K
        # delta delta G = -RTln(e.r)
        # e.r. = (1 + ee) / (1 - ee)
        # 10.1038/s41586-019-1384-z
        # modifies in place
        for idx, i in target.iterrows():
            dec = i[0] / 100 # convert from xx% to 0.xx
            temp = log((1 + dec) / (1 - dec)) * 293 * -0.0019872036 # gas constant in kcal/mol
            target.loc[idx] = temp

        target.rename(columns={"ee": "ddGTS"}, inplace=True)
        return target
    

    def make_pipe(self, model, model_name: str) -> Pipeline:
        temp = deepcopy(self._steps)
        temp.append((model_name, model))
        return Pipeline(temp)

    
    @property
    def all_pipes(self) -> dict:    # THIS IS ACTUALLY LOOKING UNNECCESARY, REMOVE
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
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)
    
    @hyperparameters.setter
    def hyperparameters(self, new_params: dict) -> None:
        self._hyperparameters = new_params


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
    def models(self) -> list:
        return self._models
    
    @models.setter
    def models(self, new_models: list) -> None:
        try:
            self._models = new_models
            self._all_pipes = self._init_pipelines(self._steps, self._models)
        except Exception as exp:
            print(f"Error with setting new models, please try again: {exp}")

    
    @property
    def tuning_models(self) -> list:
        return self._tuning_models
    
    @tuning_models.setter
    def tuning_models(self, new_tunings: list) -> None:
        self._tuning_models = new_tunings

    
    @property
    def cv(self) -> list:
        return self._cvs
    
    @cv.setter
    def cv(self, new_cvs: list) -> None:
        self._cvs = new_cvs
    

    def feat_rank_RFECV(self, features: pd.DataFrame, target: pd.DataFrame, model, model_name: str, cv: BaseCrossValidator, selector, steps: int | float, min_feat: int, store: bool, kwargs) -> tuple:
        select_model = selector(model, cv=cv, scoring="neg_mean_absolute_error", step=steps, min_features_to_select=min_feat, **kwargs)

        pipe = self.make_pipe(select_model, model_name)
        pipe.fit(features, target.values.ravel())

        select_results = pipe.named_steps[model_name]
        df = features.loc[:, select_results.support_] # selected features dataframe
        if store:
            mean_scores = select_results.cv_results_["mean_test_score"]   # mae of all subsets (1 through n)
            n_feat = select_results.n_features_   # number of selected features
            mae = mean_scores[n_feat - 1]
        else:   # don't bother storing data from whittling features down to sample size step
            mean_scores, n_feat, mae = None, None, None

        return mae, n_feat, mean_scores, df
    

    def feat_rank_SFS(self, features: pd.DataFrame, target: pd.DataFrame, model, model_name: str, cv: BaseCrossValidator, selector, range: tuple, kwargs) -> tuple:
        select_model = selector(model, cv=cv, scoring="neg_mean_absolute_error", k_features=range, **kwargs)

        pipe = self.make_pipe(select_model, model_name)
        pipe.fit(features, target.values.ravel())

        select_results = pipe.named_steps[model_name]
        mean_scores = []
        for _, subset in select_results.subsets_.items():
            mean_scores.append(subset["avg_score"])
        
        df = features.iloc[:, list(select_results.k_feature_idx_)]   # SWITCHED TO IDX INSTEAD OF NAMES (arraylike needed to be set to list i guess)
        n_feat = len(select_results.k_feature_idx_)
        mae = select_results.k_score_

        return mae, n_feat, mean_scores, df


    def feat_rank(self, model, model_name: str, df: pd.DataFrame, target: pd.DataFrame, cv: BaseCrossValidator, selector, kwargs) -> tuple():
        num_samples = len(df)   # max number of features cannot be more than num samples or num columns
        match selector.__qualname__:
            case RFECV.__qualname__:   
                num_feats = len(self._features.columns) 
                if(num_feats > num_samples):
                    _, _, _, df = self.feat_rank_RFECV(df, target, model, model_name, cv, selector, 0.1, num_samples, False, kwargs) # cut down to features to sample size, 10% at a time

                mae, n_feat, mean_scores, subset = self.feat_rank_RFECV(df, target, model, model_name, cv, selector, 1, 1, True, kwargs)
            case SFS.__qualname__:
                range = (1, num_samples)
                mae, n_feat, mean_scores, subset = self.feat_rank_SFS(df, target, model, model_name, cv, selector, range, kwargs)
                mean_scores = np.array(mean_scores)
            case _:
                raise TypeError(f"This feature selection method is not yet implemented: {selector}")

        mean_scores.tofile(self._output + model_name + "_mean_scores.csv", sep = ",")
        return mae, n_feat, mean_scores, subset
    

    def hyperparam_tuning(self, testing_models=None, hyperparams=None): #, features: pd.DataFrame, model, model_name: str, cv: BaseCrossValidator
        """
        Using just RFECV and 3-fold (the fastest of their respective methods),
        Evaluate the END performance of models (scored based on performance after feature selection)
        Not generalizable yet, manually change code depending on models and number of hyperparams
        Can test multiple models, but only if they share the same hyperparameters to be tested
        """
        # JANKY WAY I CHANGED HYPERPARAM_TUNING RUNS FOR MY THESIS
        #testing_models = [("Ridge", Ridge), ("Lasso", Lasso)] # have to do this differently than the way stored in self._models
        #testing_models = [("Elastic Net", ElasticNet)]
        #testing_models = [("SVR", SVR)]

        if testing_models is None:
            testing_models = self._tuning_models

        if hyperparams is None:
            hyperparams = self._hyperparameters

        best_params = {}

        combos = list(product(*hyperparams.values())) # list of tuples, each tuple a combo of possible hyperparams
        param1, param2 = hyperparams.keys()   # currently hardcoded for only 2 hyperparams
        best_params["parameter 1"] = param1
        best_params["parameter 2"] = param2 

        for detector_name, detector, kwargs in self._detectors:
            best_params[detector_name] = {}
            df, target = self.outlier_detection(detector, detector_name, kwargs)
            for name, model in testing_models:
                best_params[detector_name][name] = {}
                best_params[detector_name][name]["best"] = {}
                best_score = float("-inf")
                for val1, val2 in combos:   # manual GridSearchCV
                    if val1 not in best_params[detector_name][name].keys(): # As val1 will be repeated, make sure not to overwrite!
                        best_params[detector_name][name][val1] = {}
                #search = GridSearchCV(model, params, cv=LeaveOneOut(), n_jobs=self._n_jobs, scoring="neg_mean_absolute_error")
                    temp = {param1: val1, param2: val2}     #  "random_state": 42

                    # SFS for SVR because it doesn't work with RFECV
                    # make_pipe inside of feat_rank handles preprocessing steps and pipeline
                    mae, n_feat, mean_scores, subset = self.feat_rank(model(**temp), f"{detector_name}_{name}_{val1}_{val2}", df, target,  KFold(3), RFECV, {}) # KWARGS IS LEFT BLANK

                    #pipe = self.make_pipe(model(**temp), name)     pointless ??
                    #pipe.fit(subset, target.values.ravel())

                    best_params[detector_name][name][val1][val2] = mae  # val2 will be unique for each val1, so no need to check
                    
                    if mae > best_score:
                        best_params[detector_name][name]["best"]["values"] = (val1, val2)
                        best_params[detector_name][name]["best"]["score"] = mae
                        best_score = mae
                        #best = pipe.named_steps[name].best_params_
                        #best_params[detector_name][name]["best params"] = best
                        #best_params[detector_name][name]["best score"] = best = pipe.named_steps[name].best_score_
                    to_json(best_params, self._output + "best_parameters.json")


        to_json(best_params, self._output + "best_parameters.json")
        return best_params
        
        

    def outlier_detection(self, detector, name, kwargs) -> tuple:
        if detector == None:
            return self.features, self.target
        
        if name == "Isolation Forest":
            np.array(full_aso_outliers).tofile(self._output + name + "_outliers.csv", ",")
            df = self.features.drop(full_aso_outliers)
            target = self.target.drop(full_aso_outliers)
            return df, target


        # Skipping below for IF for halved ASO runs
        detector = detector(**kwargs)
        mask = detector.fit_predict(self.features)
        inliers = [val == 1 for val in mask]
        outliers = [val == -1 for val in mask]

        df = self.features.loc[inliers, :]
        target = self.target.loc[inliers, :]
        out_df = self.features.loc[outliers, :]

        out_df.index.to_numpy().tofile(self._output + name + "_outliers.csv", ",")
        return df, target


    def run_all(self) -> pd.DataFrame:
        # prepping results dataframe
        cols = []
        for tup in self._metrics:
            cols.append(tup[0])
        cols.extend(("train R**2", "delta R**2 (test - train)"))
        results = pd.DataFrame(columns=cols)

        best_mae = {}

        for detector_name, detector, detect_kwargs in self._detectors:
            df, target = self.outlier_detection(detector, detector_name, detect_kwargs)
            best_mae[detector_name] = {}
            for name, model in self._models:
                best_mae[detector_name][name] = {}
                for cv_name, cv in self._cvs:
                    best_mae[detector_name][name][cv_name] = {}

    #                model = self.hyperparam_tuning(self.features, model, name, cv)
    #                self._save_steps_or_models([(f"{name}_{cv_name}_first_tune", model)], self._output)

                    for select_name, selector, select_kwargs in self._selectors:
                        long_name = f"{detector_name}_{name}_{cv_name}_{select_name}"
                        mae, n_feat, mean_scores, subset = self.feat_rank(model, long_name, df, target, cv, selector, select_kwargs)
                        best_mae[detector_name][name][cv_name][select_name] = mae
                        plotter([-x for x in mean_scores], n_feat, "MAE scores vs n features", f"{self._output}{long_name}_plot.png")
                        to_json(subset.columns.to_list(), f"{self._output}{long_name}_selected_features.json")

    #                    model = self.hyperparam_tuning(df, model, name, cv)
                        pipe = self.make_pipe(model, name)

                        if cv_name != "LOO":
                            no_feat_select = self.run_pipe(pipe, df, target, cv)
                            temp_results = self.run_pipe(pipe, subset, target, cv)
                        else:
                            no_feat_select = self.run_pipe_loo_r2(pipe, df, target, cv)
                            temp_results = self.run_pipe_loo_r2(pipe, subset, target, cv)
                    
                        self._save_steps_or_models([(long_name, pipe)], self._output + "pickled/pipes/")
                        results.loc[long_name] = temp_results
                        results.loc[f"{detector_name}_{name}_{cv_name}_no_feat_select"] = no_feat_select
                        results.to_csv(self._output + "all_results.csv")    # continuously updating results

        to_json(best_mae, self._output + "feat_select_mae_scores.json")

        return results


    def run_pipe(self, pipe: Pipeline, features: pd.DataFrame, target: pd.DataFrame, cv: BaseCrossValidator) -> dict:
        scorers = {}
        for metric_name, metric, kwargs in self._metrics:
            scorers[metric_name] = make_scorer(metric, **kwargs)

        scores = cross_validate(pipe, features, target.values.ravel(),
                                 cv=cv, scoring=scorers, return_train_score=True, n_jobs=self._n_jobs) # add fit_params later
        
        results = {}
        for metric_name, _ in scorers.items():
            results[metric_name] = np.mean(scores[f"test_{metric_name}"])
        results["train R**2"] = np.mean(scores["train_R**2"])
        results["delta R**2 (test - train)"] = results["R**2"] - results["train R**2"]

        return results

    
    def run_pipe_loo_r2(self, pipe: Pipeline, features: pd.DataFrame, target: pd.DataFrame, cv: BaseCrossValidator) -> float:
        # needed b.c. sklearn's r2 function does not work with LOO cv (calculates r2 with more than 1 sample)
        # circumvents this by calculating r2 from all test folds
        scorers = {}
        for metric_name, metric, kwargs in self._metrics:
            if metric_name != "R**2":
                scorers[metric_name] = make_scorer(metric, **kwargs)

        scores = cross_validate(pipe, features, target.values.ravel(),
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
                         target.iloc[train, :].values.ravel())
            pred = pipe.predict(features.iloc[test, :])

            ytest += target.iloc[test, :].values.ravel().tolist()
            ytest_preds += pred.tolist()

            pred = pipe.predict(features.iloc[train, :])

            ytrain += target.iloc[train, :].values.ravel().tolist()
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
    plt.close()

