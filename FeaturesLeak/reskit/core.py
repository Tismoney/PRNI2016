""" Core classes. """


from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from collections import OrderedDict
from itertools import product
from pandas import DataFrame
from pickle import dump, load
from numpy import mean, std, hstack, zeros
from time import time

import os

import numpy as np


class Pipeliner(object):
    """
    An object which allows you to test different data preprocessing
    pipelines and prediction models at once.

    You will need to specify a name of each preprocessing and prediction
    step and possible objects performing each step. Then Pipeliner will
    combine these steps to different pipelines, excluding forbidden
    combinations; perform experiments according to these steps and present
    results in convenient csv table. For example, for each pipeline's
    classifier, Pipeliner will grid search on cross-validation to find the best
    classifier's parameters and report metric mean and std for each tested
    pipeline. Pipeliner also allows you to cache interim calculations to
    avoid unnecessary recalculations.

    Parameters
    ----------
    steps : list of tuples
        List of (step_name, transformers) tuples, where transformers is a
        list of tuples (step_transformer_name, transformer). ``Pipeliner``
        will create ``plan_table`` from this ``steps``, combining all
        possible combinations of transformers, switching transformers on
        each step.

    eval_cv : int, cross-validation generator or an iterable, optional
        Determines the evaluation cross-validation splitting strategy.
        Possible inputs for cv are:

            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - An object to be used as cross-validation generator.
            - A list or iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y``
        is either binary or multiclass, ``StratifiedKFold`` is used. In all
        other cases, ``KFold`` is used.

        Refer scikit-learn ``User Guide`` for the various cross-validation strategies that
        can be used here.

    grid_cv : int, cross-validation generator or an iterable, optional
         Determines the grid search cross-validation splitting strategy.
         Possible inputs for cv are the same as for ``eval_cv``.

    param_grid : dict of dictionaries
        Dictionary with classifiers names (string) as keys. The keys are
        possible classifiers names in ``steps``. Each key corresponds to
        grid search parameters.

    banned_combos : list of tuples
        List of (transformer_name_1, transformer_name_2) tuples. Each row
        with both transformers will be removed from ``plan_table``.

    Attributes
    ----------
    plan_table : pandas DataFrame
        Plan of pipelines evaluation. Created from ``steps``.

    named_steps: dict of dictionaries
        Dictionary with steps names as keys. Each key corresponds to
        dictionary with transformers names from ``steps`` as keys.
        You can get any transformer object from this dictionary.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from reskit.core import Pipeliner

    >>> X, y = make_classification()
    >>> data = {'X': X, 'y': y}

    >>> scalers = [('minmax', MinMaxScaler()), ('standard', StandardScaler())]
    >>> classifiers = [('LR', LogisticRegression()), ('SVC', SVC())]
    >>> steps = [('Scaler', scalers), ('Classifier', classifiers)]

    >>> grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    >>> eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    >>> param_grid = {'LR' : {'penalty' : ['l1', 'l2']},
    >>>               'SVC' : {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}}

    >>> pipe = Pipeliner(steps, eval_cv=eval_cv, grid_cv=grid_cv, param_grid=param_grid)
    >>> pipe.get_results(data=data, grid_cv=grid_cv, eval_cv=eval_cv, scoring=['roc_auc'])
    """

    def __init__(self, steps, eval_cv=None, grid_cv=None, param_grid=dict(),
                 banned_combos=list()):
        steps = OrderedDict(steps)
        columns = list(steps)
        for column in columns:
            steps[column] = OrderedDict(steps[column])

        def accept_from_banned_combos(row_keys, banned_combo):
            if set(banned_combo) - set(row_keys) == set():
                return False
            else:
                return True

        column_keys = [list(steps[column]) for column in columns]
        plan_rows = list()
        for row_keys in product(*column_keys):
            accept = list()
            for bnnd_cmb in banned_combos:
                accept += [accept_from_banned_combos(row_keys, bnnd_cmb)]

            if all(accept):
                row_of_plan = OrderedDict()
                for column, row_key in zip(columns, row_keys):
                    row_of_plan[column] = row_key
                plan_rows.append(row_of_plan)

        self.plan_table = DataFrame().from_dict(plan_rows)[columns]
        self.named_steps = steps
        self.eval_cv = eval_cv
        self.grid_cv = grid_cv
        self.param_grid = param_grid
        self._cached_data = OrderedDict()
        self.best_params = dict()
        self.scores = dict()

    def get_results(self, data, caching_steps=list(), scoring='accuracy',
            results_file='results.csv', logs_file='results.log', collect_n=None):
        """ 


        Parameters:
        -----------
        data : 

        caching_steps : 

        scoring : 

        results_file : 

        logs_file : 
        
        collect_n :


        Returns:
        --------

        """
        if type(scoring) == str:
            scoring = [scoring]

        columns = list(self.plan_table.columns)
        without_caching = [step for step in columns
                                if step not in caching_steps]

        for metric in scoring:
            grid_steps = ['grid_{}_mean'.format(metric),
                          'grid_{}_std'.format(metric),
                          'grid_{}_best_params'.format(metric)]

            eval_steps = ['eval_{}_mean'.format(metric),
                          'eval_{}_std'.format(metric),
                          'eval_{}_scores'.format(metric)]

            columns += grid_steps + eval_steps

        results = DataFrame(columns=columns)

        try:
            os.remove(results_file)
            print('Removed previous results file -- {}.'.format(results_file))
        except:
            print('No previous results found.')
        if results_file != None:
            results.to_csv(results_file)

        columns = list(self.plan_table.columns)
        results[columns] = self.plan_table


        with open(logs_file, 'w+') as logs:
            N = len(self.plan_table.index)
            for idx in self.plan_table.index:
                print('Line: {}/{}'.format(idx + 1, N))
                logs.write('Line: {}/{}\n'.format(idx + 1, N))
                logs.write('{}\n'.format(str(self.plan_table.loc[idx])))
                row = self.plan_table.loc[idx]
                caching_keys = list(row[caching_steps].values)

                time_point = time()
                if caching_keys != []:
                    X_featured, y = self.transform_with_caching(data, caching_keys)
                else:
                    X_featured = data['X']
                    y = data['y']
                spent_time = round(time() - time_point, 3)
                logs.write('Got Features: {} sec\n'.format(spent_time))

                for metric in scoring:
                    #############################################
                    np.savetxt('test_my.txt', X_featured)
                    #############################################
                    logs.write('Scoring: {}\n'.format(metric))
                    ml_keys = list(row[without_caching].values)
                    time_point = time()
                    grid_res = self.get_grid_search_results(X_featured, y,
                                                            ml_keys,
                                                            metric)
                    spent_time = round(time() - time_point, 3)
                    logs.write('Grid Search: {} sec\n'.format(spent_time))
                    logs.write('Grid Search Results: {}\n'.format(grid_res))

                    for key, value in grid_res.items():
                        results.loc[idx][key] = value

                    time_point = time()
                    scores = self.get_scores(X_featured, y,
                                             ml_keys,
                                             metric,
                                             collect_n)
                    spent_time = round(time() - time_point, 3)
                    logs.write('Got Scores: {} sec\n'.format(spent_time))

                    mean_key = 'eval_{}_mean'.format(metric)
                    scores_mean = mean(scores)
                    results.loc[idx][mean_key] = scores_mean
                    logs.write('Scores mean: {}\n'.format(scores_mean))

                    std_key = 'eval_{}_std'.format(metric)
                    scores_std = std(scores)
                    results.loc[idx][std_key] = scores_std
                    logs.write('Scores std: {}\n'.format(scores_std))

                    scores_key = 'eval_{}_scores'.format(metric)
                    results.loc[idx][scores_key] = str(scores)
                    logs.write('Scores: {}\n\n'.format(str(scores)))
                results.loc[[idx]].to_csv(results_file, header=False, mode='a+')

        return results

    def transform_with_caching(self, data, row_keys):
        """
        Takes ``data`` and sends in to pipeliner from ``plan_table`` with
        transformers names as in ``row_keys`` list.

        Parameters
        ----------
        data : any type transformer can handle
            A data on which you want to transform using transformers
            from ``named_steps``.

        row_keys : list of strings
            List of transformers names. ``Pipeliner`` takes
            transformers from ``named_steps`` using keys from
            ``row_keys`` and creates pipeline to transform.

        Returns
        -------
        transformed_data : (X, y) tuple, where X and y array-like
            Data transformed corresponding to pipeline, created from
            ``row_keys``, to (X, y) tuple.
        """
        columns = list(self.plan_table.columns[:len(row_keys)])

        def remove_unmatched_caching_data(row_keys):
            cached_keys = list(self._cached_data)
            unmatched_caching_keys = list(self._cached_data)
            for row_key, cached_key in zip(row_keys, cached_keys):
                if not row_key == cached_key:
                    break
                unmatched_caching_keys.remove(row_key)

            for unmatched_caching_key in unmatched_caching_keys:
                del self._cached_data[unmatched_caching_key]

        def transform_data_from_last_cached(row_keys, columns):
            prev_key = list(self._cached_data)[-1]
            for row_key, column in zip(row_keys, columns):
                transformer = self.named_steps[column][row_key]
                data = self._cached_data[prev_key]
                self._cached_data[row_key] = transformer.fit_transform(data)
                prev_key = row_key

        if 'init' not in self._cached_data:
            self._cached_data['init'] = data
            transform_data_from_last_cached(row_keys, columns)
        else:
            row_keys = ['init'] + row_keys
            columns = ['init'] + columns
            remove_unmatched_caching_data(row_keys)
            cached_keys = list(self._cached_data)
            cached_keys_length = len(cached_keys)
            for i in range(cached_keys_length):
                del row_keys[0]
                del columns[0]
            transform_data_from_last_cached(row_keys, columns)

        last_cached_key = list(self._cached_data)[-1]

        return self._cached_data[last_cached_key]

    def get_grid_search_results(self, X, y, row_keys, scoring):
        """
        Make grid search for pipeline, created from ``row_keys`` for
        defined ``scoring``.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression; None
            for unsupervised learning.

        row_keys : list of strings
            List of transformers names. ``Pipeliner`` takes transformers
            from ``named_steps`` using keys from ``row_keys`` and creates
            pipeline to transform.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer
            callable object / function with signature
            ``scorer(estimator, X, y)``. If None, the score method of the
            estimator is used.

        Returns
        -------
        results : dict
            Dictionary with keys: 'grid_{}_mean', 'grid_{}_std' and
            'grid_{}_best_params'. In the middle of keys will be
            corresponding scoring.
        """
        classifier_key = row_keys[-1]
        if classifier_key in self.param_grid:
            columns = list(self.plan_table.columns)[-len(row_keys):]

            steps = list()
            for row_key, column in zip(row_keys, columns):
                steps.append((row_key, self.named_steps[column][row_key]))

            param_grid = dict()
            for key, value in self.param_grid[classifier_key].items():
                param_grid['{}__{}'.format(classifier_key, key)] = value

            self.asdf = param_grid
            self.asdfasdf = self.param_grid[classifier_key]

            grid_clf = GridSearchCV(estimator=Pipeline(steps),
                                    param_grid=param_grid,
                                    scoring=scoring,
                                    n_jobs=-1,
                                    cv=self.grid_cv)
            grid_clf.fit(X, y)

            best_params = dict()
            classifier_key_len = len(classifier_key)
            for key, value in grid_clf.best_params_.items():
                key = key[classifier_key_len + 2:]
                best_params[key] = value
            param_key = ''.join(row_keys) + str(scoring)
            self.best_params[param_key] = best_params

            results = dict()
            for i, params in enumerate(grid_clf.cv_results_['params']):
                if params == grid_clf.best_params_:

                    k = 'grid_{}_mean'.format(scoring)
                    results[k] = grid_clf.cv_results_['mean_test_score'][i]

                    k = 'grid_{}_std'.format(scoring)
                    results[k] = grid_clf.cv_results_['std_test_score'][i]

                    k = 'grid_{}_best_params'.format(scoring)
                    results[k] = str(best_params)

            return results
        else:
            param_key = ''.join(row_keys) + str(scoring)
            self.best_params[param_key] = dict()
            results = dict()
            results['grid_{}_mean'.format(scoring)] = 'NaN'
            results['grid_{}_std'.format(scoring)] = 'NaN'
            results['grid_{}_best_params'.format(scoring)] = 'NaN'
            return results

    def get_scores(self, X, y, row_keys, scoring, collect_n=None):
        """
        Gives scores for prediction on cross-validation.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression; None
            for unsupervised learning.

        row_keys : list of strings
            List of transformers names. ``Pipeliner`` takes transformers
            from ``named_steps`` using keys from ``row_keys`` and creates
            pipeline to transform.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer
            callable object / function with signature
            ``scorer(estimator, X, y)``. If None, the score method of the
            estimator is used.

        collect_n : list of strings
            List of keys from data dictionary you want to collect and
            create feature vectors.

        Returns
        -------
        scores : array-like
            Scores calculated on cross-validation.
        """
        columns = list(self.plan_table.columns)[-len(row_keys):]
        param_key = ''.join(row_keys) + str(scoring)

        steps = list()
        for row_key, column in zip(row_keys, columns):
            steps.append((row_key, self.named_steps[column][row_key]))

        steps[-1][1].set_params(**self.best_params[param_key])

        if not collect_n:
            scores = cross_val_score(Pipeline(steps), X, y,
                                     scoring=scoring,
                                     cv=self.eval_cv,
                                     n_jobs=-1)
        else:
            init_random_state = self.eval_cv.random_state
            scores = list()
            for i in range(collect_n):
                fold_prediction = cross_val_predict(Pipeline(steps), X, y,
                                                    cv=self.eval_cv,
                                                    n_jobs=-1)
                metric = check_scoring(steps[-1][1],
                                       scoring=scoring).__dict__['_score_func']
                scores.append(metric(y, fold_prediction))
                #score_n = cross_val_score(Pipeline(steps), X, y,
                #                         scoring=scoring,
                #                         cv=self.eval_cv,
                #                         n_jobs=-1)
                #scores.append(mean(score_n))
                
                self.eval_cv.random_state += 1

            self.eval_cv.random_state = init_random_state
        return scores


class Transformer(TransformerMixin, BaseEstimator):
    """
    Helps to add you own transformation through usual functions.

    Parameters
    ----------
    func : function
        Function that transforms input data.

    params : dict
        Parameters for the function.

    collect : list of strings
        Takes values by keys in collect from input data dictionary.
    """

    def __init__(self, func, params=None, collect=None):
        self.func = func
        self.params = params
        self.collect = collect

    def fit(self, data, target=None, **fit_params):
        """
        Fits the data.

        Parameters
        ----------
        data : dict
            Dictionary with keys ``X`` and ``y``. You can store other
            needed staff here.
        """
        return self

    def transform(self, data):
        """
        Transforms the data according to function you set.

        Parameters
        ----------
        data : dict
            Dictionary with keys ``X`` and ``y``. You can store other
            needed staff here.
        """
        if isinstance(data, dict):
            data = data.copy()

        if self.params:
            result = self.func(**self.params)
        else:
            result = self.func(data)

        if self.collect:
            y = result['y']
            X = result[self.collect[0]]

            for key in self.collect[1:]:
                X = hstack((X, result[key]))
            return X, y

        return result

__all__ = ['Transformer', 'Pipeliner']
