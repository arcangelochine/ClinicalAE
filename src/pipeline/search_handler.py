from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, HalvingGridSearchCV, HalvingRandomSearchCV


class GridSearchHandler:
    def __init__(self, classifiers, param_grids, seed=42, cv=5, verbose=0, metric='f1_macro'):
        self.classifiers = classifiers
        self.param_grids = param_grids
        self.cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        self.verbose = verbose
        self.metric = metric

    def perform_search(self, x_tune, y_tune, refit):
        results = {}
        best_classifiers = []

        for clf, param_grid in zip(self.classifiers, self.param_grids):
            search = HalvingGridSearchCV(
                estimator=clf,
                param_grid=param_grid,
                cv=self.cv_strategy,
                min_resources=int(0.05 * len(y_tune)),
                scoring=self.metric,
                refit=refit,
                error_score='raise',
                random_state=42,
                n_jobs=-1,
                verbose=self.verbose
            )
            search.fit(x_tune, y_tune)
            results[clf.__class__.__name__] = (search.best_params_, search.best_score_)

            if refit:
                best_classifiers.append(search.best_estimator_)
            else:
                best_classifiers.append(clf)

        if refit:
            return best_classifiers, results
        return results


class RandomSearchHandler:
    def __init__(self, classifiers, param_distributions, seed=42, cv=5, verbose=0, metric='f1_macro'):
        self.classifiers = classifiers
        self.param_distributions = param_distributions
        self.cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        self.verbose = verbose
        self.metric = metric
        self.seed = seed

    def perform_search(self, x_tune, y_tune, refit):
        results = {}
        best_classifiers = []

        for clf, param_grid in zip(self.classifiers, self.param_distributions):
            search = HalvingRandomSearchCV(
                estimator=clf,
                param_distributions=param_grid,
                cv=self.cv_strategy,
                min_resources=int(0.05 * len(y_tune)),
                scoring=self.metric,
                refit=refit,
                error_score='raise',
                random_state=42,
                n_jobs=-1,
                verbose=self.verbose
            )
            search.fit(x_tune, y_tune)
            results[clf.__class__.__name__] = (search.best_params_, search.best_score_)

            if refit:
                best_classifiers.append(search.best_estimator_)

        if refit:
            return best_classifiers, results
        return results


__all__ = ['GridSearchHandler', 'RandomSearchHandler']
