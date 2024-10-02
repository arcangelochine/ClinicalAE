from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

from .search_handler import GridSearchHandler, RandomSearchHandler

type CLF = RandomForestClassifier or AdaBoostClassifier or SVC or LinearRegression or MLPClassifier
type SRC_HANDLER = GridSearchHandler or RandomSearchHandler


class ClassifierHandler:
    def __init__(self, classifiers: list[CLF], seed: int = 42, verbose: int = 0):
        assert len(classifiers) > 0, "No classifier provided"
        self.classifiers = classifiers
        self.seed = seed
        self.verbose = verbose

        self._param_set = False
        self._clfs_fit = False

    def set_params(self, params: list[dict]):
        if len(params) != len(self.classifiers):
            raise ValueError("Number of params must match number of classifiers.")
        for clf, param_grid in zip(self.classifiers, params):
            clf.set_params(**param_grid)
        self._param_set = True

    def fit(self, x_train, y_train):
        if self.verbose and not self._param_set:
            print("You may want to set classifiers params first.")
        for clf in self.classifiers:
            clf.fit(x_train, y_train)
        self._clfs_fit = True

    def predict(self, x_test):
        if self.verbose and not self._clfs_fit:
            print("You may want to fit the classifiers first.")
        predictions = {}
        for clf in self.classifiers:
            predictions[clf.__class__.__name__] = clf.predict(x_test)
        return predictions


__all__ = ['ClassifierHandler', 'CLF']
