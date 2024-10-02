from .ae_handler import AutoEncoderHandler, AE
from .clf_handler import ClassifierHandler, CLF
from .search_handler import GridSearchHandler, RandomSearchHandler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score

type SEARCH_HANDLER = GridSearchHandler or RandomSearchHandler


def evaluate_metrics(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    if len(set(y_test)) == 2 or hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(x_test)[:, 1] if hasattr(classifier, "predict_proba") else None
        auroc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    else:
        auroc = None

    return {
        'Accuracy': accuracy,
        'F1-macro': f1_macro,
        'F1-weighted': f1_weighted,
        'AUROC': auroc
    }


class PipelineManager:
    def __init__(self, autoencoder: AE, classifiers: list[CLF], x_train, y_train, x_tune, y_tune, x_test, y_test,
                 verbose=0, seed=42):
        self.autoencoder_handler = AutoEncoderHandler(autoencoder, x_train, y_train, verbose) if autoencoder else None
        self.classifier_handler = ClassifierHandler(classifiers, seed, verbose)
        self.x_train = x_train
        self.y_train = y_train
        self.x_tune = x_tune
        self.y_tune = y_tune
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose
        self.seed = seed
        self.clf_trained = False

    def train_autoencoder(self, epochs=50, batch_size=2, shuffle=True, validation_split=0.2, callbacks=None):
        if self.autoencoder_handler:
            return self.autoencoder_handler.train(epochs, batch_size, shuffle, validation_split, callbacks)

    def tune_classifiers(self, param_grids, cv=5, metric='f1_macro', tuner: SEARCH_HANDLER = GridSearchHandler, refit: bool = False):
        x_tune = self.autoencoder_handler.encode(self.x_tune) if self.autoencoder_handler else self.x_tune

        tuner = tuner(self.classifier_handler.classifiers, param_grids, cv=cv, metric=metric, verbose=self.verbose)

        if refit:
            clfs, results = tuner.perform_search(x_tune, self.y_tune, refit=refit)
            self.classifier_handler.classifiers = clfs
            self.clf_trained = True
            return results

        return tuner.perform_search(x_tune, self.y_tune, refit=refit)

    def test_classifiers(self, params=None, refit=False):
        if params:
            assert len(self.classifier_handler.classifiers) == len(params), "Params classifier mismatch."

        x_train = self.autoencoder_handler.encode(self.x_tune) if self.autoencoder_handler else self.x_tune
        x_test = self.autoencoder_handler.encode(self.x_test) if self.autoencoder_handler else self.x_test

        if params:
            self.classifier_handler.set_params(params)

        if not self.clf_trained or refit:
            self.classifier_handler.fit(x_train, self.y_tune)

        results = {}
        for clf in self.classifier_handler.classifiers:
            results[clf.__class__.__name__] = evaluate_metrics(clf, x_test, self.y_test)

        return results

    def get_roc_data(self):
        x_test = self.autoencoder_handler.encode(self.x_test) if self.autoencoder_handler else self.x_test

        roc_data = {}
        for clf in self.classifier_handler.classifiers:
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(x_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
                roc_data[clf.__class__.__name__] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds
                }

        return roc_data


__all__ = ['PipelineManager']
