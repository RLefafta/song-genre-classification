from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
import numpy as np
from rich.table import Table
from rich import print


class Models:
    """
    A class used to define models with their parameters and showing the output

    ...

    Methods
    -------
    initialize_models_and_params()
        Initialize models with a GridSearch for parameters

    get_classifier_and_params()
        Recovers classifier and parameters associated

    fit_predict(X_train, y_train, vectorizer: str)
        Run ML models on train data according to the vectorizer

    table_resume()
        Create a table to resume different models with their best parameters

    show_result()
        Show the table created

    best_model(y_test, X_test)
        Find the best model according to the vectorizer

    output(vecto: str)
        Print out the best model on test data with some metrics

    """

    def __init__(self):
        self.classifier_and_params = list()
        self.initialize_models_and_params()

    def initialize_models_and_params(self) -> list:
        """
        Return
        ------
            List
                Each classifiers associated to the parameters to test
        """
        classifier = LogisticRegression()
        params = {
            "penalty": ["l2"],
            "C": [0.1, 1.0, 10, 100],
            "max_iter": [10000, 20000],
        }
        self.classifier_and_params.append((classifier, params))

        classifier = KNeighborsClassifier()
        params = {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        }

        self.classifier_and_params.append((classifier, params))

        classifier = GaussianNB()
        params = {
            "var_smoothing": [1e-11, 1e-10, 1e-9],
        }
        self.classifier_and_params.append((classifier, params))

        classifier = DecisionTreeClassifier()
        params = {
            "max_features": ["None", "auto", "sqrt", "log2"],
            "random_state": [42],
            "min_samples_split": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "max_features": ["auto", "sqrt", "log2"],
        }
        self.classifier_and_params.append((classifier, params))

        classifier = RandomForestClassifier()
        params = {
            "n_estimators": [10, 50, 100],
            "random_state": [42],
            "max_features": ["log2", "sqrt", "auto"],
            "criterion": ["entropy", "gini"],
            "min_samples_split": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        }
        self.classifier_and_params.append((classifier, params))

    def get_classifier_and_params(self) -> list:
        """
        Return
        ------
            List
        """

        return self.classifier_and_params

    def _fit_predict(self, X_train, y_train, vectorizer):
        """
        Parameters
        ----------
            X_train
                Value train data
            y_train
                Target train data
            vectorizer
                The encoding choice
        """
        classifier_and_params = self.get_classifier_and_params()
        self.X_train = X_train
        self.y_train = y_train
        self.vectorizer = vectorizer

        if vectorizer == "both":
            self.results_tfidf = {}
            self.results_bow = {}
            self.X_train_bow = self.X_train[0]
            self.X_train_tfidf = self.X_train[1]
            X = {"bow": self.X_train_bow, "tfidf": self.X_train_tfidf}
            for classifier, params in classifier_and_params:
                for name, value in X.items():
                    self.current_classifier_name = (
                        classifier.__class__.__name__ + " with " + name
                    )
                    grid_search = GridSearchCV(classifier, params, cv=5, n_jobs=-1)
                    grid_search.fit(value, self.y_train)
                    if name == "bow":
                        self.results_bow[grid_search] = grid_search.best_score_
                        print(self.current_classifier_name, "Done")
                    elif name == "tfidf":
                        self.results_tfidf[grid_search] = grid_search.best_score_
                        print(self.current_classifier_name, "Done")

        elif vectorizer == "bow" or vectorizer == "tfidf":
            self.results = {}
            for classifier, params in classifier_and_params:
                self.current_classifier_name = classifier.__class__.__name__
                grid_search = GridSearchCV(classifier, params, cv=5, n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)
                self.results[grid_search] = grid_search.best_score_
                print(self.current_classifier_name, "Done")

    def _table_resume(self, result):
        """
        Parameters
        ----------
            result : dict
                Results stored in a dictionnary
        """
        resume = Table()
        self.resume = resume
        resume.add_column("Score - Train data")
        resume.add_column("Model")
        resume.add_column("Best parameters")
        for score, modele in sorted(
            [(score, modele) for modele, score in result.items()], key=lambda x: x[0]
        ):
            mod_str = str(modele.best_estimator_)
            resume.add_row(
                str(score), mod_str[: mod_str.find("(")], str(modele.best_params_)
            )

    def show_result(self) -> Table:
        """
        Return
        ------
            A table with score and best parameters for each model
        """
        if self.vectorizer == "bow" or self.vectorizer == "tfidf":
            self._table_resume(result=self.results)
            print(self.resume)
        elif self.vectorizer == "both":
            self._table_resume(result=self.results_bow)
            print(self.resume)
            self._table_resume(result=self.results_tfidf)
            print(self.resume)

    def best_model(self, y_test, X_test):
        """
        Parameters
        ----------
            y_test
                target test data

            X_test
                value test data
        Return
        ------
            Return the best model score on test data
        """
        self.X_test_bow = X_test[0]
        self.X_test_tfidf = X_test[1]
        self.y_test = y_test
        if self.vectorizer == "bow" or self.vectorizer == "tfidf":
            self.X_test = X_test
            self._output(self.vectorizer)
        elif self.vectorizer == "both":
            best_score_bow = max(self.results_bow.values())
            best_score_tfidf = max(self.results_tfidf.values())
            if best_score_bow > best_score_tfidf:
                self.vectorizer = "bow"
                self.results = self.results_bow
                self.X_train = self.X_train_bow
                self.X_test = self.X_test_bow
                self._output(vecto="bow")
            else:
                self.vectorizer = "tfidf"
                self.results = self.results_tfidf
                self.X_train = self.X_train_tfidf
                self.X_test = self.X_test_tfidf
                self._output(vecto="tfidf")

    def _output(self, vecto: str):
        """
        Parameters
        ----------
           vecto : str
                Enconding choice, either 'both', 'tfidf' or 'bow
        Return
                The encoding choice, the best model, best model's parameters, cross-validation score, score on train and test data, confusion matrix.
        ------
            Return the best model score on test data
        """
        best_score = max(self.results.values())
        best_model, *_ = [
            model for model, score in self.results.items() if score == best_score
        ]
        print("Best encoding:", vecto)
        print("Best model on train data: ", best_model.best_estimator_)
        print("Parameters: ", best_model.best_params_)
        print("Crossvalidation score: ", best_model.best_score_)
        best_model.fit(self.X_train, self.y_train)
        print("Score on train data: ", best_model.score(self.X_train, self.y_train))
        print("Score on unseen test data: ", best_model.score(self.X_test, self.y_test))
        print(
            "Confusion matrix on train data: \n",
            confusion_matrix(self.y_train, best_model.predict(self.X_train)),
        )
        print(
            "Confusion matrix on test data: \n",
            confusion_matrix(self.y_test, best_model.predict(self.X_test)),
        )
