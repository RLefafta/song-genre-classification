from functions import preprocessing, lyricsencoding
import dataloader
import models
import pandas as pd


dict_artistes = {
    "Orelsan": "rap",
    "Ninho": "rap",
    "SCH": "rap",
    "Georgio": "rap",
    "Dinos": "rap",
    "Indochine": "rock",
    "Noir Désir": "rock",
    "Louise Attaque": "rock",
    "BB Brunes": "rock",
    "Téléphone": "rock",
    "Kyo": "rock",
    "Superbus": "rock",
}


class RapGenius:
    """
    A class used to predict songs' genre based on lyrics

    Attributes
    ----------
    folder_path: str
        path where html files are stored


    Methods
    -------
    _get_dataframe() -> pd.DataFrame
        create the dataframe with 3 columns ('artist', 'title', 'lyrics') by parsing all html_files

    cleaning_df() -> pd.DataFrame
        clean the dataframe

    _get_train_test():
        create the train, test data

    classifier(vectorizer: str):
        using train data, fit and predict for each models and for each parameters

    show_result():
        print out a table to resume cross-validation score for each best parameters models

    final(y_test, X_test):
        print out the best model among models, cross-validation score on train data, score on train data, score on test data and confusion matrix.

    """

    def __init__(self, folder_path):
        """
        Parameters
        ----------
        folder_path: str
            path where html files are stored

        Return
        ------
        self.df : pd.DataFrame
            A dataframe with every songs for an artist.
        """

        self.folder_path = folder_path
        self.loaddata = dataloader.LoadData(folder_path=self.folder_path)
        self.df = self._get_dataframe()
        self.models = models.Models()

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Return
        ------
            pd.DataFrame
        """
        return self.loaddata.dataframe_creator()

    def cleaning_df(self) -> pd.DataFrame:
        """
        Return
        ------
            pd.DataFrame
        """
        self.df = preprocessing.do_all(self.df, dict_artistes)
        return self.df

    def _get_train_test(self, df, vectorizer):
        """
        Parameters
        ----------
            vectorizer: str
        Return
        ------
            Train, test data
        """
        train_test = lyricsencoding.do_encoding(df=self.df, vectorizer=vectorizer)
        self.y_train = train_test[2]
        self.y_test = train_test[3]
        if len(train_test[0]) == 2:
            self.X_train_bow = train_test[0]["bow"]
            self.X_train_tfidf = train_test[0]["tfidf"]
            self.X_test_bow = train_test[1]["bow"]
            self.X_test_tfidf = train_test[1]["tfidf"]
            self.X_train = [self.X_train_bow, self.X_train_tfidf]
            self.X_test = [self.X_test_bow, self.X_test_tfidf]
        else:
            self.X_train = train_test[0]
            self.X_test = train_test[1]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def classifier(self, vectorizer):
        """
        Parameters
        ----------
            vectorizer: str
        """
        self._get_train_test(df=self.df, vectorizer=vectorizer)
        self.models._fit_predict(self.X_train, self.y_train, vectorizer)

    def show_result(self):
        """
        Return
        ------
            Table
        """
        self.models.show_result()

    def final(self):
        self.models.best_model(y_test=self.y_test, X_test=self.X_test)
