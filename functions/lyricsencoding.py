import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def dummy_fun(doc):
    return doc


def tokens_to_list(df: pd.DataFrame, col: str) -> list:
    """
    Create a list with tokenized lyrics

    Parameters
    ----------
        df: pd.DatFrame

        col: str

    Return
    ------
        list
    """
    return df[col].tolist()


def train_test(df: pd.DataFrame, value: np.array, target: str) -> list:
    """
    Split data into train and test

    Parameters
    ----------
        value: numpy array

        target: str
    Return
    ------
        X_train, X_test, y_train, y_test: list
    """
    X_train, X_test, y_train, y_test = train_test_split(
        value, df[target], test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def bow_encoding(df: pd.DataFrame) -> np.array:
    """
    Create bag of words

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        bow: numpy array
    """
    bow_vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
    )
    bow = bow_vectorizer.fit_transform(
        tokens_to_list(col="lyrics_clean", df=df)
    ).toarray()

    return bow


def tfidf_encoding(df: pd.DataFrame) -> np.array:
    """
    Create term frequency-inverse document frequency

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
       tfidf: numpy array
    """
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
    )
    tfidf = tfidf_vectorizer.fit_transform(
        tokens_to_list(col="lyrics_clean", df=df)
    ).toarray()

    return tfidf


def strategy1(df: pd.DataFrame) -> list:
    """
    Define the strategy according the choice  of encoding

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        list
    """
    bow = bow_encoding(df)
    return train_test(df=df, value=bow, target="genre")


def strategy2(df: pd.DataFrame) -> list:
    """
    Define the strategy according the choice  of encoding

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        list
    """
    tfidf = tfidf_encoding(df)
    return train_test(df=df, value=tfidf, target="genre")


def strategy3(df: pd.DataFrame) -> list:
    """
    Define the strategy according the choice  of encoding

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        list
    """
    bow = bow_encoding(df)
    tfidf = tfidf_encoding(df)
    X = {"bow": bow, "tfidf": tfidf}
    train = dict()
    test = dict()
    for vector, value in X.items():
        train_test_values = train_test(df=df, value=value, target="genre")
        train[vector] = train_test_values[0]
        test[vector] = train_test_values[1]
    return train, test, train_test_values[2], train_test_values[3]


def do_encoding(df: pd.DataFrame, vectorizer: str):
    """
    Split data into train and test according to the strategy

    Parameters
    ----------
        df: pd.DataFrame

        vectorizer: str
    """
    if vectorizer == "bow":
        return strategy1(df)
    elif vectorizer == "tfidf":
        return strategy2(df)
    elif vectorizer == "both":
        return strategy3(df)
