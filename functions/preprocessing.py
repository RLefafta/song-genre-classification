from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language="french")
import pandas as pd


stopwords_perso = [
    "tout",
    "tous",
    "va",
    "ça",
    "non",
    "oui",
    "cet",
    "cette",
    "plus",
    "sous",
]


my_stopwords = stopwords.words("french") + stopwords_perso

stemmer = SnowballStemmer(language="french")


def drop_na_lyrics_artist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop NA values based on the subset column : lyrcis, artist

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        df: pd.DataFrame
    """
    df = df.dropna(subset=["lyrics", "artist"])
    return df


def drop_na_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop NA values based on the column : genre

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        df: pd.DataFrame
    """
    df = df.dropna(subset=["genre"])
    return df


def remove_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate songs for an artist

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        df: pd.DataFrame
    """
    df = df.drop_duplicates(subset=["title", "artist"])
    return df


def index_reset(df: pd.DataFrame) -> pd.DataFrame:
    """
    index_reset()

    Parameters
    ----------
        df: pd.DataFrame

    Return
    ------
        df: pd.DataFrame
    """
    df = df.reset_index(drop=True)
    return df


def cleaning_html(
    df: pd.DataFrame, column: list = ["artist", "title", "lyrics"]
) -> pd.DataFrame:
    """
    Cleaning output from hmtl for listed column : remove duplicates whitespaces, new line, text between brackets

    Parameters
    ----------
        df: pd.DataFrame
        column: list

    Return
    ------
        df: pd.DataFrame
    """
    for col in column:
        if col == "lyrics":
            df[col] = (
                df[col]
                .str.replace("\n", "", regex=True)
                .str.replace("[\(\[].*?[\)\]]", " ", regex=True)
                .str.replace("[^\w\s]", " ", regex=True)
                .str.replace("\s+", " ", regex=True)
            )
        else:
            df[col] = df[col].str.replace("\n", "", regex=True).str.strip()
    return df


def lowercases(text: str) -> str:
    """
    Lowering the text

    Parameters
    ----------
        text: str

    Return
    ------
        text: str
    """
    text = text.lower()
    return text


def remove_words(text: str) -> str:
    """
    Remove words from the stopwords list and words with a lenght below 3.

    Parameters
    ----------
        text: str

    Return
    ------
        text: str
    """
    text = " ".join([word for word in text.split() if word not in (my_stopwords)])
    text = " ".join([word for word in text.split() if len(word) > 2])
    return text


def tokenization(text: str) -> str:
    """
    Tokenization of the text

    Parameters
    ----------
        text: str

    Return
    ------
        text: str
    """
    text = word_tokenize(text)
    return text


def stemming(text: str) -> str:
    """
    Stemmatization of the text

    Parameters
    ----------
        text: str

    Return
    ------
        text: str
    """
    text = [stemmer.stem(word) for word in text]
    return text


def preprocessing_lyrics(df: pd.DataFrame, col: str = "lyrics") -> pd.DataFrame:
    """
    Parameters
    ----------
        col: str

    Return
    ------
        df: pd.DataFrame
    """

    df[col + "_clean"] = (
        df[col]
        .apply(lowercases)
        .apply(remove_words)
        .apply(tokenization)
        .apply(stemming)
    )
    return df


def genre_attribution(df: pd.DataFrame, dict_artistes: dict) -> pd.DataFrame:
    """
    Give to every artist a genre, based on a predifined dictionnary

    Parameters
    ----------
        df: pd.DataFrame
        dict_artistes : dict

    Return
    ------
        df: pd.DataFrame
    """
    for idx in df.index:
        search = df.loc[idx, "artist"]
        for artist, genre in dict_artistes.items():
            if artist == search:
                df.loc[idx, "genre"] = genre
                break
            else:
                continue
    return df


def do_all(df: pd.DataFrame, dict_artistes: dict) -> pd.DataFrame:
    """
    Applying functions every functions to the dataframe

    Parameters
    ----------
        df: pd.DatFrame
    Return
    ------
        df: pd.DataFrame
            Cleaned dataframe
    """

    df = remove_duplicate(df)
    df = cleaning_html(df)
    df = drop_na_lyrics_artist(df)
    df = preprocessing_lyrics(df)
    df = genre_attribution(df, dict_artistes)
    df = drop_na_genre(df)
    df = index_reset(df)
    return df
