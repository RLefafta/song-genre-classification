from functions import preprocessing, lyricsencoding


def test_lowercases(text="Le processUs est TERMiné EN avance"):
    lowering = preprocessing.lowercases(text)
    assert lowering == "le processus est terminé en avance"


def test_remove_words(text="le processus est terminé en avance"):
    removing = preprocessing.remove_words(text)
    assert removing == "processus terminé avance"


def test_tokenization(text="processus terminé avance"):
    tokens = preprocessing.tokenization(text)
    assert tokens == ["processus", "terminé", "avance"]


def test_stemmatization(text=["processus", "terminé", "avance"]):
    stem = preprocessing.stemming(text)
    assert stem == ["processus", "termin", "avanc"]
