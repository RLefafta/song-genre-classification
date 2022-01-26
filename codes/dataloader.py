import os
import pandas as pd
from bs4 import BeautifulSoup
import re








class LoadData:
    """
    A class used to create a pandas DataFrame based on html files

    ...

    Attributes
    ----------
    folder_path: str
        path where html files are stored

    Methods
    -------
    open_file()
        Create a list with every html files

    dataframe_creator():
        Create the dataframe by parsing html files
    """

    def __init__(self, folder_path: str):
        """
        Parameters
        ----------
        folder_path: str
            Path where html files are stored
        """
        self.folder_path = folder_path

    def open_file(self) -> list:
        html_files = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                html_files.append(os.path.join(root, file))
        return html_files

    def dataframe_creator(self):
        """
        Returns
        -------
        pd.DataFrame
            a pandas dataframe with lyrics for each songs from each artist
        """
        self.html_files = self.open_file()
        df = pd.DataFrame(columns=["artist", "title", "lyrics"])

        for idx, file in enumerate(self.html_files):
            aggregate_song = []
            html = open(file, "r", encoding="utf8")
            content = html.read()
            soup = BeautifulSoup(content, "html.parser")
            try:
                lyrics = soup.find_all(class_="Lyrics__Container-sc-1ynbvzw-6")
                for lyric in lyrics:
                    aggregate_song.append(lyric.get_text())
                    df.loc[idx + 1, "lyrics"] = " ".join(aggregate_song)
                df.loc[idx + 1, "title"] = soup.find(
                    class_="SongHeaderVariantdesktop__HiddenMask-sc-12tszai-10"
                ).get_text()

                df.loc[idx + 1, "artist"] = soup.find(class_="hwdSYP").get_text()
                html.close()
            except:
                html.close()
                continue
        return df





