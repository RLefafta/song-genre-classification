import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium import webdriver
from requests import get
from random import randint
from time import sleep
import re
import os


class Scraping:
    """
    A class used to scrap every songs from one or multiple artists.

    ...
    Attributes
    ----------
    artistes : listt
        List of artist to scrap songs


    Methods
    -------
    acces_site()
        Get access to the website RapGenius

    clean_title(title: str)
        Clean title's song in order to save it more easily

    verification_artiste()
        Verify if the artist on RapGenius is the good one

    recupere_code_html()
        Save every songs' html in a folder

    do_all()
        Use above functions to do the scraping

    """

    def __init__(self, artistes: list):
        """
        Parameters
        ----------
        artistes: list
            List of artist to scrap songs

        Return
        ------
            Folder with every songs from each artist
        """
        self.artistes = artistes
        self.do_all(artistes=self.artistes)

    def _acces_site(self, site="rapgenius"):
        self.driver.get("https://www.google.fr/")
        sleep(1)
        self.driver.find_element(by=By.ID, value="L2AGLb").click()
        sleep(1)
        barre = self.driver.find_element(by=By.CLASS_NAME, value="gLFyf")
        barre.send_keys(site)
        sleep(1)
        self.driver.find_element(by=By.CLASS_NAME, value="gNO89b").click()
        sleep(1)
        self.driver.find_element(by=By.CLASS_NAME, value="LC20lb").click()
        sleep(1)
        self.driver.find_element(by=By.ID, value="onetrust-accept-btn-handler").click()
        sleep(1)

    @staticmethod
    def _clean_title(title: str) -> str:
        """
        Parameters
        ----------
        title: str
            Title to clean

        Return
        ------
            Cleaned title
        """
        return re.sub('[\\/*?:"<>|%,.;=]', " ", title)

    def _verification_artiste(self, artiste: str):
        """
        Parameters
        ----------
        artiste: str
            Artist to verify

        Return
        ------
            Click on the good artist
        """
        art = self.driver.find_element(
            by=By.XPATH,
            value="/html/body/routable-page/ng-outlet/search-results-page/div/div[2]/div[2]/div[4]/search-result-section/div/div[2]/search-result-items/div/search-result-item/div/mini-artist-card/a/div[2]/div",
        )
        if art.text == artiste:
            self.driver.find_element(
                by=By.XPATH,
                value="/html/body/routable-page/ng-outlet/search-results-page/div/div[2]/div[2]/div[4]/search-result-section/div/div[2]/search-result-items/div/search-result-item/div/mini-artist-card/a/div[2]/div",
            ).click()
        else:
            self.driver.find_element(
                by=By.XPATH,
                value="/html/body/routable-page/ng-outlet/search-results-page/div/div[2]/div[1]/div[1]/search-result-section/div/div[2]/search-result-items/div/search-result-item/div/mini-artist-card/a/div[1]",
            ).click()

    def _recupere_code_html(self, artiste: str):
        """
        Parameters
        ----------
        artiste: str
            Artist

        Return
        ------
            Save html songs from every album for an artist
        """
        songs = self.driver.find_elements(
            by=By.CLASS_NAME, value="chart_row-content-title"
        )
        url_album = self.driver.current_url
        sleep(1 + randint(1, 2))
        for song in range(len(songs)):
            sleep(1 + randint(1, 2))
            songs[song].click()
            sleep(1 + randint(1, 2))
            url = self.driver.current_url
            if url_album == url:
                continue
            else:
                page = get(url)
                soup = BeautifulSoup(page.text, "html.parser")
                titre = soup.title.get_text()
                titre = Scraping._clean_title(title=titre)
                path = r"C:/Users/remil/Desktop/M2/Python/Projet/htmlv2/%s" % (artiste)
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(
                    r"C:/Users/remil/Desktop/M2/Python/Projet/htmlv2/%s/%s.html"
                    % (artiste, titre),
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(str(soup.prettify()))
                self.driver.back()
                sleep(1 + randint(1, 2))
                songs = self.driver.find_elements(
                    by=By.CLASS_NAME, value="chart_row-content-title"
                )
                sleep(1 + randint(1, 2))

    def do_all(self, artistes: list):
        """
        Parameters
        ----------
        artiste: list
            Artist to scrap

        Return
        ------
            Every html files songs from each artists in a folder
        """
        self.driver = webdriver.Chrome("C:/Users/remil/chromedriver/chromedriver")
        self._acces_site()
        for artiste in artistes:
            recherche = self.driver.find_element(by=By.CLASS_NAME, value="quick_search")
            recherche.send_keys(artiste + Keys.ENTER)
            sleep(5 + randint(1, 2))
            self._verification_artiste(artiste=artiste)
            sleep(1 + randint(1, 2))
            self.driver.find_element(
                by=By.XPATH,
                value="/html/body/routable-page/ng-outlet/routable-profile-page/ng-outlet/routed-page/profile-page/div[3]/div[2]/artist-songs-and-albums/album-grid/div[2]",
            ).click()
            sleep(5 + randint(1, 2))
            albums = self.driver.find_elements(by=By.CLASS_NAME, value="mini_card-info")
            for album in range(len(albums)):
                try:
                    albums[album].click()
                    sleep(5 + randint(1, 2))
                    self._recupere_code_html(artiste=artiste)
                    sleep(1 + randint(1, 2))
                    self.driver.back()
                    sleep(2)
                    if albums[album] != albums[-1]:
                        self.driver.find_element(
                            by=By.XPATH,
                            value="/html/body/routable-page/ng-outlet/routable-profile-page/ng-outlet/routed-page/profile-page/div[3]/div[2]/artist-songs-and-albums/album-grid/div[2]",
                        ).click()
                        sleep(5 + randint(1, 2))
                        albums = self.driver.find_elements(
                            by=By.CLASS_NAME, value="mini_card-info"
                        )
                    else:
                        break
                except:
                    continue
        self.driver.quit()
