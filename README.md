# Classification du genre d'une musique à partir de ses lyrics

## Présentation

Ce projet a pour but d'essayer de classifier le genre d'une musique à partir des lyrics. 


## Résolution

Dans un premier temps, il faut récuperer les lyrics sur le site RapGenius.
Dans un second temps, il faut créer le dataframe avec les lyrcis, l'artiste, le genre et le nettoyer.
Dans un dernier temps, l'application des différents modèles sur les données d'entrainement afin de choisir le meilleur modèle.

Le module functions contient les fichiers suivants : 

    * preprocessing.py pour nettoyer les lyrics.
    * lyricsencoding.py pour vectoriser les lyrics selon l'approche choisie.

Le module codes contient les fichiers suivants :

    * scraping.py pour récuper le code html des musiques sur le site RapGenius.
    * dataloader.py pour générer un dataframe à partir des fichiers html.
    * models.py pour créer les différents modèles de machine learning à utiliser.
    