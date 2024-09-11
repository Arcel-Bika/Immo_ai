"""
Description:
___________
    Ce fichier contient toutes les fonctions principales du programme Immo AI.

Fonctions
---------
    - data_file(file):
        Charge et retourne les données à partir d'un fichier CSV.

    - select_features(data):
        Sélectionne les features pertinentes pour l'entraînement du modèle.

    - data_encoder(X):
        Encode les données catégorielles.

    - test(X, y):
        Divise les données en ensembles d'entraînement et de test.

    - model(X_train, y_train):
        Entraîne un modèle de régression linéaire.

    - output_data(model, X_test):
        Retourne les prédictions du modèle.

    - value_ai(y_test, y_pred):
        Évalue les prédictions avec l'erreur quadratique moyenne.
"""
from statistics import LinearRegression

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def data_file(file):
    """
    Charge les données à partir d'un fichier CSV.

    Parameters
    ----------
    file : str
        Chemin du fichier CSV à charger.

    Returns
    -------
    DataFrame
        Les données sous forme de DataFrame.
    """
    return pd.read_csv(file)


def select_features(data):
    """
    Sélectionne les features pertinentes et la cible.

    Parameters
    ----------
    data : DataFrame
        Données d'entrée.

    Returns
    -------
    tuple
        Tuple contenant X (features) et y (cible).
    """
    X = data[['nb_chambres', 'nb_salon', 'taille_parcelle', 'commune', 'quartier']]
    y = data['prix']
    return X, y


def data_encoder(X):
    """
    Encode les colonnes catégorielles 'commune' et 'quartier'.

    Parameters
    ----------
    X : DataFrame
        Les features à encoder.

    Returns
    -------
    DataFrame
        Les données encodées.
    """
    return pd.get_dummies(X, columns=['commune', 'quartier'], drop_first=True)


def test(X, y):
    """
    Divise les données en ensembles d'entraînement et de test.

    Parameters
    ----------
    X : DataFrame
        Features pour l'entraînement.
    y : Series
        Cible associée.

    Returns
    -------
    tuple
        Tuple contenant X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def model(X_train, y_train):
    """
    Entraîne un modèle de régression linéaire.

    Parameters
    ----------
    X_train : DataFrame
        Features d'entraînement.
    y_train : Series
        Cible d'entraînement.

    Returns
    -------
    LinearRegression
        Modèle entraîné.
    """
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    return reg_model


def output_data(model, X_test):
    """
    Retourne les prédictions du modèle.

    Parameters
    ----------
    model : LinearRegression
        Modèle entraîné.
    X_test : DataFrame
        Données de test.

    Returns
    -------
    ndarray
        Prédictions du modèle.
    """
    return model.predict(X_test)


def value_ai(y_test, y_pred):
    """
    Évalue les prédictions avec l'erreur quadratique moyenne.

    Parameters
    ----------
    y_test : Series
        Valeurs réelles.
    y_pred : ndarray
        Valeurs prédites par le modèle.

    Returns
    -------
    float
        Erreur quadratique moyenne (MSE).
    """
    mse = mean_squared_error(y_test, y_pred)
    print(f'Erreur quadratique moyenne: {mse}')
    return mse


# Utilisation des fonctions

if __name__ == "__main__":
    # Charger les données
    data = data_file('maisons.csv')

    # Sélectionner les features et la cible
    X, y = select_features(data)

    # Encoder les données catégorielles
    X = data_encoder(X)

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = test(X, y)

    # Modèle de régression linéaire
    reg_model = model(X_train, y_train)

    # Prédictions
    y_pred = output_data(reg_model, X_test)

    # Évaluation du modèle
    mse = value_ai(y_test, y_pred)
