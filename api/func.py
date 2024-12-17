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

    - load_or_train_model():
        Charge le modèle existant ou entraîne un nouveau si nécessaire.
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os


def data_file(file):
    """
    Charge les données à partir d'un fichier CSV.
    """
    return pd.read_csv(file)


def select_features(data):
    """
    Sélectionne les features pertinentes et la cible.
    Ajuste les noms de colonnes si nécessaire.
    """
    X = data[['nb_chambres', 'nb_salon', 'taille_parcelle', 'commune', 'quartier']]
    y = data['prix']
    return X, y


def data_encoder(X):
    """
    Encode les colonnes catégorielles 'commune' et 'quartier'.
    """
    return pd.get_dummies(X, columns=['commune', 'quartier'], drop_first=True)


def test(X, y):
    """
    Divise les données en ensembles d'entraînement et de test.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def model(X_train, y_train):
    """
    Entraîne un modèle de régression linéaire.
    """
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    return reg_model


def output_data(model, X_test):
    """
    Retourne les prédictions du modèle.
    """
    return model.predict(X_test)


def load_or_train_model():
    """
    Charge le modèle existant s'il existe, sinon l'entraîne et sauvegarde.
    """
    model_path = "reg_model.pkl"

    if os.path.exists(model_path):
        # Charger le modèle existant
        print("Chargement du modèle existant...")
        reg_model = joblib.load(model_path)
    else:
        # Entraîner un nouveau modèle
        print("Entraînement du modèle...")
        data = pd.read_csv('assets//source.csv', delimiter=';')
        print("Colonnes disponibles dans le fichier CSV :")
        print(data.columns)

        X, y = select_features(data)
        X = data_encoder(X)
        X_train, X_test, y_train, y_test = test(X, y)
        reg_model = model(X_train, y_train)

        # Sauvegarder le modèle
        joblib.dump(reg_model, model_path)
        print("Modèle entraîné et sauvegardé.")

    return reg_model
