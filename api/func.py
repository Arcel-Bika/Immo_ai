from sklearn.neighbors import KNeighborsRegressor
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
    X = data[['nb_chambres', 'nb_salon', 'taille_maison', 'commune', 'quartier']]
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
    Entraîne un modèle KNN.
    """
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    print(knn_model)
    return knn_model


def model_score(X_test, y_test, model):
    """
    Retourne le score du modèle.
    """
    return model.score(X_test, y_test)


def output_data(model, X_test):
    """
    Retourne les prédictions du modèle.
    """
    return model.predict(X_test)


def load_or_train_model():
    """
    Charge le modèle existant s'il existe, sinon l'entraîne et sauvegarde.
    """
    model_path = "knn_model.pkl"

    if os.path.exists(model_path):
        # Charger le modèle existant
        print("Chargement du modèle existant...")
        knn_model = joblib.load(model_path)
    else:
        # Entraîner un nouveau modèle
        data = pd.read_csv('assets//source.csv', delimiter=';')
        print("Colonnes disponibles dans le fichier CSV :")
        print(data.columns)

        X, y = select_features(data)
        X = data_encoder(X)
        X_train, X_test, y_train, y_test = test(X, y)
        knn_model = model(X_train, y_train)
        print("Le score est de : ", model_score(X_test, y_test, knn_model))

        # Sauvegarder le modèle
        joblib.dump(knn_model, model_path)
        print("Modèle entraîné et sauvegardé.")

    return knn_model