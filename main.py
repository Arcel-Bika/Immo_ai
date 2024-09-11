"""
Immo AI - Application Streamlit pour prédire le prix des maisons

Cette application permet à l'utilisateur de saisir des caractéristiques d'une maison, telles que
la taille de la parcelle, le nombre de chambres et de salons, ainsi que la commune et le quartier,
pour obtenir une estimation du prix via un modèle d'intelligence artificielle (IA).

Le modèle d'IA est entraîné à partir des données précédemment fournies, encodées et traitées avec
des algorithmes de machine learning.

Modules
-------
    - streamlit : pour créer l'interface utilisateur
    - pandas : pour gérer les données sous forme de DataFrame
    - func : un module externe qui contient les fonctions `data_encoder` et `output_data`
             ainsi qu'un modèle de régression linéaire (`reg_model`).

Fonctions
---------
    - st.title : Affiche le titre de l'application.
    - st.text : Affiche une brève description de l'application.
    - st.text_input : Permet à l'utilisateur de saisir des informations sur les caractéristiques de la maison.
    - st.error : Affiche un message d'erreur si les données saisies ne sont pas valides.
    - st.write : Affiche le résultat de la prédiction du modèle.

Données saisies par l'utilisateur
---------------------------------
    - Commune : La commune où se trouve la maison.
    - Quartier : Le quartier où se trouve la maison.
    - Taille de la parcelle : Taille du terrain de la maison (doit être un nombre entier).
    - Nombre de salons : Le nombre de salons dans la maison (doit être un nombre entier).
    - Nombre de chambres : Le nombre de chambres dans la maison (doit être un nombre entier).

Le modèle prédira ensuite le prix estimé en fonction de ces caractéristiques.
"""

from api import func
import streamlit as st
import pandas as pd

# Titre
st.title("Immo :blue[AI] 🤖​")
st.text("Recherchez, trouvez selon votre budget")

# Initialiser les variables si elles n'existent pas encore dans st.session_state
if 'visibility' not in st.session_state:
    st.session_state.visibility = "visible"
if 'disabled' not in st.session_state:
    st.session_state.disabled = False
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = "Entrez une valeur"

# Création des colonnes
col1, col2 = st.columns(2)

# Données à rechercher
with col1:
    commune = st.text_input(
        "Entrez la commune",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )
    quartier = st.text_input(
        "Entrez le quartier",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )
    taille_parcelle = st.text_input(
        "Entrez la taille de la parcelle",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

with col2:
    nb_salon = st.text_input(
        "Entrez le nombre de salons",
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )
    nb_chambres = st.text_input(
        "Entrez le nombre de chambres",
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

# Vérification des données
if taille_parcelle.isdigit() and nb_salon.isdigit() and nb_chambres.isdigit():
    taille_parcelle = int(taille_parcelle)
    nb_salon = int(nb_salon)
    nb_chambres = int(nb_chambres)
else:
    st.error("Veuillez entrer des nombres valides pour les salons, les chambres et la taille de la parcelle.")

# Exemple des données entrées par l'utilisateur
user_data = {
    'nb_chambres': [nb_chambres],
    'nb_salon': [nb_salon],
    'taille_parcelle': [taille_parcelle],
    'commune': [commune],
    'quartier': [quartier]
}

user_df = pd.DataFrame(user_data)

# Encodage des données
user_df_encoded = func.data_encoder(user_df)

# Vérification de l'existence du modèle
if hasattr(func, 'reg_model'):
    # Prédiction du prix de la maison
    predicted_price = func.output_data(func.reg_model, user_df_encoded)
    st.write(f"Le prix estimé pour la maison est : {predicted_price[0]} $")
else:
    st.error("Le modèle de régression n'est pas disponible.")
