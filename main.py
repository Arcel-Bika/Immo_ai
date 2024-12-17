"""
Immo AI - Application Streamlit pour prédire le prix des maisons

Cette application permet à l'utilisateur de saisir des caractéristiques d'une maison, telles que
la taille de la parcelle, le nombre de chambres et de salons, ainsi que la commune et le quartier,
pour obtenir une estimation du prix via un modèle d'intelligence artificielle (IA).

Modules
-------
    - streamlit : pour créer l'interface utilisateur
    - pandas : pour gérer les données sous forme de DataFrame
    - func : un module externe qui contient les fonctions `data_encoder` et `output_data`
             ainsi qu'un modèle de régression linéaire (`reg_model`).
"""

from api.func import load_or_train_model, data_encoder, output_data
import streamlit as st
import pandas as pd


# Titre
st.title("Immo :blue[AI] 🤖​")
st.text("Recherchez, trouvez selon votre budget")

# Vérifier et charger le modèle
reg_model = load_or_train_model()


# Initialiser les variables si elles n'existent pas encore dans st.session_state
if 'visibility' not in st.session_state:
    st.session_state.visibility = "visible"
if 'disabled' not in st.session_state:
    st.session_state.disabled = False
if 'placeholder' not in st.session_state:
    st.session_state.placeholder = "Entrez une valeur"

# Création des colonnes
col1, col2 = st.columns(2)

# Données saisies par l'utilisateur
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
    st.error("Veuillez entrer des valeurs numériques valides.")

# Données utilisateur pour la prédiction
user_data = {
    'nb_chambres': [nb_chambres],
    'nb_salon': [nb_salon],
    'taille_parcelle': [taille_parcelle],
    'commune': [commune],
    'quartier': [quartier]
}

user_df = pd.DataFrame(user_data)

# Encodage
user_df_encoded = data_encoder(user_df)

# Prédiction avec le modèle
if reg_model:
    predicted_price = output_data(reg_model, user_df_encoded)
    st.write(f"Le prix estimé pour la maison est : {predicted_price[0]} $")
else:
    st.error("Le modèle n'est pas prêt.")
