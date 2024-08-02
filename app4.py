import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Charger le modèle
model = tf.keras.models.load_model('optimized_classification_model.h5')

# Fonction pour prédire le chiffre
def predict_digit(image):
    image = image.resize((28, 28)).convert('L')  # Convertir en niveaux de gris et redimensionner
    image = np.array(image).astype('float32')

    # Normalisation cohérente avec l'entraînement
    image = image / 255.0

    # Inverser les couleurs si nécessaire
    # image = 1 - image  # Optionnel : inverser les couleurs

    image = image.reshape(1, 28, 28, 1)  # Ajouter la dimension batch

    # Prédiction
    prediction = model.predict(image)
    return np.argmax(prediction), max(prediction[0])

# Interface utilisateur avec Streamlit
st.title('Reconnaissance de Chiffres')

# Diviser l'interface en deux colonnes
col1, col2 = st.columns([1, 1])

with col1:
    st.write('Dessinez un chiffre ci-dessous:')
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=150,
        width=150,
        drawing_mode='freedraw',
        key='canvas'
    )

    if st.button('Prédire le dessin'):
        if canvas_result.image_data is not None:
            image_data = canvas_result.image_data
            image_data = np.array(image_data)[:, :, 0]  # Extraire le canal de niveau de gris

            # Inverser les couleurs pour correspondre aux données d'entraînement (si nécessaire)
            image_data = 255 - image_data  # Assurez-vous que cette étape est correcte

            image = Image.fromarray(image_data.astype('uint8'))
            predicted_digit, confidence = predict_digit(image)
            st.write(f'Prédiction pour le dessin: {predicted_digit}, Confiance: {confidence:.2f}')
        else:
            st.write("Veuillez dessiner un chiffre avant de prédire.")

with col2:
    st.write('Ou choisissez une image aléatoire du dataset MNIST:')
    if st.button('Choisir une image aléatoire'):
        (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        random_idx = np.random.randint(0, X_train.shape[0])
        image = X_train[random_idx]

        st.image(image, width=150, caption="Image MNIST aléatoire")

        image = Image.fromarray(image)
        predicted_digit, confidence = predict_digit(image)
        st.write(f'Prédiction pour l\'image aléatoire: {predicted_digit}, Confiance: {confidence:.2f}')

# Validation de la prédiction pour le dessin
if canvas_result.image_data is not None:
    st.write('Le modèle a-t-il correctement identifié le chiffre dessiné?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Oui'):
            st.success('Super ! Le modèle a fait une bonne prédiction pour le dessin.')
    with col2:
        if st.button('Non'):
            st.error('Désolé, le modèle a fait une erreur pour le dessin.')
