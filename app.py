# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Activer CORS pour les requêtes depuis React

# Charger le modèle, l'encodeur et le scaler
try:
    model = joblib.load('pretrained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    print(f"Erreur : {e}. Assurez-vous que pretrained_model.pkl, label_encoder.pkl et scaler.pkl existent.")
    exit(1)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Récupérer les données de l'utilisateur
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Aucune donnée fournie'}), 400

        # Valider les données d'entrée
        user_data = {
            'likes': float(data.get('likes', 0)),
            'comments': float(data.get('comments', 0)),
            'shares': float(data.get('shares', 0)),
            'watch_time': float(data.get('watch_time', 0))
        }

        # Vérifier que les valeurs sont positives
        for key, value in user_data.items():
            if value < 0:
                return jsonify({'status': 'error', 'message': f'La valeur de {key} doit être positive'}), 400

        # Convertir en DataFrame
        input_data = pd.DataFrame([user_data])

        # Normaliser les données avec le scaler
        input_data_scaled = scaler.transform(input_data)

        # Prédire la catégorie
        prediction = model.predict(input_data_scaled)
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            'status': 'success',
            'recommended_category': predicted_category
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': f'Erreur de format des données : {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Erreur serveur : {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Utiliser le port fourni par Render ou 5001 en local
    app.run(host='0.0.0.0', port=port)
