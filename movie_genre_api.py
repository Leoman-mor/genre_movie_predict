# movie_genre_api.py - API para clasificación de géneros de películas
import pandas as pd
import numpy as np
import joblib
import json
import os
import spacy
from flask import Flask
from flask_restx import Api, Resource, fields, reqparse

# Inicializar Flask y API
app = Flask(__name__)
api = Api(app, 
          version='1.0', 
          title='API de Clasificación de Géneros de Películas',
          description='API para predecir géneros de películas basado en sinopsis. La sinopsis debe estar en inglés para mejores resultados.')

# Namespace
ns = api.namespace('movies', description='Clasificación de géneros')

# Rutas de archivos
MODEL_PATH = 'movie_genre_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
GENRE_LIST_PATH = 'genre_list.json'

# Verificar archivos y cargar modelo
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
    model = None
    label_encoder = None
    genre_list = None
    nlp = None
else:
    # Cargar componentes
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Cargar lista de géneros
    with open(GENRE_LIST_PATH, 'r') as f:
        genre_list = json.load(f)
    
    # Cargar spaCy
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000
    nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe not in ["tokenizer", "lemmatizer"]])

# Modelo de respuesta
resource_fields = api.model('PredictionResult', {
    'result': fields.String(description='Género con mayor probabilidad'),
    'top_5_genres': fields.Raw(description='Top 5 géneros más probables')
})

# Parser
parser = reqparse.RequestParser()
parser.add_argument('plot', type=str, required=True, help='Sinopsis de la película')

# Función de preprocesamiento
def preprocess_text(text):
    if not text:
        return ""
    
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_digit]
    return ' '.join(tokens)

# Función de predicción
def predict_genres(plot_text):
    processed_text = preprocess_text(plot_text)
    probabilities = model.predict_proba([processed_text])[0]
    
    # Crear lista de géneros con probabilidades
    genre_probs = []
    for i, genre in enumerate(label_encoder.classes_):
        prob = float(probabilities[i])
        genre_probs.append((genre, prob))
    
    # Ordenar por probabilidad
    genre_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Género con mayor probabilidad
    best_genre = genre_probs[0][0]
    
    # Top 5
    top_5 = [{"genre": genre, "probability": round(prob, 4)} for genre, prob in genre_probs[:5]]
    
    return best_genre, top_5

# Endpoint principal
@ns.route('/')
class MovieGenreApi(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        if model is None:
            api.abort(500, "El modelo no está disponible. Verifica los logs del servidor.")
        
        args = parser.parse_args()
        plot_text = args['plot']
        
        best_genre, top_5 = predict_genres(plot_text)
        
        return {
            "result": best_genre,
            "top_5_genres": top_5
        }, 200

# Ejecutar aplicación
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)