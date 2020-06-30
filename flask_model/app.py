# Serve model as a flask application

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

model = None
app = Flask(__name__)


def load_model():
    global model
    global vectorizer
    # model variable refers to the global variable
    model = load('models/model.joblib')
    preprocessor = load('models/preprocessing.joblib')

@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        comment_vectorized = preprocessor.transform(data)
        out = list(model.predict_proba(comment_vectorized)[:,1])      
    return json.dumps(out)


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80, ssl_context='adhoc')