import json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


model = load('models/model.joblib')
preprocessor = load('models/preprocessing.joblib')

def get_toxicity_prediction(request):
    # For more information about CORS and CORS preflight requests, see
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    # for more information.

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows POST requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    
    data = request.get_json(silent=True)  # Get data posted as a json
    comment_vectorized = preprocessor.transform(data)
    pred_proba = model.predict_proba(comment_vectorized)[:,1]
    out = list(zip(data,pred_proba))
    #prediction = model.predict_proba(data_vectorized)[:,1]  # runs globally loaded model on the data
    return (json.dumps(out), 200, headers)
