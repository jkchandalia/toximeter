# toxic-comment-classifier
## Problem Statement

The goal of this project was to build Toximeter, a general purpose model capable of identifying toxic commentary. The data was provided by Jigsaw (an Alphabet subsidiary) in a kaggle competition, https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge. The dataset consisted of ~225K Wikipedia comments, roughly 10% of which were considered toxic.

## Exploration

I explored more 'classical' NLP techniques like bag-of-words models using simple word counts as well tf-idf (term frequency - inverse document frequency) weights with a Naive Bayes model. These model performed surprisingly well (AUC for PR-curve of 0.8).

I did spend some time exploring more state-of-the-art models like LSTMs with pretrained word embeddings (GloVe) as well as BERT and DistilBERT based models. Of the deep learning models I explored, the GloVe+LSTM performed the best. I achieved an AUC-PR of 0.86. My takeaway is that a sequence layer after a feature extractor/embedding layer provides a lot of improvement. Further model improvement avenues include making the base models such as the GloVe embeddings and the BERT models traininable. To date, I have kept them frozen. Other ideas include trying other tokenizers and exploring different NN architectures.

## Deployment

The toxic comment classifier will be deployed as an API accessible through a POST request. It will take an array of text comments as input and return an array as output where each item is an array with the original text and a score indicating toxicity. 

This API will be deployed as a serverless function. The code for this deployment is in the gcp folder. The google cloud function is deployed with the following command:
> gcloud functions deploy get_toxicity_prediction --runtime python37 --trigger-http --allow-unauthenticated

The gcp endpoint can be tested with:

> curl -X POST    https://us-central1-toxicity-90.cloudfunctions.net/get_toxicity_prediction    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

For faster local iteration, a simple flask-based application can be used for model serving. This is the flask_model folder. To test locally, first start the server:
> python app.py

Test the API using:
> curl -X POST    0.0.0.0:80/predict    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

This deployed toxicity classifier can then be used in a Chrome Extension to label toxic comments on sites like Twitter. The code for the chrome extension can be found in the chrome_extension_twitter folder. To install this extension, navigate to chrome://extensions/ in your chrome browser. Click on "Load Unpacked" and load all the files in the chrome_extension folder in this repo. Navigate to a particular conversation on topic on twitter like: 
> https://twitter.com/search?q=trump&src=typed_query

As one scrolls down the twitter feed, toxic tweets will automatically be labeled with a red background. Responses from the deployed model can be seen using the "Inspect" tool (right click in a browser window to get this option). Model output can be seen in the console. Due to possible rate-limiting or outages for the google cloud function, it is possible to see 429 or 500 status codes. These will generally resolve after a few minutes. If the problem persists, it may require redeployment of the google cloud function.

This version of the deployment is a serverless deployment where the cloud service will manage all of the infrastructure and scaling. An even better way to deploy this for the Chrome extension would be client-side using using a javascript model. 

Another possibility for consuming the model is a simple front-end displaying the toxicity of several twitter topics, i.e., politics and the pandemic. 

## Improvements

The data used to train the models and the data that the model is scoring is qualitatively different, Wikipedia comments vs. Twitter comments. The available data provided a starting point but the full system would take a sample of the scored twitter comments the model and send them out to be labeled as toxic/nontoxic. The model would then need to be retrained periodically. The full monitoring solution would note changes in the toxicity scoring distributions over time and offer spot-checking capabilities for a small sample of comments.

Notably, commercially available versions of this type of model have not necessarily performed well: 

1. Tune, a Chrome extension: https://chrome.google.com/webstore/detail/tune-experimental/gdfknffdmmjakmlikbpdngpcpbbfhbnp?hl=en
2. Perspective API, an API for toxicity labeling from Jigsaw