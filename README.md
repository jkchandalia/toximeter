# toxic-comment-classifier

## Deployment

The toxic comment classifier will be deployed as an API accessible through a POST request. It will take an array of text comments as input and return an array as output where each item is an array with the original text and a score indicating toxicity. 

This API will be deployed as a serverless function. The code for this deployment is in the gcp folder. The gcp endpoint can be tested with:

> curl -X POST    https://us-central1-toxicity-90.cloudfunctions.net/get_toxicity_prediction    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

For faster local iteration, a simple flask-based application can be used for model serving. This is the flask folder. To test locally, first start the server:
> python app.py

Test the API using:
> curl -X POST    0.0.0.0:80/predict    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

This deployed toxicity classifier will then be used in a Chrome Extension to label toxic comments on sites like Reddit. The code for the chrome extension can be found in the chrome_extension folder.  

This version of the deployment is a serverless deployment where the cloud service provided will manage all of the infrastructure and scaling. 

Another possibility for consuming the model is a simple front-end displaying the toxicity of several twitter topics, i.e., politics and the pandemic. 