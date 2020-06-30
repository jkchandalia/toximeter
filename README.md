# toxic-comment-classifier

## Deployment

The toxic comment classifier will be deployed as an API accessible through a POST request. It will take an array of text comments as input and return an array as output where each item is an array with the original text and a score indicating toxicity. 

This API will be deployed as a serverless function. The code for this deployment is in the gcp folder. The gcp endpoint can be tested with:

> curl -X POST    https://us-central1-toxicity-90.cloudfunctions.net/get_toxicity_prediction    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

For faster local iteration, a simple flask-based application can be used for model serving. This is the flask_model folder. To test locally, first start the server:
> python app.py

Test the API using:
> curl -X POST    0.0.0.0:80/predict    -H 'Content-Type: application/json'    -d '["hello","fuck, this is a toxic comment"]'

This deployed toxicity classifier can then be used in a Chrome Extension to label toxic comments on sites like Reddit. The code for the chrome extension can be found in the chrome_extension folder. To install this extension, navigate to chrome://extensions/ in your chrome browser. Click on "Load Unpacked" and load all the files in the chrome_extension folder in this repo. Navigate to a particular conversation on reddit like: 
> https://www.reddit.com/r/politics/comments/hie3tq/trump_got_written_briefing_in_february_on/

Click on "View Entire Discussion" to load more comments for labeling and then click on the chrome extension icon. Click the "Label Toxicity" button and then toxic comments will be labeled with a dark red background to both identify them and make them hard to read. 

This version of the deployment is a serverless deployment where the cloud service will manage all of the infrastructure and scaling. 

Another possibility for consuming the model is a simple front-end displaying the toxicity of several twitter topics, i.e., politics and the pandemic. 