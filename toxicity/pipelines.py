import logging
import pandas as pd
import sys
from toxicity import data, model, constants

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def build_toxicity_model(input_data_path, output_model_path=None):
    """Data pipeline and predictions.
    Parameters
    ----------
    input_data_path: str
        Path to the Kaggle Toxicity dataset
    output_model_path: str
        Path where the output models will be saved
    """

    logging.info('Starting the model building pipeline')

    
    df_train = data.load(input_data_path, filter=True)
    logging.info('Data loaded from ' + input_data_path)

    xtrain, xvalid, ytrain, yvalid = model.make_train_test(df_train)
    logging.info('Data split into train/test sets')
    logging.info('xtrain length: ' + str(len(xtrain)))
    logging.info('xvalid length: ' + str(len(xvalid)))
    
    count_train, count_valid = model.apply_count_vectorizer(xtrain, xvalid, output_model_path)
    logging.info('Vectorization of data complete')
    logging.info('count_train shape: ' + str(count_train.shape))
    logging.info('count_valid shape: ' + str(count_valid.shape))
    if output_model_path:
        logging.info("Vectorizer saved here: " + output_model_path)

    nb_classifier = model.run_naive_bayes_classifier(count_train, ytrain, output_model_path)
    logging.info('Classifier built')
    if output_model_path:
        logging.info("Model saved here: " + output_model_path)

    logging.info('The model building pipeline has finished')

    return