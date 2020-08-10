import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model



import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, TensorBoard, History
from tensorflow.keras.metrics import Accuracy, AUC



def save_embeddings(output_path):
    pass

def load_embeddings(input_path):
    pass

def build_BERT_model_classification(transformer, max_len=512, transformer_trainable=False):
    """
    Function for training the BERT model
    """
    transformer.trainable = transformer_trainable
    input_word_ids = Input(shape=(max_len,), dtype='int32', name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    pre_classified = Dense(768,
            activation="relu",
            name="pre_classifier")(cls_token)
    logits = Dropout(.2)(pre_classified)
    logits = Dense(2)(logits)
    out = Dense(1, activation='sigmoid')(logits)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC(curve='PR')])
    
    return model

def build_BERT_model_lstm(transformer, max_len=512, transformer_trainable=False):
    """
    Function for training the BERT model
    """
    transformer.trainable = transformer_trainable
    input_word_ids = Input(shape=(max_len,), dtype='int32', name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    lstm_out = LSTM(100, activation="tanh", recurrent_activation="sigmoid")(sequence_output)
    out = Dense(1, activation='sigmoid')(lstm_out)
    
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', AUC(curve='PR')])
    
    return model

def fast_encode(texts, tokenizer, chunk_size=256, max_len=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    #Only a small fraction of input is > max_len, not biased across toxic/nontoxic.
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(length=max_len)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def smart_sample(x,y,multiplier=1):
    xpos = x[y==1]
    xneg = np.random.choice(xtrain, multiplier*sum(y))
    xnew = np.concatenate((xpos, xneg))
    length_new = len(xnew)
    ynew = np.concatenate((np.full(len(xpos), 1), np.full(len(xneg), 0)))
    p = np.random.permutation(length_new)
    return xnew[p], ynew[p]
