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


def create_embedding_index(glove_embedding_path):
    embeddings_index = {}
    f = open(glove_embedding_path + 'glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
    for line in tqdm(f):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray([float(val) for val in values[1:]])
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, output_path=None):
    # create an embedding matrix for the words we have in the dataset
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    if output_path:
        np.save(output_path, embedding_matrix)
    return embedding_matrix


def load_embeddings(input_path):
    embedding_matrix = np.load(input_path)
    return embedding_matrix


def build_model(word_index, embedding_matrix, max_len, transformer_trainable=False):
    """
    Function for training the model
    """
    # A simple LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=transformer_trainable))

    model.add(LSTM(100, activation="tanh",
        recurrent_activation="sigmoid", dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', AUC(curve='PR')])
    
    return model


def tokenize(xtrain, xvalid, max_len):
    # using keras tokenizer here
    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(xtrain) + list(xvalid))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xvalid_seq = token.texts_to_sequences(xvalid)

    #zero pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

    word_index = token.word_index
    return xtrain_pad, xvalid_pad, word_index

