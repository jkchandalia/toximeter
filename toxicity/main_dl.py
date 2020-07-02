
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
from keras.callbacks import EarlyStopping, History, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam


max_len = int(round(train['comment_text'].apply(lambda x:len(str(x).split())).max(), -2)+100)
print("Max length of comment text is: {}".format(max_len))


# ### First do Tokenization of input corpus

# using keras tokenizer here
token = text.Tokenizer(num_words=None)
token_toxic = text.Tokenizer(num_words=None)
token_nontoxic = text.Tokenizer(num_words=None)

token.fit_on_texts(list(xtrain) + list(xvalid))
token_toxic.fit_on_texts(train.comment_text.values[train.toxic==1])
token_nontoxic.fit_on_texts(train.comment_text.values[train.toxic==0])

xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index


word_toxic = token_toxic.word_index
word_nontoxic = token_nontoxic.word_index


print(len(word_toxic))
print(len(word_nontoxic))


# Example for fitting tokenizer line-by-line if corpus is too big to fit into memory
# 
# with open('/Users/liling.tan/test.txt') as fin: for line in fin:
# t.fit_on_texts(line.split()) # Fitting the tokenizer line-by-line.
# 
# M = []
# 
# with open('/Users/liling.tan/test.txt') as fin: for line in fin:
# 
#     # Converting the lines into matrix, line-by-line.
#     m = t.texts_to_matrix([line], mode='count')[0]
#     M.append(m)

# ## Use pretrained word embeddings

# ## Convert our one-hot word index into semantic rich GloVe vectors


# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open(pre_path + 'glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))



words_not_in_corpus = ddict(int)
words_in_corpus = ddict(int)
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_nontoxic.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        words_in_corpus[word]+=1
    else:
        words_not_in_corpus[word]+=1



print(len(words_not_in_corpus))
print(len(words_in_corpus))
max(words_not_in_corpus.values())
max(words_in_corpus.values())

#For the full dataset, more than half the 'words' are not found in the glove embeddings
#For the 10K sample dataset, only ~25% of the words are not found in the glove embeddings



print(len(words_not_in_corpus))
print(len(words_in_corpus))
max(words_not_in_corpus.values())
max(words_in_corpus.values())

#For the full dataset, more than half the 'words' are not found in the glove embeddings
#For the 10K sample dataset, only ~25% of the words are not found in the glove embeddings


#Save embeddings so they can be easily loaded
np.save('/kaggle/working/glove_embedding_for_full_data', embedding_matrix)



#Load embeddings
embedding_matrix = np.load('/kaggle/working/glove_embedding_for_10K_sample.npy')


embedding_matrix.shape


# ## Simple RNN Model


opt = Adam(learning_rate=0.0001)



model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                 300,
                 input_length=max_len))
model1.add(SimpleRNN(100))
model1.add(Dense(1, activation='relu'))
model1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
model1.summary()

from keras.callbacks import ModelCheckpoint,TensorBoard, EarlyStopping
EPOCHS = 10
checkpoint_filepath = '/kaggle/working/'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)


my_callbacks = [
    model_checkpoint_callback,
    TensorBoard(log_dir='/kaggle/working/logs'),
    EarlyStopping(monitor='val_loss', patience=3)
]
model_checkpoint_callback



model1.fit(xtrain_pad, 
           ytrain, 
           epochs=50, 
           batch_size=100, 
           callbacks=my_callbacks,
           validation_split=0.2,)


scores = model1.predict(xvalid_pad)[:, 0]
preds = scores>.5
run_metrics(preds, scores, yvalid)


# ## Simple LSTM Model

get_ipython().run_cell_magic('time', '', "# A simple LSTM with glove embeddings and one dense layer\nmodel = Sequential()\nmodel.add(Embedding(len(word_index) + 1,\n                 300,\n                 weights=[embedding_matrix],\n                 input_length=max_len,\n                 trainable=False))\n\nmodel.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))\nmodel.add(Dense(1, activation='sigmoid'))\nmodel.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])\n    \nmodel.summary()")


model.fit(xtrain_pad, 
          ytrain, 
          epochs=50, 
          batch_size=100,
          callbacks=my_callbacks,
          validation_split=0.2,)



scores = model.predict(xvalid_pad)
preds = scores>.5
run_metrics(preds, scores, yvalid)


# # Summary

# So far, with very little preprocessing, we have achieved high accuracy. This is a little bit misleading however because the training set is highly imbalanced (roughly 10% positive/toxic class). 
# 
# Slightly older techniques, bag-of-words and tf-idf have done better than a simple deep learning models out-of-the-box. This can been seen by the higher AUCs and accuracy of these models in contrast to the simple RNN model. In addition, training these models was extremely fast, even on a local machine. In contrast, the deep learning models required more than 10 minutes to train even five epochs. In addition, trainingg the simple RNN required playing around with the learning rate to get network to learn. The first few attempts produced labels of all zeros. 
# 
# The simple LSTM model starts to improve dramatically over the simple RNN model even with only 5 epochs, showing that using the semantic rich word embeddings and including memory already improve simple deep learning results. Though the overall accuracy has decreased in the LSTM model vs the Naive Bayes models, the AUC and precision-recall and ROC curves are much better than the simple models. As we approach more state-of-the-art (SOTA) models and move beyond simple proof-of-concept model training, i.e., try different network parameters, experiment with data preprocessing, do hyperparameter optimization, train until the results start to degrade, add regularization, etc., the results will likely improve even more dramatically.
# 

# ## Try a GRU Model


get_ipython().run_cell_magic('time', '', "# GRU with glove embeddings and two dense layers\n model = Sequential()\n model.add(Embedding(len(word_index) + 1,\n                 300,\n                 weights=[embedding_matrix],\n                 input_length=max_len,\n                 trainable=False))\n model.add(SpatialDropout1D(0.3))\n model.add(GRU(300))\n model.add(Dense(1, activation='sigmoid'))\n\n model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   \n    \nmodel.summary()")

#%%time
# # GRU with glove embeddings and two dense layers
#  model = Sequential()
#  model.add(Embedding(len(word_index) + 1,
#                  300,
#                  weights=[embedding_matrix],
#                  input_length=max_len,
#                  trainable=False))
#  model.add(SpatialDropout1D(0.3))
#  model.add(GRU(300))
#  model.add(Dense(1, activation='sigmoid'))
# 
#  model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
#     
# model.summary()

# model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64)

# scores = model.predict(xvalid_pad)
# 

# ## Bidirectional RNN Model


# 
# # A simple bidirectional LSTM with glove embeddings and one dense layer
# model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                  300,
#                  weights=[embedding_matrix],
#                  input_length=max_len,
#                  trainable=False))
# model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))
# 
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
#     
#     
# model.summary()

# model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64)

# scores = model.predict(xvalid_pad)
# 

# ## Seq2seq Architecture




# 
# 

# ## Transformers/Attention/BERT

# # Loading Dependencies
# import os
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint
# from kaggle_datasets import KaggleDatasets
# import transformers
# 
# from tokenizers import BertWordPieceTokenizer

# Encoder FOr DATA for understanding waht encode batch does read documentation of hugging face tokenizer :
# https://huggingface.co/transformers/main_classes/tokenizer.html here

# def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
#     """
#     Encoder for encoding the text into sequence of integers for BERT Input
#     """
#     tokenizer.enable_truncation(max_length=maxlen)
#     tokenizer.enable_padding(max_length=maxlen)
#     all_ids = []
#     
#     for i in tqdm(range(0, len(texts), chunk_size)):
#         text_chunk = texts[i:i+chunk_size].tolist()
#         encs = tokenizer.encode_batch(text_chunk)
#         all_ids.extend([enc.ids for enc in encs])
#     
#     return np.array(all_ids)

# #IMP DATA FOR CONFIG
# 
# AUTO = tf.data.experimental.AUTOTUNE
# 
# 
# # Configuration
# EPOCHS = 3
# BATCH_SIZE = 16 
# MAX_LEN = 192

# ## Tokenization
# 
# For understanding please refer to hugging face documentation again

# # First load the real tokenizer
# tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# # Save the loaded tokenizer locally
# tokenizer.save_pretrained('.')
# # Reload it with the huggingface tokenizers library
# fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
# fast_tokenizer

# x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
# x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
# x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
# 
# y_train = train1.toxic.values
# y_valid = valid.toxic.values

# train_dataset = (
#     tf.data.Dataset
#     .from_tensor_slices((x_train, y_train))
#     .repeat()
#     .shuffle(2048)
#     .batch(BATCH_SIZE)
#     .prefetch(AUTO)
# )
# 
# valid_dataset = (
#     tf.data.Dataset
#     .from_tensor_slices((x_valid, y_valid))
#     .batch(BATCH_SIZE)
#     .cache()
#     .prefetch(AUTO)
# )
# 
# test_dataset = (
#     tf.data.Dataset
#     .from_tensor_slices(x_test)
#     .batch(BATCH_SIZE)
# )

# def build_model(transformer, max_len=512):
#     """
#     function for training the BERT model
#     """
#     input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     sequence_output = transformer(input_word_ids)[0]
#     cls_token = sequence_output[:, 0, :]
#     out = Dense(1, activation='sigmoid')(cls_token)
#     
#     model = Model(inputs=input_word_ids, outputs=out)
#     model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#     
#     return model

# ## Starting Training
# 
# If you want to use any another model just replace the model name in transformers._____ and use accordingly

## %%time
# with strategy.scope():
#     transformer_layer = (
#         transformers.TFDistilBertModel
#         .from_pretrained('distilbert-base-multilingual-cased')
#     )
#     model = build_model(transformer_layer, max_len=MAX_LEN)
# model.summary()

n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# n_steps = x_valid.shape[0] // BATCH_SIZE
# train_history_2 = model.fit(
#     valid_dataset.repeat(),
#     steps_per_epoch=n_steps,
#     epochs=EPOCHS*2
# )
