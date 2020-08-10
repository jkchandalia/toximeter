import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import transformers
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, History
from tensorflow.keras.metrics import Accuracy, AUC
from tokenizers import BertWordPieceTokenizer


#Load BERT tokenizers and transformers
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
transformer_layer = (
    transformers.TFDistilBertModel
    .from_pretrained('distilbert-base-cased')
    )

def find_max_len(texts):
    max_len = int(round(texts.apply(lambda x:len(str(x).split())).max(), -2)+100)
    print("Max length of comment text is: {}".format(max_len))
    return max_len

def make_callbacks(dir_path, project_name):
  # Create a callback for tensorboard
  tb_callback = TensorBoard(log_dir=dir_path+'Graph/'+project_name, histogram_freq=0, write_graph=True, write_images=True)

  # Create a callback that saves the model's weights every epoch
  checkpoint_path = dir_path+'training/'+project_name+'/cp-{epoch:04d}.ckpt'
  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = ModelCheckpoint(
      filepath=checkpoint_path, 
      verbose=1, 
      save_weights_only=True,
      save_freq='epoch',
      period=5)

  # Callback for early stopping if model isn't improving
  es = EarlyStopping(
      monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto',
      baseline=None, restore_best_weights=True
  )
  return [cp_callback, tb_callback]

def load_model_from_checkpoint(model, checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

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
