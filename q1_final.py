import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import time
import re, string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud, STOPWORDS
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import os
from matplotlib.font_manager import FontProperties
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K



def creating_data(language,path = "/content/drive/MyDrive/dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv"):
    
    val_tsv = path.format(language, language, "dev")
    test_tsv = path.format(language, language, "test")
    train_tsv = path.format(language, language, "train")
    return train_tsv, val_tsv, test_tsv

def model_layer_type(name, units, dropout, return_state=False, return_sequences=False):
    temp = layers.GRU(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)    
    if name=="rnn":
      temp = layers.SimpleRNN(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)
    elif name == 'lstm':
      temp = layers.LSTM(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)
    return temp

def create_layer_for_Enc(no_of_layer, layer_type, units, dropout):
  temp = []
  for i in range(no_of_layer):
    ly = model_layer_type(layer_type, units, dropout, return_state=True, return_sequences=True)
    temp.append(ly)
  return temp

class Encoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, encoder_vocab_size, embedding_dim, dropout):
        super(Encoder, self).__init__()
        self.layer_type = layer_type
        self.dropout = dropout
        self.n_layers = n_layers
        self.units = units
        self.rnn_layers = create_layer_for_Enc(self.n_layers, self.layer_type, self.units, self.dropout)
        self.embedding = tf.keras.layers.Embedding(encoder_vocab_size, embedding_dim)
        
        
    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.rnn_layers[0](x, initial_state=hidden)
        temp = self.rnn_layers[1:]
        for layer in temp:
            x = layer(x)
        return x[0], x[1:]
    
def create_layer_for_Dec(no_of_layer, layer_type, units, dropout):
  temp = [] 
  for i in range(no_of_layer):            
    if i == no_of_layer-1:
      ly = model_layer_type(layer_type, units, dropout,return_sequences=False,return_state=True)
    else:
      ly = model_layer_type(layer_type, units, dropout,return_sequences=True,return_state=True)
    temp.append(ly)
  return temp

class Decoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, decoder_vocab_size, embedding_dim, dropout, attention=False):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.units = units
        self.attention = attention
        self.dense = layers.Dense(decoder_vocab_size, activation="softmax")        
        self.layer_type = layer_type
        self.dropout = dropout
        self.flatten = layers.Flatten() 
        self.rnn_layers = create_layer_for_Dec(self.n_layers, self.layer_type, self.units, self.dropout)
        self.embedding_layer = layers.Embedding(input_dim=decoder_vocab_size,output_dim=embedding_dim)      

    def call(self, x, hidden, enc_out=None):
        
        x = self.embedding_layer(x)
        x = self.rnn_layers[0](x, initial_state=hidden)
        temp = self.rnn_layers[1:]
        for layer in temp:
            x = layer(x)
        return self.dense(self.flatten(x[0])), x[1:], None
    

class Seq2SeqModel():
    def __init__(self, embedding_dim, encoder_layers, decoder_layers, layer_type, units, dropout, loss, optimizer, metric, attention=False):
        
        self.layer_type = layer_type
        self.encoder_layers = encoder_layers
        self.stats = []
        self.decoder_layers = decoder_layers
        self.attention = attention
        self.embedding_dim = embedding_dim
        self.units = units
        self.batch_size = 128
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
    
    def create_model(self):

        encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        decoder_vocab_size = len(self.targ_tokenizer.word_index) + 1

        self.encoder = Encoder(self.layer_type, self.encoder_layers, self.units, encoder_vocab_size,
                               self.embedding_dim, self.dropout)

        self.decoder = Decoder(self.layer_type, self.decoder_layers, self.units, decoder_vocab_size,
                               self.embedding_dim,  self.dropout, self.attention)

    @tf.function
    def train_step(self, input, target, enc_state):

        loss = 0 

        with tf.GradientTape() as tape: 

            enc_out, enc_state = self.encoder(input, enc_state)

            dec_state = enc_state
            dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

            ## We use Teacher forcing to train the network
            ## Each target at timestep t is passed as input for timestep t + 1

            if random.random() < self.teacher_forcing_ratio:

                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)
                    
                    dec_input = tf.expand_dims(target[:,t], 1)         
            else:
                for t in range(1, target.shape[1]):
                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)

                    preds = tf.argmax(preds, 1)
                    dec_input = tf.expand_dims(preds, 1)
            batch_loss = loss / target.shape[1]
            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, self.metric.result()
    @tf.function
    def validation_step(self, input, target, enc_state):
        loss = 0  
        enc_out, enc_state = self.encoder(input, enc_state)
        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)
        for t in range(1, target.shape[1]):
            preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
            loss += self.loss(target[:,t], preds)
            self.metric.update_state(target[:,t], preds)

            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        batch_loss = loss / target.shape[1]
        
        return batch_loss, self.metric.result()
    def fit(self, dataset, val_dataset, batch_size=128, epochs=10, use_wandb=False, teacher_forcing_ratio=1.0):
        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        steps_per_epoch = len(dataset) // self.batch_size
        steps_per_epoch_val = len(val_dataset) // self.batch_size
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)
        sample_inp, sample_targ = next(iter(dataset))
        self.max_target_len = sample_targ.shape[1]
        self.max_input_len = sample_inp.shape[1]
        print("------------------------------------------------------------------------------------------------------------------------------------------")
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}\n")

            ## Training loop ##
            total_loss = 0
            total_acc = 0
            self.metric.reset_states()

            if self.layer_type != "lstm":
              enc_state = [tf.zeros((batch_size, self.units))]
            else:
              enc_state = [tf.zeros((batch_size, self.units))]*2      
            print("Training ...\n")
            for batch, (input, target) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss, acc = self.train_step(input, target, enc_state)
                total_loss += batch_loss
                total_acc += acc


                if batch==0 or ((batch + 1) % 100 == 0):
                    print(f"Batch {batch+1} Loss {batch_loss:.4f}")

            avg_acc = total_acc / steps_per_epoch
            avg_loss = total_loss / steps_per_epoch

            total_val_acc = 0
            total_val_loss = 0
            
            self.metric.reset_states()

            if self.layer_type != "lstm":
              enc_state = [tf.zeros((batch_size, self.units))]
            else:
              enc_state = [tf.zeros((batch_size, self.units))]*2

            print("\nValidating ...")
            for batch, (input, target) in enumerate(val_dataset.take(steps_per_epoch_val)):
                batch_loss, acc = self.validation_step(input, target, enc_state)
                total_val_loss += batch_loss
                total_val_acc += acc

            avg_val_acc = total_val_acc / steps_per_epoch_val
            avg_val_loss = total_val_loss / steps_per_epoch_val

            print(f"\nTrain Loss: {avg_loss} Train Accuracy: {avg_acc*100} Validation Loss: { avg_val_loss} Validation Accuracy: {avg_val_acc*100}")           
          
        print("\nOur Model trained successfully.....")
 


def Train_Model(language,type_layer,encoder_layers,decoder_layers,units,dropout,attention,embedding_dim,test_beam_search=False):
    ## 1. Our Language ##
    TRAIN_TSV, VAL_TSV, TEST_TSV = creating_data(language)

    ## 2. data proccessed ##

    dataframe = pd.read_csv(TRAIN_TSV, sep="\t", header=None)
    def add_tokens(s, sos="\t", eos="\n"):  
        return sos + str(s) + eos    
    cols = [0,1]
    for col in cols:
        dataframe[col] = dataframe[col].apply(add_tokens)
    
    tokenizer = None
    if tokenizer is None:
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(dataframe[1].astype(str).tolist())
    lang_tensor = tokenizer.texts_to_sequences(dataframe[1].astype(str).tolist())
    lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,padding='post')      
    input_lang_tensor, input_tokenizer = lang_tensor, tokenizer

    tokenizer = None
    if tokenizer is None:
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(dataframe[0].astype(str).tolist())
    lang_tensor = tokenizer.texts_to_sequences(dataframe[0].astype(str).tolist())
    lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,padding='post')

    targ_lang_tensor, targ_tokenizer = lang_tensor, tokenizer
    dataset = tf.data.Dataset.from_tensor_slices((input_lang_tensor, targ_lang_tensor))
    dataset = dataset.shuffle(len(dataset))    
    
    dataframe = pd.read_csv(VAL_TSV, sep="\t", header=None)
    def add_tokens(s, sos="\t", eos="\n"):  
        return sos + str(s) + eos    
    cols = [0,1]
    for col in cols:
        dataframe[col] = dataframe[col].apply(add_tokens)
    lang_tensor = tokenizer.texts_to_sequences(dataframe[1].astype(str).tolist())
    lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,padding='post')    
    input_lang_tensor = lang_tensor

    lang_tensor = tokenizer.texts_to_sequences(dataframe[0].astype(str).tolist())
    lang_tensor = tf.keras.preprocessing.sequence.pad_sequences(lang_tensor,padding='post')
    targ_lang_tensor = lang_tensor
    val_dataset = tf.data.Dataset.from_tensor_slices((input_lang_tensor, targ_lang_tensor))
    val_dataset = dataset.shuffle(len(val_dataset))
    model = Seq2SeqModel(embedding_dim,encoder_layers, decoder_layers,type_layer,units,dropout,loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer = tf.keras.optimizers.Adam(),metric = tf.keras.metrics.SparseCategoricalAccuracy(), attention=False)                                                                                                                                                                                                                          
    model.input_tokenizer = input_tokenizer
    model.targ_tokenizer = targ_tokenizer
    model.create_model()   
    model.fit(dataset, val_dataset, epochs=10, use_wandb=True, teacher_forcing_ratio=1.0)
#Train_Model("hi", test_beam_search=False)


#embedding_dim=256,
#encoder_layers=3
#decoder_layers=3,
#layer_type="lstm",
#units=256,
#dropout=0,
#attention=False
#test_beam_search=False
#Train_Model("hi",type_layer,encoder_layers,decoder_layers,units,dropout,attention,embedding_dim,test_beam_search)

from sys import argv

if __name__ == "__main__":

    if(len(argv) !=9):
        print("Invalid num of parameters passed ")
        exit()
    type_layer=argv[1]
    encoder_layers=int(argv[2])
    decoder_layers=int(argv[3])
    units=int(argv[4])
    dropout=float(argv[5])
    if argv[6] == "True":
      attention = True
    else:
      attention=False
    embedding_dim=int(argv[7])

    if argv[8]=="True":
      test_beam_search=True
    else:
      test_beam_search=False

    Train_Model("hi",type_layer,encoder_layers,decoder_layers,units,dropout,attention,embedding_dim,test_beam_search)