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



########################### CREATING DATASET ###################################
def creating_data(language,path = "/content/drive/MyDrive/dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv"):
    #returning train tsv, val tsv, test tsv
    return path.format(language, language, "train"), path.format(language, language, "dev"), path.format(language, language, "test")

#########################  LAYER TYPE ##########################################
def model_layer_type(name, units, dropout, return_state=False, return_sequences=False):
    temp = layers.GRU(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)    
    if name=="rnn":
      temp = layers.SimpleRNN(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)
    elif name == 'lstm':
      temp = layers.LSTM(units=units, dropout=dropout, return_state=return_state, return_sequences=return_sequences)
    return temp
###############################################################################

def create_layer_for_Enc(no_of_layer, layer_type, units, dropout):
  temp = []
  for i in range(no_of_layer):
    ly = model_layer_type(layer_type, units, dropout, return_state=True, return_sequences=True)
    temp.append(ly)
  return temp
############################## ENCODER #########################################
class Encoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, encoder_vocab_size, embedding_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(encoder_vocab_size, embedding_dim)
        self.dropout = dropout
        self.n_layers = n_layers
        self.units = units
        self.layer_type = layer_type
        self.rnn_layers = create_layer_for_Enc(self.n_layers, self.layer_type, self.units, self.dropout)
        
    def call(self, x, hidden):
      pass
        
    def Enc_out_state(self, x, hidden):
        x = self.embedding(x)
        x = self.rnn_layers[0](x, initial_state=hidden)
        temp = self.rnn_layers[1:]
        for layer in temp:
            x = layer(x)
        return x[0], x[1:]  #returning enc output and enc state
################################################################################   
def create_layer_for_Dec(no_of_layer, layer_type, units, dropout):
  temp = [] 
  for i in range(no_of_layer):            
    if i == no_of_layer-1:
      ly = model_layer_type(layer_type, units, dropout,return_sequences=False,return_state=True)
    else:
      ly = model_layer_type(layer_type, units, dropout,return_sequences=True,return_state=True)
    temp.append(ly)
  return temp
####################  DECODER ##################################################
class Decoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, decoder_vocab_size, embedding_dim, dropout, attention=False):
        super(Decoder, self).__init__()
        self.dense = layers.Dense(decoder_vocab_size, activation="softmax") 
        self.n_layers = n_layers
        self.units = units
        self.layer_type = layer_type
        self.dropout = dropout
        self.flatten = layers.Flatten() 
        self.rnn_layers = create_layer_for_Dec(self.n_layers, self.layer_type, self.units, self.dropout)
        self.embedding_layer = layers.Embedding(input_dim=decoder_vocab_size,output_dim=embedding_dim)   
        self.attention = attention

    def call(self, x, hidden, enc_out=None): pass

    def Dec_pred_state(self, x, hidden, enc_out=None):        
        x = self.embedding_layer(x)
        x = self.rnn_layers[0](x, initial_state=hidden)
        temp = self.rnn_layers[1:]
        for layer in temp:
            x = layer(x)
        return self.dense(self.flatten(x[0])), x[1:], None # returning dec pred and dec state
################################################################################

#####################  SEQ 2 SEQ MODEL##########################################
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
    	#making object of Encoder class with appropiate params
        self.encoder = Encoder(self.layer_type, self.encoder_layers, self.units, len(self.input_tokenizer.word_index) + 1, self.embedding_dim, self.dropout)
        
        #making object of Decoder class with appropiate params
        self.decoder = Decoder(self.layer_type, self.decoder_layers, self.units, len(self.targ_tokenizer.word_index) + 1, self.embedding_dim,  self.dropout, self.attention)

  
    def train_step(self, input, target, enc_state):
        loss = 0 
        with tf.GradientTape() as tape: 
            enc_out, enc_state = self.encoder.Enc_out_state(input, enc_state)
            dec_input, dec_state = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1), enc_state
            for t in range(target.shape[1]-1):
                    x = t+1
                    preds, dec_state, _ = self.decoder.Dec_pred_state(dec_input, dec_state, enc_out)
                    self.metric.update_state(target[:,x], preds)
                    loss = loss + self.loss(target[:,x], preds)                    
                    dec_input = tf.expand_dims(target[:,x], 1) 

            if random.random() < self.teacher_forcing_ratio:    
               pass
            else:
                preds = tf.argmax(preds, 1)
                dec_input = tf.expand_dims(preds, 1)

            gradients = tape.gradient(loss, self.encoder.variables + self.decoder.variables)
            self.optimizer.apply_gradients(zip(gradients, self.encoder.variables + self.decoder.variables))
        return loss / target.shape[1], self.metric.result()  #return batch_loss and metric for training
        
   
    def validation_step(self, input, target, enc_state):
        loss = 0  
        enc_out, enc_state = self.encoder.Enc_out_state(input, enc_state)
        dec_input, dec_state = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1), enc_state
        for t in range(target.shape[1]-1):
            preds, dec_state, _ = self.decoder.Dec_pred_state(dec_input, dec_state, enc_out)
            loss = loss + self.loss(target[:,t+1], preds)
            self.metric.update_state(target[:,t+1], preds)
            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)        
        return loss / target.shape[1], self.metric.result() #returning batch_size and metric for validation


    def fit(self, dataset, val_dataset, batch_size=128, epochs=10, use_wandb=False, teacher_forcing_ratio=1.0):
        
        self.batch_size = batch_size        
        steps_per_epoch = len(dataset) // self.batch_size
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        sample_inp, sample_targ = next(iter(dataset))
        steps_per_epoch_val = len(val_dataset) // self.batch_size        
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)

        
        self.max_input_len, self.max_target_len = sample_inp.shape[1],sample_targ.shape[1]
        self.teacher_forcing_ratio = teacher_forcing_ratio
       
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
                loss, acc = self.train_step(input, target, enc_state)
                total_loss = total_loss + loss
                total_acc = total_acc + acc
                                
                if (batch+1) % 100 == 0:
                    print(f"Batch {batch+1} Loss {loss:.4f}")
                if batch==0:
                    print(f"Batch {batch+1} Loss {loss:.4f}")  


            avg_acc = total_acc / steps_per_epoch
            avg_loss = total_loss / steps_per_epoch           
            
            self.metric.reset_states()

            if self.layer_type != "lstm":
              enc_state = [tf.zeros((batch_size, self.units))]
            else:
              enc_state = [tf.zeros((batch_size, self.units))]*2

            total_val_acc = 0
            total_val_loss = 0
            print("\nValidating ...")
            for batch, (input, target) in enumerate(val_dataset.take(steps_per_epoch_val)):
                loss, acc = self.validation_step(input, target, enc_state)
                total_val_loss = total_val_loss + loss
                total_val_acc = total_val_acc + acc

            print(f"\nTrain Loss: {avg_loss} Train Accuracy: {avg_acc*100} Validation Loss: { total_val_loss / steps_per_epoch_val} Validation Accuracy: {(total_val_acc / steps_per_epoch_val)*100}")           
          
        print("\nOur Model trained successfully.....")

#############################################################################################################        
def Train_Model(language,type_layer,encoder_layers,decoder_layers,units,dropout,attention,embedding_dim,test_beam_search=False):
    #Language
    TRAIN_TSV, VAL_TSV, TEST_TSV = creating_data(language)

    #Preprocessing the data
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
    
    #model building by calling Sqe2seqModel class with appropiate arguments
    model = Seq2SeqModel(embedding_dim,encoder_layers, decoder_layers,type_layer,units,dropout,loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer = tf.keras.optimizers.Adam(),metric = tf.keras.metrics.SparseCategoricalAccuracy(), attention=False)                                                                                                                                                                                                                          
    model.input_tokenizer = input_tokenizer
    model.targ_tokenizer = targ_tokenizer
    model.create_model()  
    
    #Fitting our model 
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
############################################################################################################################


##################################### CODE FOR COMMAND LINE#################################################################
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
############################################################################################################################
