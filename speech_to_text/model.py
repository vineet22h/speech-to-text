import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical

class Model(tf.keras.Model):
    def __init__(self, char_maxlen, audio_maxlen, lstm_size, num_classes, embedding_size, batch_size):
        super().__init__()
        self.num_classes = num_classes
        self.char_maxlen = char_maxlen
        self.audio_maxlen = audio_maxlen
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.enc_lstm = LSTM(units=lstm_size, return_sequences=True, return_state = True, recurrent_dropout=0.0)
        self.char_embed = Embedding(input_dim = num_classes, output_dim = embedding_size, input_length = char_maxlen)
        self.dec_lstm = LSTM(units= lstm_size, return_sequences=True, return_state = True, recurrent_dropout=0.0)
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, x):
        return self.model(x)
    
    def initialize_states(self):
        return [tf.zeros((self.batch_size, self.lstm_size)), tf.zeros((self.batch_size, self.lstm_size))]
 
    def model(self, x):
        enc_inp, dec_inp = x[0], x[1]
        output_state, enc_h, enc_c = self.enc_lstm(enc_inp, initial_state = self.initialize_states())
        char_embed = self.char_embed(dec_inp)
        output_state, _, _ = self.dec_lstm(char_embed, initial_state = [enc_h, enc_c])
        return self.fc(output_state)

class PredModel(tf.keras.Model):
    def __init__(self, char_index, char_maxlen, audio_maxlen, lstm_size, num_classes, embedding_size, batch_size):
        super().__init__()
        self.num_classes = num_classes
        self.char_maxlen = char_maxlen
        self.audio_maxlen = audio_maxlen
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.char_index = char_index
        self.enc_lstm = LSTM(units=lstm_size, return_sequences=True, return_state = True, recurrent_dropout=0.0)
        self.char_embed = Embedding(input_dim = num_classes, output_dim = embedding_size, input_length = char_maxlen)
        self.dec_lstm = LSTM(units= lstm_size, return_sequences=True, return_state = True, recurrent_dropout=0.0)
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, x):
#         print(x)
        return self.model(x)
    
    def initialize_states(self):
        return [tf.zeros((self.batch_size, self.lstm_size)), tf.zeros((self.batch_size, self.lstm_size))]
 
    def model(self, x):
        enc_inp= x
        output_state, enc_h, enc_c = self.enc_lstm(enc_inp, initial_state = self.initialize_states())
        pred = tf.expand_dims([self.char_index['<SOT>']], 0)
        dec_h = enc_h
        dec_c = enc_c
        all_pred = []
        for t in range(self.char_maxlen): 
            pred = self.char_embed(pred)
            pred, dec_h, dec_c = self.dec_lstm(pred, [dec_h, dec_c])
            pred = self.fc(pred)
            pred = tf.argmax(pred, axis = -1)
            all_pred.append(pred)
        return all_pred