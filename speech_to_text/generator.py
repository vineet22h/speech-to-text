import numpy as np
import math
import re
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .utils import spectrogram_from_file

class BatchGenerator(Sequence):
    def __init__(self, train_file, char_index, batch_size, num_classes, char_maxlen, audio_maxlen, shuffle = True):
        self.paths = train_file.path.values
        self.labels = train_file.label.values
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.char_maxlen = char_maxlen
        self.audio_maxlen = audio_maxlen
        self.char_index = char_index
        self.shuffle = shuffle      
        
        if self.shuffle :
            np.random.shuffle([self.paths, self.labels])
    
    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle :
            np.random.shuffle([self.paths, self.labels])

    def preprocess_label(self, text):
        label = []
        text = text.lower()
        text = re.sub('[^A-Za-z0-9 ]+', '', text)
        label.append(self.char_index['<SOT>'])
        for char in text:
            label.append(self.char_index[char])
        label.append(self.char_index['<EOT>'])
        inp = pad_sequences([label[:-1]], maxlen = self.char_maxlen+2, padding = 'post')[0]
        out = pad_sequences([label[1:]], maxlen = self.char_maxlen+2, padding = 'post')[0]
        return inp, out

    def __getitem__(self, idx):     
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size
        
        if r_bound > len(self.paths):
            r_bound = len(self.paths)
            l_bound = r_bound - len(self.paths)
        
        enc_inp = []
        dec_inp = []
        y = np.zeros((self.batch_size, self.char_maxlen+2))
        for ind, i in enumerate(range(l_bound, r_bound)):   
            if ind >= self.batch_size or ind < 0:
                continue
  
            x_data = spectrogram_from_file(self.paths[i])
            x_data = pad_sequences([x_data], maxlen = self.audio_maxlen, padding = 'post', dtype = np.float32)[0]
            inp, out = self.preprocess_label(self.labels[i])
            enc_inp.append(x_data)
            dec_inp.append(inp)
            y[ind] = out
        return tuple([[tf.convert_to_tensor(enc_inp), tf.convert_to_tensor(dec_inp)], tf.convert_to_tensor(y)])