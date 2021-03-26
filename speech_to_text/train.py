import glob
import json
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split 
from tensorflow.keras.callbacks import *

from .generator import BatchGenerator
from .model import Model, PredModel

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

files = glob.glob('Data/train-clean-100/*/*/*.trans.txt')
df = []
for file_path in tqdm(files):
    file = open(file_path).read().split('\n')
    file_path = '\\'.join(file_path.split('\\')[:-1])
    for i in range(len(file)):
        a = file[i].split()
        if len(a) > 2:
            filename = file_path+'\\'+a[0]+'.flac'
            df.append([filename, ' '.join(a[1:])])
df = pd.DataFrame(df, columns = ['path', 'label'])
df.head()

train, test = train_test_split(df, test_size = 0.1, random_state = 0)
train, val = train_test_split(train, test_size = 0.1, random_state = 0)

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
         'u', 'v', 'w', 'x', 'y', 'z', ' ', '<SOT>', '<EOT>']
char_index = {char: index+1 for index, char in enumerate(vocab)}
index_char = {index+1: char for index, char in enumerate(vocab)}

char_maxlen = 315
audio_maxlen = 1709
batch_size = 8
num_classes = len(vocab)+1
lstm_size = 64


train_datagen = BatchGenerator(train, char_index, batch_size, num_classes, char_maxlen, audio_maxlen)
val_datagen  = BatchGenerator(val, char_index, batch_size, num_classes, char_maxlen, audio_maxlen)
test_datagen = BatchGenerator(test, char_index, batch_size, num_classes, char_maxlen, audio_maxlen)

model = Model(char_index, char_maxlen, audio_maxlen, lstm_size, num_classes, 100)
model.compile(optimizer="adam", loss= 'sparse_categorical_crossentropy')

callbacks = [ModelCheckpoint('tts_BiLSTM_segmented_{}.h5'.format(lstm_size), save_best_only= True, verbose = 1),
             EarlyStopping(patience = 5, verbose = 1),
             ReduceLROnPlateau(patience = 3, verbose = 1)]

model.fit(x = train_datagen, 
          steps_per_epoch = train.shape[0]//batch_size,
          validation_data = val_datagen,
          validation_steps = val.shape[0]//batch_size,
          epochs = 5,  
          callbacks = callbacks,
          verbose = 1)