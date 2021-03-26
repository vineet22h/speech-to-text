from tensorflow.keras.preprocessing import pad_sequences
import numpy as np

from .utils import spectrogram_from_file
from .model import PredModel

class Predict:
    def __init__(self):
        vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
                'u', 'v', 'w', 'x', 'y', 'z', ' ', '<SOT>', '<EOT>']
        self.char_index = {char: index+1 for index, char in enumerate(vocab)}
        self.index_char = {index+1: char for index, char in enumerate(vocab)}
        char_maxlen = 315
        self.audio_maxlen = 1709
        batch_size = 8
        num_classes = len(vocab)+1
        embedding_size = 100
        lstm_size = 64
        self.model = PredModel(self.char_index, char_maxlen, self.audio_maxlen, lstm_size, num_classes, embedding_size, batch_size)
        self.model.compile(optimizer="adam", loss= 'sparse_categorical_crossentropy')
        self.model.build(input_shape = (1, self.audio_maxlen, 161))
        self.model.load_weights('weights/tts_BiLSTM_segmented_{}.h5'.format(lstm_size))

    def predict(self, audio_path):
        x_data = spectrogram_from_file(audio_path)
        x_data = pad_sequences([x_data], maxlen = self.audio_maxlen, padding = 'post', dtype = np.float32)
        pred = self.model.predict(x_data)
        sentence = []
        for i in pred:
            if self.index_char[i[0][0]] == '<EOT>':
                break
            sentence.append(self.index_char[i[0][0]])
        return ''.join(sentence)
