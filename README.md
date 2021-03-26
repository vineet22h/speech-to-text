# speech-to-text
The Project deals with creation of a basic model that takes input .flac audio file and returns text corresponding to the audio. Implemented model is basic encoder decoder LSTM model which takes spectrogram input of audio file and returns text corresponding to it.

# Future Work
1. Fine tuning model by increasing lstm size, character embeddings, and maximum length of audio and text files.
2. Using pre-trained character embeddings.
3. Trying better models like Bidirectional LSTM, Bidirectional LSTM with attention mechanism, Transformers, etc.
4. Creating Metric for better evaluation of model which will based on 1-gram, 2-gram, and 3-gram character matching.
