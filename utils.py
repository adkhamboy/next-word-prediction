import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
import pickle
data = pd.read_excel('data/data_uz_new.xlsx')


tokenizer = pickle.load(open('utilities/tokenizer_2810.pkl', 'rb'))

total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in data['Content']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[1]

def generate_word_best(next_words,seed_text,model):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)

        classes_x = np.argmax(predicted, axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == classes_x:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

