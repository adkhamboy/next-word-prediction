import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

import pickle



tokenizer = pickle.load(open('utilities/tokenizer_2810.pkl', 'rb'))

total_words = len(tokenizer.word_index) + 1


def generate_word_best(next_words,seed_text,model):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=95, padding='pre')
        predicted = model.predict(token_list, verbose=0)

        classes_x = np.argmax(predicted, axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == classes_x:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

