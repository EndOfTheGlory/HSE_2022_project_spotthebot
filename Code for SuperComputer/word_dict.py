import numpy as np
import pandas as pd
import sys
import json



def getting_dict(sigma: np.ndarray, u: np.ndarray, full_words_list: np.ndarray, shape: int):
    # getting number for words
    words_vector_mult = np.dot(u, np.diag(sigma))
    words_vect_dict = dict(zip(full_words_list, words_vector_mult))
    new_dict = {}
    for first, second in words_vect_dict.items():
        new_dict[first] = second.tolist()[:shape]
    return new_dict

all_tfidfedwords = np.load(sys.argv[1])
u = np.load(sys.argv[2])
sig = np.load(sys.argv[3])

SHAPE_OF_VECT = 1024

word_and_vector_dictionary = getting_dict(sig, u, all_tfidfedwords, SHAPE_OF_VECT)

# saving sa json type
name_of_the_file = "JP_word_dict.json"
with open(name_of_the_file, "w") as file_name:
    json.dump(word_and_vector_dictionary, file_name)