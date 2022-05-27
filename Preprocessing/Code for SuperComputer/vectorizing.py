from tqdm import tqdm
import numpy as np
import pandas as pd
import json

def vectorizing_sent(sent: str, words_dict: dict):
    splitted_text = sent.split()
    vects = list(map(words_dict.get, splitted_text))
    result = dict(zip(splitted_text, vects))
    return result

def create_word_dict(sentence: str, words_values: dict):
    word_dict = dict()
    words_in_sent = sentence.split(sep=' ')
    # delete all empty strings
    words_in_sent = list(filter(lambda a: a != '', words_in_sent))
    for one_word in words_in_sent:
        word_dict[one_word] = words_values[one_word]
    return word_dict

def vectorize_corpus_trial(data_arr: list, separator: str, word_vectors: dict):
    all_data_vectorized = list()
    for one_id, article in enumerate(tqdm(data_arr)):
        all_sentences = article[0].split(sep=separator)
        vectorizing_text_data = list()
        for sent_id, sentence in enumerate(tqdm(all_sentences)):
            if not sentence:
                continue
            vected_data = vectorizing_sent(sentence, word_vectors)
            what_to_add = {"article_id": one_id, "sentence_id": sent_id, "sentence_text": create_word_dict(sentence, word_vectors)}
            vectorizing_text_data.append(what_to_add)
        all_data_vectorized.append(vectorizing_text_data)
    return np.array(all_data_vectorized)

# change the name
our_data = pd.read_csv("../Cleaning CSV/fully_cleaned_trial_100_with_sep.csv")
word_and_vector_dictionary = json.load(open("JP_word_dict.json"))

vectors_for_words = vectorize_corpus(our_data.values.tolist(), '\u3002', word_and_vector_dictionary)

np.save("JP_vectorized_matrix", vectors_for_words)