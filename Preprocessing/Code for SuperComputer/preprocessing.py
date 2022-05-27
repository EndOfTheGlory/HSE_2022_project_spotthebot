import pandas as pd
# import numpy as np
import re
# import stanza
import nagisa
# import nltk

full_text = pd.DataFrame()
for chunk in pd.read_csv("../Reading and bringing together wiki/GroupedArticles.csv", chunksize=30):
    full_text = pd.concat([full_text, chunk])
    break


def leave_only_japanese(text_to_fix: str):
    '''
    Function that leave only kanji, hiragana, katakana and period with comma.
    '''
    # we don't need ! or ? => don't use codes \uFF01\uFF1F
    where_is_japanese = r"[\u3040-\u309F \u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A\u3002\uFF01\uFF1F]"
    please_work = ''.join(re.findall(where_is_japanese, text_to_fix))
    return please_work


def do_full_cleaning(our_text: str):
    final_text = leave_only_japanese(our_text)

    return final_text


checker = full_text['content'].apply(lambda text_one_by_one: do_full_cleaning(text_one_by_one))


stopping_words = pd.read_csv("stopwords-ja.txt", header=None)[0]

print(checker)

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def spearating_deleting_stopwords(data, stopwords):
    i = 0
    for one_art in data:
        words_in_art = nagisa.tagging(one_art).words
        for one_stop in stopwords:
            if one_stop in words_in_art:
                words_in_art = remove_values_from_list(words_in_art, one_stop)
        words_in_art = remove_values_from_list(words_in_art, "\u3000")
        new_article_cleaned = ""
        for one_word in words_in_art:
            new_article_cleaned += one_word + " "
        # we don't need ! or ? => we change \uFF01\uFF1F to \u3002
        new_article_cleaned = new_article_cleaned.replace("\uFF01", "\u3002")
        new_article_cleaned = new_article_cleaned.replace("\uFF1F", "\u3002")
        new_article_cleaned = new_article_cleaned.replace("!", "\u3002")
        new_article_cleaned = new_article_cleaned.replace("?", "\u3002")
        data[i] = new_article_cleaned
        i += 1

# whole_pure_text = checker.copy()
# spearating_deleting_stopwords(whole_pure_text, stopping_words)

# nlp = stanza.Pipeline(lang='ja', processors='tokenize, lemma')

# result_whole = whole_pure_text.copy()
# for i in range(len(result_whole)):
#     doc = nlp(result_whole[i])
#     words_array = [word.lemma for sent in doc.sentences for word in sent.words]
#     whole_article_again = ""
#     for word_or_comma in words_array:
#         if word_or_comma != None:
#             whole_article_again += word_or_comma + " "
#     result_whole[i] = whole_article_again
#
# result_whole.to_csv("fully_cleaned_trial_100_with_sep.csv", index=False)