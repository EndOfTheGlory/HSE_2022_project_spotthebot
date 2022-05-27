import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import svd


our_data = pd.read_csv("../Cleaning CSV/fully_cleaned_trial_100_with_sep.csv")

def remove_separators(dataframe: pd.DataFrame, col_name: str, separator: str):
    '''
    Remove separators from text
    Parameters
    ----------
    dataframe: pd.DataFrame
    col_name: str
    separator : str
    Returns
    -------
    text : pd.DataFrame
    Text without separators
    '''
    for index, one_col in enumerate(dataframe[col_name]):
        without_separators = ' '.join(one_col.split(sep=separator))
        dataframe[col_name][index] = without_separators

data_without_separator = our_data.copy()
remove_separators(data_without_separator, "content", "\u3002")

def creating_article_word_matrix(data: pd.DataFrame):
    # r"\S+" - all words and numbers
    vectorize = TfidfVectorizer(token_pattern=r"\S+")
    # data['article']?
    matrix_art_word = vectorize.fit_transform(data['content'].values)
    return matrix_art_word.toarray(), np.array(vectorize.get_feature_names())

matrix_tfidf, all_tfidfedwords = creating_article_word_matrix(data_without_separator)

print(matrix_tfidf.shape)

# np.save("MAATRIX_WORDS", matrix_tfidf)
# np.save("WORD_LIST", all_tfidfedwords)


u, sig, v = svd(matrix_tfidf.T, full_matrices=False)

# np.save("U", u)
# np.save("sigma", sig)
# np.save("v", v)