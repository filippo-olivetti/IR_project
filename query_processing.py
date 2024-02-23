import string
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import operator
import itertools
import webbrowser

from collections import Counter


np.seterr(all='ignore')
# Importing the stored files
vocab = pd.read_pickle(r'./Storage/words.pkl')
idf = pd.read_pickle(r'./Storage/inv_doc_freq.pkl')
doc_vector = pd.read_pickle('./Storage/doc_vec.pkl', 'bz2')
zone = pd.read_pickle(r'./Storage/zone.pkl')
zone_vec = pd.read_pickle(r'./Storage/zone_vec.pkl')

# Creating the data-frame to store our query vector and zone vector
buffer = pd.read_pickle('./Storage/df.pkl', 'bz2')
buffer.drop(buffer.index, inplace=True)
buffer.loc[0] = 0
zone_buffer = pd.read_pickle('./Storage/zone_df.pkl', 'bz2')
zone_buffer.drop(zone_buffer.index, inplace=True)
zone_buffer.loc[0] = 0

#Utility function to calculate cosine similarity between 2 vectors
#cosine_similarity = (v1 . v2) / (||v1|| * ||v2||)
#complexity: O(|V|) size of vocabulary
def cosine_similarity(x, y):
    """Calculate cosine similarity between 2 vectors (same dimension).
    Arguments:
        x {numpy.ndarray} -- vector 1
        y {numpy.ndarray} -- vector 2
    
    Returns:
        numpy.float64 -- cosine similarity between vector 1 & vector 2
    """
    if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
        raise ValueError("Input vectors must have numeric data types.")
    if x.shape != y.shape:
        raise ValueError("Input vectors must have the same dimensions.")
    
    if np.all(x == 0) or np.all(y == 0):
        return 0
    return np.dot(x, y)

def vectorize_query(query_words):
    """
    Function to process and retrieve the docs
        -query_words: tokenized query as a list of strings
    return: 
        np.ndarray vector of the query
    """
    # Resetting buffer and zone_buffer
    buffer.loc[0] = 0

    # This is the idf below which do not want to consider the words. Removes very frequent words from the zone.
    for token in query_words:
        if token in buffer.columns:
            buffer[token] += 1

    # Vectorising the query doc frequency and calculating weights
    query_vec = (1+np.log10(np.array(buffer.loc[0])))*list(idf.values())
    query_vec[query_vec == -np.inf] = 0
    query_vec = query_vec/(np.sqrt(sum(query_vec**2)))
    # Converting NaN values to zero
    query_vec = np.nan_to_num(query_vec)

    return query_vec

def get_scores(query_words, is_vec=False, use_zones=True, imp_factor=1.75):
    """
    Function to process and retrieve the docs
        -query_words: tokenized query as a list of strings
        -is_vec: True if query is already vectorized
        -use_zones: bool set to True to enable Zonal indexing
        -imp_factor: float, should be greater than 1 if we want to give more importance to zones
    return: 
        dictionary of scores corresponding to their document id
    """
    # Resetting buffer and zone_buffer
    buffer.loc[0] = 0
    zone_buffer.loc[0] = 0

    # Populating the query term frequency data-frame
    if is_vec == False:
        threshold = 0.1
        # This is the idf below which do not want to consider the words. Removes very frequent words from the zone.
        for token in query_words:
            if token in buffer.columns:
                buffer[token] += 1
                if token in zone_buffer.columns and idf[token] > threshold:
                    zone_buffer[token] += idf[token]

        # Vectorising the query doc frequency and calculating weights
        query_vec = (1+np.log10(np.array(buffer.loc[0])))*list(idf.values())
        query_vec[query_vec == -np.inf] = 0
        query_vec = query_vec/(np.sqrt(sum(query_vec**2)))
        # Converting NaN values to zero
        query_vec = np.nan_to_num(query_vec)

        # Vectorising the query zone doc frequency and calculating weights
        zone_query_vec = np.array(zone_buffer.loc[0])
        zone_query_vec = zone_query_vec/(np.sqrt(sum(zone_query_vec**2)))
        zone_query_vec = np.nan_to_num(zone_query_vec)

    else:
        query_vec = query_words
        
    # Computing scores for the query vector corresponding to each document
    scores = {}
    for doc_id, sub_vector in doc_vector.items():
        scores[doc_id] = cosine_similarity(query_vec, sub_vector)
    
    if is_vec:
        return scores
    # max-val stores the highest score recorded for document content matching
    # We are ADDING EXTRA SCORE if the title also matches
    if use_zones:
        max_val = max(scores.values())
        for doc_id, sub_vector in zone_vec.items():
            scores[doc_id] += cosine_similarity(zone_query_vec, sub_vector)*max_val*(imp_factor-1)

    return scores


def search(preprocessed_query, is_vec=False, imp_factor=1.75, top=10, use_zones=True):
    """
    Searching a free text query to print top 10 doc ids, with their score and title
        -preprocessed_query: preprocessed query
        -is_vec: True if query is already vectorized
        -imp_factor: float, should be greater than 1 if we want to give more importance to zones
        -use_zones: set to True to enable Zonal indexing
        -top: int find the first top documents satisfying the query
    return: 
        -scored_doc_ids: documents ranking
    """
    # Be sure the query is already pre-processed to remove punctuations and special characters

    '''
    scored_doc_ids: final score of top 10 docs
    '''
    scored_doc_ids = []

    # Scoring 
    if is_vec:
        temp_score = get_scores(preprocessed_query, is_vec=True, imp_factor=imp_factor, use_zones=use_zones)
    else:
        temp_score = get_scores(preprocessed_query, is_vec=False, imp_factor=imp_factor, use_zones=use_zones)
    temp_score = dict(sorted(temp_score.items(), key=operator.itemgetter(1), reverse=True))
    scored_doc_ids = list(itertools.islice(temp_score.items(), top))

    #for k, v in scored_doc_ids:
        #print(k, round(v, 3), zone[k])
        # Opening the web-pages in a browser for easy checking
        #webbrowser.open('https://en.wikipedia.org/wiki?curid=' + str(k))

    return [tup for tup in scored_doc_ids if tup[0] != 290]
