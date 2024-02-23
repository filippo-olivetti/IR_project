import pickle
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import bz2


def create_index():
    """
    Creates index tf-idf indexes
    """

    df = pd.read_pickle('./Storage/df.pkl', 'bz2')
    zone_df = pd.read_pickle('./Storage/zone_df.pkl', 'bz2')

    with open('./Storage/words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open('./Storage/doc_no.pkl', 'rb') as f:
        doc_no = pickle.load(f)


    # Constructing a dictionary containing the term and it's inverse document frequency. Formula: idf=log(N/tf)
    inv_doc_freq = {}
    no_of_docs = len(doc_no)
    for word in words:
        inv_doc_freq[word] = np.log10(no_of_docs / sum(df[word] > 0))

    inv_doc_freq_file = open('./Storage/inv_doc_freq.pkl', 'wb')
    pickle.dump(inv_doc_freq, inv_doc_freq_file)
    inv_doc_freq_file.close()

    # Creating and population a dictionary containg the vector of the documents
    doc_vec = {}
    for doc_id in doc_no:
        # Creating a vector for each document
        vec = (1 + np.log10(np.array(df.loc[doc_id])))  # *list(doc_freq.values())
        # Replacing all -inf values with zeros. -inf reached when we take log of 0
        vec[vec == -np.inf] = 0
        # Normalizing the vector
        vec = vec / (np.sqrt(sum(vec ** 2)))
        # Storing the vector
        doc_vec[doc_id] = vec
        print("\r" + "Document Vector created for doc_no:" + str(doc_id), end="")

    doc_vec_file = bz2.BZ2File('./Storage/doc_vec.pkl', 'w')
    pickle.dump(doc_vec, doc_vec_file)
    doc_vec_file.close()

    # Creating and population a dictionary containing the vector of the documents
    zone_vec = {}
    for doc_id in doc_no:
        # Creating a vector for each document
        vec = (1 + np.log10(np.array(zone_df.loc[doc_id])))  # *list(doc_freq.values())
        # Replacing all -inf values with zeros. -inf reached when we take log of 0
        vec[vec == -np.inf] = 0
        # Normalizing the vector
        vec = vec / (np.sqrt(sum(vec ** 2)))
        # Storing the vector
        zone_vec[doc_id] = vec
        print("\r" + "Zone Vector created for doc_no:" + str(doc_id), end="")

    zone_vec_file = open('./Storage/zone_vec.pkl', 'wb')
    pickle.dump(zone_vec, zone_vec_file)
    zone_vec_file.close()

    print("\nDocument vector creation done")
