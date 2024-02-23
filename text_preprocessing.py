import re
import pickle
import os
import string
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import bz2
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Intializing stopwords list
stop_list = stopwords.words('english')

# Initializing Porter Stemmer object
st = PorterStemmer()

nltk.download('punkt')


def corpus_preprocessing(location):
    """
    Do a preprocessing on each document of the corpus
    :param location: address to the text corpus
    :return: None
    """
    # Creating a list of document ids
    doc_no = []
    # Creating a list of words in the documents
    words = []
    # Creating a list of words in the document zones i.e headings
    zone_words = []

    # Stores the document id and it's corresponding zone i.e heading
    zone = {}

    # Stores the document id and corresponding tokenized words of the document
    tokenized = {}

    # Stores the document id and corresponding tokenized words of the document zone
    zone_tokenized = {}

    # Opening the corpus and reading the file
    f = open(location, 'r', encoding='utf8')
    content = f.read()
    content = str(content)

    # Removing <a>...</a> tags
    pattern = re.compile("<(/)?a[^>]*>")
    content_new = re.sub(pattern, "", content)

    # Creating a folder to hold the seperated documents
    if not os.path.exists("./Documents"):
        os.mkdir("./Documents")

    # Creating the folder to store dictionaries as pickle files
    if not os.path.exists("./Storage"):
        os.mkdir("./Storage")

    # Creating a soup using a html parser and iterating through each 'doc'
    soup = BeautifulSoup(content_new, 'html.parser')
    for doc in soup.findAll('doc'):
        # Opening a file to write the contents of the doc
        o = open('./Documents/' + str(doc['id']) + ".txt", 'w', encoding='utf8')

        # Adding the document id to doc_no and extracting the text in that doc
        doc_no = doc_no + [(int(doc['id']))]
        text = doc.get_text()

        # Writing the text and closing the file
        o.write(doc.get_text())
        o.close()

        # Storing the heading of the document in the dictionary called 'zone'
        zone[int(doc['id'])] = str(text).partition('\n\n')[0][1:]

        # Extracting the heading of the document
        zone_text = zone[int(doc['id'])]

        # 1. Making all the text lowercase
        text = text.lower()
        zone_text = zone_text.lower()
        # 2. Removing numbers
        text = re.sub(r'[0-9]', '', text)
        zone_text = re.sub(r'[0-9]', '', zone_text)
        # 3. Replaces punctuations with spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        zone_text = zone_text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # 4. Removes weird punctuations. Add a space and symbol you want to replace respectively
        text = text.translate(str.maketrans("‘’’–——−", '       '))
        zone_text = zone_text.translate(str.maketrans("‘’’–——−", '       '))
        # 5. Remove stop_words +  Stemming words
        text_list = [st.stem(word) for word in text.split() if word not in stop_list]
        text = ' '.join(text_list)
        zone_list = [st.stem(word) for word in zone_text.split() if word not in stop_list]
        zone_text = ' '.join(zone_list)


        # Tokenizing word from the doc and adding it to 'words' dictionary
        words = words + word_tokenize(text)
        zone_words = zone_words + word_tokenize(zone_text)

        # Adding the token stream to a dictionary indexed by doc_id
        tokenized[int(doc['id'])] = word_tokenize(text)
        zone_tokenized[int(doc['id'])] = word_tokenize(zone_text)

        # Eliminating the duplicate words
        words = list(set(words))
        zone_words = list(set(zone_words))

        # Printing progress of processing documents
        print("\r" + "Parsing Progress: Document_id = " + doc['id'] + " : " + zone[int(doc['id'])], end='')
    f.close()

    zone_file = open('./Storage/zone.pkl', 'wb')
    pickle.dump(zone, zone_file)
    zone_file.close()

    doc_no_file = open('./Storage/doc_no.pkl', 'wb')
    pickle.dump(doc_no, doc_no_file)
    doc_no_file.close()

    words_file = open('./Storage/words.pkl', 'wb')
    pickle.dump(words, words_file)
    words_file.close()

    zone_words_file = open('./Storage/zone_words.pkl', 'wb')
    pickle.dump(zone_words, zone_words_file)
    zone_words_file.close()

    zone_file = open('./Storage/zone.pkl', 'wb')
    pickle.dump(zone, zone_file)
    zone_file.close()

    tokenized_file = open('./Storage/tokenized.pkl', 'wb')
    pickle.dump(tokenized, tokenized_file)
    tokenized_file.close()

    zone_tokenized_file = open('./Storage/zone_tokenized.pkl', 'wb')
    pickle.dump(zone_tokenized, zone_tokenized_file)
    zone_tokenized_file.close()
    print("\nDocuments separated and parsed")

    # Creating empty dataframe
    df = pd.DataFrame(0, index=doc_no, columns=words)
    zone_df = pd.DataFrame(0, index=doc_no, columns=zone_words)

    # Populating Document-Term Frequency Table
    for doc_id, tokenstream in tokenized.items():
        print("\r" + "Populating Document-Term Frequency Table with doc " + str(doc_id), end="")
        for token in tokenstream:
            df[token].loc[doc_id] += 1

    df.to_pickle('./Storage/df.pkl', 'bz2')

    # Populating Zone-Term Frequency Table
    for doc_id, tokenstream in zone_tokenized.items():
        print("\r" + "Populating Zone-Term Frequency Table with doc " + str(doc_id), end="")
        for token in tokenstream:
            zone_df[token].loc[doc_id] += 1

    zone_df.to_pickle('./Storage/zone_df.pkl', 'bz2')
    print("\nPopulating Term-Frequency Table done")
