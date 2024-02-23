---
runme:
  id: 01HQBHYF5ND4276801C8F9YFX0
  version: v3
---

### Vector Space Model

This repository implements a Vector Space Model. The files are the following:

- ***data***: is a folder containing the txt file with all documents. Documents are more than 400 Wikipedia pages of different topics.

- ***queries***: contains a file with queries, and a file with (true) relevant documents associated to each query.

- ***text_preprocessing.py***: get the corpus, do a pre-processing of all documents and save them in Storage folder. It distinguish between zones (title and body texts).

- ***create_index.py***: stores a function that use pkl files stored in Storage folder to create corpus index. It also creates a dictionary with tuples (DocIDs, tf-idf) values.

- ***query_processing.py***: includes many functions that are useful for 1. doing query pre-processing 2. from pre-processed query get the correpondent vector in the VSM 3. from query vector do cosine similarity with all documents and rank them.

- ***vector_space_model.py***: is the main file.