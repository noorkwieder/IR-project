import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import spacy
from textblob import TextBlob
import datefinder
import datetime


# Load the pre-trained English model for POS lemmatization
nlp = spacy.load('en_core_web_sm')

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
output_file_path = "C:\\Users\\HP\\.ir_datasets\\antique\\processed_outputt_Copy.csv"

import requests
import shutil
import os
import gzip
from gensim.models import KeyedVectors

def load_word2vec_model(model_file):
    return KeyedVectors.load_word2vec_format(model_file, binary=True)


def preprocess_data_and_get_embeddings(data_file, word2vec_model):
    df = pd.read_csv(data_file)

    if 'Content' in df.columns:
        df['Content'] = df['Content'].astype(str)

    processed_data = [(row['ID'], row['Content'].split()) for idx, row in df.iterrows()]

    embeddings = []
    for doc_id, tokens in processed_data:
        doc_embedding = np.zeros(word2vec_model.vector_size)
        num_words = 0

        for word in tokens:
            if word in word2vec_model:
                doc_embedding += word2vec_model[word]
                num_words += 1
        if num_words > 0:
            doc_embedding /= num_words
        embeddings.append((doc_id, doc_embedding))

    return embeddings


def calculate_cosine_similarity(query_embedding, doc_embeddings):
    similarities = {}
    query_norm = np.linalg.norm(query_embedding)

    for doc_id, doc_embedding in doc_embeddings:
        doc_norm = np.linalg.norm(doc_embedding)
        if doc_norm != 0:
            similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
            similarities[doc_id] = similarity

    sorted_docs = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}

    return sorted_docs


# Cell 6: Preprocess and get embeddings for a query
def preprocess_query(query, word2vec_model):
    tokens = process_document_text(query)
    query_embedding = np.zeros(word2vec_model.vector_size)
    num_words = 0

    for word in tokens:
        if word in word2vec_model:
            query_embedding += word2vec_model[word]
            num_words += 1
    if num_words > 0:
        query_embedding /= num_words

    return query_embedding


def get_retrieved_docs(sorted_docs, processed_data):
    retrieved_docs = []
    data_dict = {doc[0]: doc[1] for doc in processed_data}  # Assuming processed_data is (doc_id, data) tuples
    i=0
    for doc_id, similarity in sorted_docs.items():
        if doc_id in data_dict:
            retrieved_docs.append(doc_id)
            i+=1
    print(i)
    return retrieved_docs


def get_retrieved_docs_sim(sorted_docs, processed_data):
    retrieved_docs = []
    data_dict = {doc[0]: doc[1] for doc in processed_data}  # Assuming processed_data is (doc_id, data) tuples
    i = 0

    for doc_id, similarity in sorted_docs.items():
        if similarity > 0.5 and doc_id in data_dict:
            retrieved_docs.append(doc_id)
            i += 1

    print(f"Number of retrieved docs with similarity > 0: {i}")
    return retrieved_docs
