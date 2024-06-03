from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from preprocess import process_documents, process_document_text



def load_documents(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            documents.append(line.strip())  # Assuming each line is a document

    doc_ids = [str(i) for i in range(1, len(documents) + 1)]  # Document IDs are 1, 2, ..., N

    return doc_ids, documents


def tfidf_representation(dataset_file_path, output_file_path):
    doc_ids, documents = load_documents(dataset_file_path)

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv(output_file_path, index=False)

    # # Save the TF-IDF matrix representation to a CSV file
    # feature_names = vectorizer.get_feature_names_out()


    return vectorizer


# Example usage
tfidf_representation_path = "C:\\Users\\HP\\.ir_datasets\\antique\\test_tfidf.csv"
processed_dataset_path = "C:\\Users\\HP\\.ir_datasets\\antique\\processed_output_test.csv"

# Generate TF-IDF representation and save the output
vectorizer = tfidf_representation(processed_dataset_path, tfidf_representation_path)
import pickle

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


    def create_inverted_index(tfidf_df, output_file_path):
        inverted_index = {}

        # Ensure tfidf_df[term] is numeric
        tfidf_df = tfidf_df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
        for term in tfidf_df.columns:
            for doc_id, tfidf_score in tfidf_df[term].items():
                if not pd.isnull(tfidf_score) and tfidf_score > 0:
                    if term in inverted_index:
                        inverted_index[term].append((doc_id, tfidf_score))
                    else:
                        inverted_index[term] = [(doc_id, tfidf_score)]

        with open(output_file_path, 'w') as output_file:
            for term, postings in inverted_index.items():
                output_file.write(f'{term}: {postings}\n')

        print(f"Inverted index saved to {output_file_path}")


    # Set the paths
    tfidf_file_path = "C:\\Users\\HP\\.ir_datasets\\antique\\test_tfidf.csv"
    inverted_index_output_file = "C:\\Users\\HP\\.ir_datasets\\antique\\inverted_index.txt"

    tfidf_df = pd.read_csv(tfidf_file_path)
    # Build inverted index and save to file
    create_inverted_index(tfidf_df, inverted_index_output_file)

 def get_query_tfidf_representation(processed_query_tokens, vectorizer):

        tfidf_matrix_query = vectorizer.transform([' '.join(processed_query_tokens)])
        # Convert the TF-IDF matrix of the query to a Pandas DataFrame
        df_query = pd.DataFrame(tfidf_matrix_query.toarray(), columns=vectorizer.get_feature_names_out())

        return df_query

def retrieve_matching_docs(query_tokens,inverted_index):

    # Create a list to store the document IDs matching the query terms
    matching_docs = []

    # Iterate over the query terms
    for query_term in query_tokens:
        # Check if the query term exists in the inverted index
        if query_term in inverted_index:
            # Retrieve the list of document scores for the query term
            doc_scores = inverted_index[query_term]
            # Extract the document IDs from the document scores
            doc_ids = [doc_score[0] for doc_score in doc_scores]
            # Add the document IDs to the matching_docs list
            matching_docs.extend(doc_ids)

    # Remove duplicates from the matching_docs list
    matching_docs = list(set(matching_docs))

    return matching_docs


def calculate_cosine_similarity(matching_docs,df_query,df):
    # Calculate the cosine similarity between the query and each matching document
    cosine_similarities = df.loc[matching_docs,].dot(df_query.values[0])
    # Sort the documents based on cosine similarity in descending order
    sorted_docs = cosine_similarities.sort_values(ascending=False)
    return sorted_docs


def get_retrieved_docs(sorted_docs, processed_data):
    retrieved_docs = []
    data_dict = {i: doc[0] for i, doc in enumerate(processed_data)}  # Convert data to a dictionary for quick access
    # print(data_dict)
    for doc_id, similarity in sorted_docs.items():
        if doc_id in data_dict:
            retrieved_docs.append(data_dict[doc_id])

    return retrieved_docs


