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

    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     # Write header
    #     output_file.write(",".join(feature_names) + "\n")

    #     # Write TF-IDF values
    #     for i in range(len(documents)):
    #         output_file.write(doc_ids[i] + ",")
    #         for j in range(len(feature_names)):
    #             output_file.write(str(tfidf_matrix[i, j]) + ",")
    #         output_file.write("\n")

    # print(f"TF-IDF representation saved to {output_file_path}")

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



