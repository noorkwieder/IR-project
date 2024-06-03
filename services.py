extract_id_and_content,load_documents,read_inverted_index_from_file,process_query,read_processed_data_from_file,read_tfidf_representation,read_queries_from_file,parse_qrels
read_tfidf_representation,get_query_id_from_query,get_query_id

def extract_id_and_content(document_text):
    # Split the document text to separate ID and content
    parts = document_text.split(maxsplit=1)
    if len(parts) > 1:
        document_id = parts[0]
        document_content = parts[1].strip()  # Remove leading/trailing whitespaces
        return document_id, document_content


def load_documents(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            documents.append(line.strip())  # Assuming each line is a document

    doc_ids = [str(i) for i in range(1, len(documents) + 1)]  # Document IDs are 1, 2, ..., N

    return doc_ids, documents


def read_inverted_index_from_file(file_path):
    inverted_index = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                term, postings = line.split(':')
                term = term.strip()
                postings = eval(postings.strip())
                inverted_index[term] = postings

    return inverted_index


def process_query(query):
    # Step 1: Tokenization
    query_tokens = tokenize(query)

    # Step 2: Lowercasing
    query_tokens = lowercase(query_tokens)

    # Step 3: Handle abbreviation expansion
    query_tokens = expand_abbreviations(query_tokens)

    # Step 4: Convert date formats
    query_tokens = convert_dates(query_tokens)

    # Step 5: Lemmatization
    query_tokens = lemmatize(query_tokens)

    # Step 6: Stemming
    query_tokens = stem(query_tokens)

    # Step 7: Remove stop words
    query_tokens = remove_stopwords(query_tokens)

    # Step 8: Remove special characters
    query_tokens = remove_special_chars(query_tokens)

    # Step 9: Remove empty tokens
    query_tokens = remove_empty(query_tokens)

    # Step 10: Spell correction
    query_tokens = spell_correction(query_tokens)

    return query_tokens


def read_processed_data_from_file(file_path):
    df_processed = pd.read_csv(file_path)

    def get_query_tfidf_representation(processed_query_tokens, vectorizer):

        tfidf_matrix_query = vectorizer.transform([' '.join(processed_query_tokens)])
        # Convert the TF-IDF matrix of the query to a Pandas DataFrame
        df_query = pd.DataFrame(tfidf_matrix_query.toarray(), columns=vectorizer.get_feature_names_out())

        return df_query

    # Ensure the 'Content' column contains strings
    if 'Content' in df_processed.columns:
        df_processed['Content'] = df_processed['Content'].astype(str)
    else:
        raise ValueError("'Content' column not found in the CSV file")

    # Convert the 'Content' column to strings and split
    processed_data = [(row['ID'], row['Content'].split()) for idx, row in df_processed.iterrows()]

    return processed_data

def read_tfidf_representation(file_path):
    df = pd.read_csv(file_path)
    return df

def get_query_id_from_query(query):

    query_id = None
    query_parts = query.split('\t')
    if len(query_parts) == 2:
        query_id = int(query_parts[0])
    return query_id

def get_query_id(query, query_file):
    with open(query_file, 'r') as file:
        lines = file.readlines()

    query_id = None
    for line in lines:
        line = line.strip()
        query_parts = line.split('\t')
        if len(query_parts) == 2 and query_parts[1] == query:
            query_id = int(query_parts[0])
            break

    return query_id


def parse_qrels(qrels_file):
    relevant_docs = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if int(relevance) > 0:  # Consider only relevant documents
                query_id = int(query_id)  # Convert query ID to integer

                # doc_id should remain as string to preserve underscores
                if query_id in relevant_docs:
                    relevant_docs[query_id].append(doc_id)
                else:
                    relevant_docs[query_id] = [doc_id]
    return relevant_docs

def read_queries_from_file(file_path):
    with open(file_path, 'r') as file:
        queries = file.readlines()
    # Strip whitespace (including newline characters) from each query
    queries = [query.strip() for query in queries if query.strip()]
    return queries
