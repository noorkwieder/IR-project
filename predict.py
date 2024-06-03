from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
output_file_path_test = "C:\\Users\\HP\\.ir_datasets\\antique\\processed_outputt_Copy.csv"
output_file_path_tfidf = "C:\\Users\\HP\\.ir_datasets\\antique\\tfidf.csv"
query_file = r"C:\\Users\\HP\\.ir_datasets\\antique\\train\\queries.txt"
inverted_index_output_file = "C:\\Users\\HP\\.ir_datasets\\antique\\inverted_index.txt"

query_id = get_query_id(query, query_file)

df = read_tfidf_representation(output_file_path_tfidf)
df_query_tfidf = get_query_tfidf_representation(query,loaded_vectorizer )
inverted_index_j = read_inverted_index_from_file(inverted_index_output_file)
matching_docs = retrieve_matching_docs(query, inverted_index_j)
sorted_docs = calculate_cosine_similarity(matching_docs,df_query_tfidf,  df)
retrieved_docs = get_retrieved_docs(sorted_docs, data)

@app.route('/')
def index():
    return render_template('index.html')

# API route to retrieve documents from Dataset 1 based on query
@app.route('/api/retrieve_dataset1', methods=['POST'])
def retrieve_dataset1():
    data = request.get_json()
    query = data.get('query')
    print(query)

    documents = [doc for doc in output_file_path_test]
    print(documents)

    return jsonify({'documents': retrieved_docs})

# API route to retrieve documents from Dataset 2 based on query
@app.route('/api/retrieve_dataset2', methods=['POST'])
def retrieve_dataset2():
    data = request.get_json()
    query = data.get('query')
    print(query)


    documents = [doc for doc in dataset2 if query.lower() in list(doc.values())[0].lower()]
    print(documents)

    return jsonify({'documents': documents})

if __name__ == '__main__':
    app.run(debug=True)
