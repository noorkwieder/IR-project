from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Dummy datasets (replace with your actual data handling logic)
dataset1 = [{'doc1': 'Document 1 from Dataset 1'}, {'doc2': 'Document 2 from Dataset 1'}]
dataset2 = [{'doc1': 'Document 1 from Dataset 2'}, {'doc2': 'Document 2 from Dataset 2'}]

@app.route('/')
def index():
    return render_template('index.html')

# API route to retrieve documents from Dataset 1 based on query
@app.route('/api/retrieve_dataset1', methods=['POST'])
def retrieve_dataset1():
    data = request.get_json()
    query = data.get('query')
    print(query)

    documents = [doc for doc in dataset1]
    print(documents)

    return jsonify({'documents': documents})

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
