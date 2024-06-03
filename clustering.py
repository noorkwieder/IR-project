from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Initialize lists to store document IDs and content
all_docs_content = []
all_doc_ids = []

# Read the first 100 documents from the file
with open("C:\\Users\\HP\\.ir_datasets\\antique\\processed_outputt_Copy.csv", "r") as file:
    next(file)  # Skip the header line
    doc_count = 0
    for line in file:
        doc_id, content = line.strip().split(",", 1)  # Limit split to only 1 split
        all_doc_ids.append(doc_id)
        all_docs_content.append(content)
        doc_count += 1
        if doc_count >= 100:
            break


# Convert the content of relevant documents to lowercase (optional)
all_docs_content = [doc.lower() for doc in all_docs_content]


# Ensure the vectorizer and KMeans are defined
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_docs_content)


# Apply K-Means clustering
num_clusters = 4  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(X)

# Associate document IDs with clusters
docs_clusters = {}
for doc_id, cluster_id in zip(all_doc_ids, clusters):
    docs_clusters[doc_id] = cluster_id

# Print document IDs and their corresponding clusters
for doc_id, cluster_id in docs_clusters.items():
    print(f"Document {doc_id}: Cluster {cluster_id}")

def retrieve_docs_from_cluster(cluster_id, docs_clusters):
    # Initialize an empty list to store document IDs belonging to the specified cluster
    cluster_docs = [doc_id for doc_id, doc_cluster_id in docs_clusters.items() if doc_cluster_id == cluster_id]
    return cluster_docs


user_query = "Why are truffles so expensive?"

# Preprocess and transform the query
query_vector = vectorizer.transform([user_query])

# Predict the cluster for the query
predicted_cluster = kmeans.predict(query_vector)[0]

# Retrieve documents from the predicted cluster
retrieved_documents = retrieve_docs_from_cluster(predicted_cluster, docs_clusters)

# Display or return the retrieved documents to the user
print("Retrieved documents:")
for doc in retrieved_documents:
    print(doc)
