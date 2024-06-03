from services import process_query,read_processed_data_from_file,read_inverted_index_from_file,parse_qrels,read_queries_from_file,get_query_id_from_query,read_tfidf_representation;
from train_tf_idf import retrieve_matching_docs,calculate_cosine_similarity,get_retrieved_docs,get_query_tfidf_representation;
import pickle
def get_relevant_docs(query_id, relevant_docs):
    if query_id in relevant_docs:
        return relevant_docs[query_id]


def precision_at_k(retrieved_docs, relevant_docs, k):
    if k == 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    relevant_and_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    precision = len(relevant_and_retrieved) / k
    return precision


def precision_at_k_for_all_2(retrieved_docs, relevant_docs, k):
    if k == 0:
        return 0.0

    total_precision = 0.0
    num_queries = len(retrieved_docs)  # Assuming both dicts have the same set of keys
    query_c = 0
    for query_id in retrieved_docs:
        precision = precision_at_k(retrieved_docs[query_id], relevant_docs[query_id], k)
        if precision > 0:
            query_c += 1
            # print(precision)
            total_precision += precision
    # avg_precision = total_precision / num_queries
    avg_precision = total_precision / query_c

    return avg_precision


def recall_at_k_for_all(retrieved_docs, relevant_docs, k):
    if k <= 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    num_relevant = 0

    for doc in retrieved_k:
        for query_id, relevant_doc_list in relevant_docs.items():
            if doc in relevant_doc_list:
                num_relevant += 1
                break  # No need to check further query IDs once relevant doc is found

    recall = num_relevant / len(relevant_docs)
    return recall


def reciprocal_rank_for_all(retrieved_docs, relevant_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_for_all(retrieved_docs_dict, relevant_docs_dict):
    reciprocal_ranks = []

    for query_id in retrieved_docs_dict:
        rr = reciprocal_rank_for_all(retrieved_docs_dict[query_id], relevant_docs_dict[query_id])
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

qrels_file = "C:/Users/HP/.ir_datasets/antique/train/qrels"
relevant_docss = parse_qrels(qrels_file)
formatted_output_relevant = {}
for query_id, docs in relevant_docss.items():
    formatted_output_relevant[query_id]= docs

# Save the formatted output to a file
output_file = "C:\\Users\\HP\\.ir_datasets\\antique\\formatted_relevant_docs.txt"
with open(output_file, 'w') as f:
    f.write(formatted_output_relevant)
query_file = "C:/Users/HP/.ir_datasets/antique/train/queries.txt"
queries = read_queries_from_file(query_file)
print(f"Formatted relevant documents saved to '{output_file}'")
inverted_index_output_file = "C:\\Users\\HP\\.ir_datasets\\antique\\inverted_index.txt"
all_retrieved_docs = []
query='';
inverted_index_j = read_inverted_index_from_file(inverted_index_output_file)
with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)
df_query_tfidf = get_query_tfidf_representation(query,loaded_vectorizer )
output_file_path_tfidf = "C:\\Users\\HP\\.ir_datasets\\antique\\tfidf.csv"
df = read_tfidf_representation(output_file_path_tfidf)
output_file_path_test = "C:\\Users\\HP\\.ir_datasets\\antique\\processed_outputt_Copy.csv"
data = read_processed_data_from_file(output_file_path_test)



for  query in queries:

    processed_query = process_query(query)
    matching_docs = retrieve_matching_docs(processed_query, inverted_index_j)
    sorted_docs = calculate_cosine_similarity(matching_docs, df_query_tfidf, df)
    retrieved_docs = get_retrieved_docs(sorted_docs, data)
    all_retrieved_docs.append({'query': get_query_id_from_query(query), 'retrieved_docs': retrieved_docs})

formatted_output_retrieved = {}
for item in all_retrieved_docs:
    formatted_output_retrieved [item['query']]= item['retrieved_docs']
print(formatted_output_retrieved)
print(formatted_output_retrieved)
output_file = "C:/Users/HP/.ir_datasets/antique/formatted_retrieved_docs.txt"
with open(output_file, 'w') as f:
    f.write(formatted_output_retrieved)

# print(f"Formatted retrieved documents saved to '{output_file}'")
mrr_score = mean_reciprocal_rank_for_all(formatted_output_retrieved, formatted_output_relevant)
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")


def average_precision(retrieved_docs, relevant_docs):
    if not relevant_docs:
        return 0.0

    precisions = []
    num_relevant_docs = len(relevant_docs)

    for k, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            precisions.append(precision_at_k(retrieved_docs, relevant_docs, k))

    if not precisions:
        return 0.0

    return sum(precisions) / num_relevant_docs


def average_precision_for_all2(retrieved_docs, relevant_docs):
    avg_precisions = {}

    for query_id in retrieved_docs:
        avg_precisions[query_id] = average_precision(retrieved_docs[query_id], relevant_docs[query_id])

    return avg_precisions


def mean_average_precision2(retrieved_docs, relevant_docs):
    avg_precisions = average_precision_for_all2(retrieved_docs, relevant_docs)
    non_zero_precisions = [prec for prec in avg_precisions.values() if prec != 0]
    # print(non_zero_precisions)
    # map_score = sum(avg_precisions.values()) / len(avg_precisions)
    map_score = sum(non_zero_precisions) / len(non_zero_precisions)
    return map_score

k = 10# Example cutoff for Precision and Recall

precision = precision_at_k_for_all_2(formatted_output_retrieved, formatted_output_relevant, k)
recall = recall_at_k_for_all(formatted_output_retrieved, formatted_output_relevant, k)
map_score = mean_average_precision2(formatted_output_retrieved, formatted_output_relevant)
mrr_score = mean_reciprocal_rank_for_all(formatted_output_retrieved, formatted_output_relevant)

print(f"Precision at {k}: {precision}")
print(f"Recall at {k}: {recall}")
print(f"Mean Average Precision (MAP): {map_score}")
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")