from services import  read_inverted_index_from_file,

def get_relevant_docs(query_id, relevant_docs):
    if query_id in relevant_docs:
        return relevant_docs[query_id]


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


mrr_score = mean_reciprocal_rank_for_all(formatted_output_retrieved, formatted_output_relevant)
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")


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


def precision_at_k(retrieved_docs, relevant_docs, k):
    if k == 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    relevant_and_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    precision = len(relevant_and_retrieved) / k
    return precision

precision = precision_at_k_for_all_2(formatted_output_retrieved, formatted_output_relevant, k)
recall = recall_at_k_for_all(formatted_output_retrieved, formatted_output_relevant, k)
map_score = map_average_precision_for_all(formatted_output_retrieved, formatted_output_relevant)
mrr_score = mean_reciprocal_rank(formatted_output_retrieved, formatted_output_relevant)

print(f"Precision at {k}: {precision}")
print(f"Recall at {k}: {recall}")
print(f"Mean Average Precision (MAP): {map_score}")
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")