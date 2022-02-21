import logging
import pickle
import os
import math
import functools
import inspect
import joblib
import click
import torch
import numpy as np
import pandas as pd
from diskcache import Cache
from funcy import print_durations
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from funcy import log_durations
from joblib import Parallel, delayed
from collections import defaultdict
from torch import Tensor


def custom_hash(obj):
    try:
        obj = pd.util.hash_pandas_object(
            obj
        ).values  # for speed else joblib.hash is slow
    except:
        pass

    return joblib.hash(obj)


def cache(name=None, directory="~/.cache/python-cache", ignore=None, hash_source=False):
    """Keep a cache of previous function calls that persists on disk"""
    if ignore is None:
        ignore = set()

    def decorator_cache(func):
        source = inspect.getsource(func)
        source = "\n".join(source.split("\n")[2:])  # remove first line
        file_name = name if name else func.__module__ + "-" + func.__name__

        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            cache_key = [custom_hash(arg) for arg in args] + [
                key + custom_hash(kwarg)
                for key, kwarg in kwargs.items()
                if key not in ignore
            ]
            cache_key = "".join(cache_key)
            if hash_source:
                cache_key += custom_hash(source)

            cache_key = custom_hash(cache_key)

            if cache_key not in wrapper_cache.cache:
                val = func(*args, **kwargs)
                wrapper_cache.cache[cache_key] = val
                return val
            return wrapper_cache.cache[cache_key]

        wrapper_cache.cache = Cache(
            directory + "/" + file_name, size_limit=2 ** 34, eviction_policy="none"
        )

        def cache_clear():
            """Clear the cache"""
            wrapper_cache.cache.clear()

        def cache_close():
            """Close the diskcache"""
            wrapper_cache.cache.close()

        wrapper_cache.cache_clear = cache_clear
        wrapper_cache.cache_close = cache_close

        return wrapper_cache

    return decorator_cache


@cache(ignore=["cores", "batch_size"])
def embed_data(data, key="text", model_name="all-MiniLM-L6-v2", cores=8, batch_size=32):
    """
    Embed the sentences/text using the MiniLM language model
    """
    print("Embedding data")
    model = SentenceTransformer(model_name)
    print("Model loaded")

    sentences = data[key].tolist()

    if (cores == 1) or len(sentences) < 200:
        embeddings = model.encode(
            sentences, show_progress_bar=True, batch_size=batch_size
        )
    else:
        devices = ["cpu"] * cores

        # Start the multi-process pool on multiple devices
        print("Multi-process pool starting")
        pool = model.start_multi_process_pool(devices)
        print("Multi-process pool started")

        chunk_size = math.ceil(len(sentences) / cores)

        # Compute the embeddings using the multi-process pool
        embeddings = model.encode_multi_process(
            sentences, pool, batch_size=batch_size, chunk_size=chunk_size
        )
        model.stop_multi_process_pool(pool)

    print("Embeddings computed")

    mapping = {
        sentence: embedding for sentence, embedding in zip(sentences, embeddings)
    }
    embeddings = {
        idx: mapping[sentence] for idx, sentence in zip(data.index, sentences)
    }

    return embeddings


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(np.array(a))

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(np.array(b))

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_embeddings(ids, embeddings):
    return np.array([embeddings[idx] for idx in ids])


def reorder_and_filter_cluster(
    cluster_idx, cluster, cluster_embeddings, cluster_head_embedding, threshold
):
    cos_scores = cos_sim(cluster_head_embedding, cluster_embeddings)
    sorted_vals, indices = torch.sort(cos_scores[0], descending=True)
    bigger_than_threshold = sorted_vals > threshold
    indices = indices[bigger_than_threshold]
    sorted_vals = sorted_vals.numpy()
    return cluster_idx, [(cluster[i][0], sorted_vals[i]) for i in indices]


def get_ids(cluster):
    return [transaction[0] for transaction in cluster]


def reorder_and_filter_clusters(clusters, embeddings, threshold, parallel):
    results = parallel(
        delayed(reorder_and_filter_cluster)(
            cluster_idx,
            cluster,
            get_embeddings(get_ids(cluster), embeddings),
            get_embeddings([cluster_idx], embeddings),
            threshold,
        )
        for cluster_idx, cluster in tqdm(clusters.items())
    )

    clusters = {k: v for k, v in results}

    return clusters


def get_embeddings(ids, embeddings):
    return np.array([embeddings[idx] for idx in ids])


def get_clustured_ids(clusters):
    clustered_ids = set(
        [transaction[0] for cluster in clusters.values() for transaction in cluster]
    )
    clustered_ids |= set(clusters.keys())
    return clustered_ids


def get_clusters_ids(clusters):
    return list(clusters.keys())


def get_unclustured_ids(ids, clusters):
    clustered_ids = get_clustured_ids(clusters)
    unclustered_ids = list(set(ids) - clustered_ids)
    return unclustered_ids


def sort_clusters(clusters):
    return dict(
        sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    )  # sort based on size


def sort_cluster(cluster):
    return list(
        sorted(cluster, key=lambda x: x[1], reverse=True)
    )  # sort based on similarity


def filter_clusters(clusters, min_cluster_size):
    return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}


def unique(collection):
    return list(dict.fromkeys(collection))


def unique_txs(collection):
    seen = set()
    return [x for x in collection if not (x[0] in seen or seen.add(x[0]))]


def write_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_txt(path):
    with open(path) as f:
        data = f.read().splitlines()
    return data

def chunk(txs, chunk_size):
    if len(txs) == 0:
        return txs
    n = math.ceil(len(txs) / chunk_size)
    k, m = divmod(len(txs), n)
    return (txs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def online_community_detection(
    ids,
    embeddings,
    clusters=None,
    threshold=0.75,
    min_cluster_size=3,
    chunk_size=2500,
    iterations=20,
    cores=1,
):
    if clusters is None:
        clusters = {}

    with Parallel(n_jobs=cores) as parallel:
        for iteration in range(iterations):
            print("1. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            print("\n\n")

            print("2. Create new clusters")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            print("Unclustured", len(unclustered_ids))
            new_clusters = create_clusters(
                unclustered_ids,
                embeddings,
                clusters={},
                min_cluster_size=3,
                chunk_size=chunk_size,
                threshold=threshold,
                parallel=parallel,
            )
            new_cluster_ids = list(new_clusters.keys())
            print("\n\n")

            print("3. Merge new clusters", len(new_cluster_ids))
            max_clusters_size = 25000
            while True:
                new_cluster_ids = list(new_clusters.keys())
                old_new_cluster_ids = new_cluster_ids
                new_clusters = create_clusters(
                    new_cluster_ids,
                    embeddings,
                    new_clusters,
                    min_cluster_size=1,
                    chunk_size=max_clusters_size,
                    threshold=threshold,
                    parallel=parallel,
                )
                new_clusters = filter_clusters(new_clusters, 2)

                new_cluster_ids = list(new_clusters.keys())
                print("New merged clusters", len(new_cluster_ids))
                if len(old_new_cluster_ids) < max_clusters_size:
                    break

            new_clusters = filter_clusters(new_clusters, min_cluster_size)
            print(
                f"New clusters with min community size >= {min_cluster_size}",
                len(new_clusters),
            )
            clusters = {**new_clusters, **clusters}
            print("Total clusters", len(clusters))
            clusters = sort_clusters(clusters)
            print("\n\n")

            print("4. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            clusters = sort_clusters(clusters)

            unclustered_ids = get_unclustured_ids(ids, clusters)
            clustured_ids = get_clustured_ids(clusters)
            print("Clustured", len(clustured_ids))
            print("Unclustured", len(unclustered_ids))
            print(
                f"Percentage clustured {len(clustured_ids) / (len(clustured_ids) + len(unclustered_ids)) * 100:.2f}%"
            )

            print("\n\n")
    return clusters


def get_ids(cluster):
    return [transaction[0] for transaction in cluster]


def nearest_cluster_chunk(
    chunk_ids, chunk_embeddings, cluster_ids, cluster_embeddings, threshold
):
    cos_scores = cos_sim(chunk_embeddings, cluster_embeddings)
    top_val_large, top_idx_large = cos_scores.topk(k=1, largest=True)
    print(top_val_large.shape)
    top_idx_large = top_idx_large[:, 0].tolist()
    top_val_large = top_val_large[:, 0].tolist()
    cluster_assignment = []
    for i, (score, idx) in enumerate(zip(top_val_large, top_idx_large)):
        cluster_id = cluster_ids[idx]
        if score < threshold:
            cluster_id = None
        cluster_assignment.append(((chunk_ids[i], score), cluster_id))
    return cluster_assignment


def nearest_cluster(
    transaction_ids,
    embeddings,
    clusters=None,
    parallel=None,
    threshold=0.75,
    chunk_size=2500,
):
    cluster_ids = list(clusters.keys())
    if len(cluster_ids) == 0:
        return clusters
    cluster_embeddings = get_embeddings(cluster_ids, embeddings)

    c = list(chunk(transaction_ids, chunk_size))

    with log_durations(logging.info, "Parallel jobs nearest cluster"):
        out = parallel(
            delayed(nearest_cluster_chunk)(
                chunk_ids,
                get_embeddings(chunk_ids, embeddings),
                cluster_ids,
                cluster_embeddings,
                threshold,
            )
            for chunk_ids in tqdm(c)
        )
        cluster_assignment = [assignment for sublist in out for assignment in sublist]

    for (transaction_id, similarity), cluster_id in cluster_assignment:
        if cluster_id is None:
            continue
        clusters[cluster_id].append(
            (transaction_id, similarity)
        )

    clusters = {
        cluster_id: unique_txs(sort_cluster(cluster))
        for cluster_id, cluster in clusters.items()
    }  # Sort based on similarity

    return clusters


def create_clusters(
    ids,
    embeddings,
    clusters=None,
    parallel=None,
    min_cluster_size=3,
    threshold=0.75,
    chunk_size=2500,
):
    to_cluster_ids = np.array(ids)
    np.random.shuffle(
        to_cluster_ids
    )

    c = list(chunk(to_cluster_ids, chunk_size))

    with log_durations(logging.info, "Parallel jobs create clusters"):
        out = parallel(
            delayed(fast_clustering)(
                chunk_ids,
                get_embeddings(chunk_ids, embeddings),
                threshold,
                min_cluster_size,
            )
            for chunk_ids in tqdm(c)
        )

    # Combine output
    new_clusters = {}
    for out_clusters in out:
        for idx, cluster in out_clusters.items():
            new_clusters[idx] = unique_txs(cluster + new_clusters.get(idx, []))

    # Add ids from old cluster to new cluster
    for cluster_idx, cluster in new_clusters.items():
        community_extended = []
        for (idx, similarity) in cluster:
            community_extended += [(idx, similarity)] + clusters.get(idx, [])
        new_clusters[cluster_idx] = unique_txs(community_extended)

    new_clusters = reorder_and_filter_clusters(
        new_clusters, embeddings, threshold, parallel
    )  # filter to keep only the relevant
    new_clusters = sort_clusters(new_clusters)

    clustered_ids = set()
    for idx, cluster_ids in new_clusters.items():
        filtered = set(cluster_ids) - clustered_ids
        cluster_ids = [
            cluster_idx for cluster_idx in cluster_ids if cluster_idx in filtered
        ]
        new_clusters[idx] = cluster_ids
        clustered_ids |= set(cluster_ids)

    new_clusters = filter_clusters(new_clusters, min_cluster_size)
    new_clusters = sort_clusters(new_clusters)
    return new_clusters


def fast_clustering(ids, embeddings, threshold=0.70, min_cluster_size=10):
    """
    Function for Fast Clustering

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    """

    # Compute cosine similarity scores
    cos_scores = cos_sim(embeddings, embeddings)
    print(cos_scores.shape)

    # Step 1) Create clusters where similarity is bigger than threshold
    bigger_than_threshold = cos_scores >= threshold
    indices = bigger_than_threshold.nonzero()

    cos_scores = cos_scores.numpy()

    extracted_clusters = defaultdict(lambda: [])
    for row, col in indices.tolist():
        extracted_clusters[ids[row]].append((ids[col], cos_scores[row, col]))

    extracted_clusters = sort_clusters(extracted_clusters)  # FIXME

    # Step 2) Remove overlapping clusters
    unique_clusters = {}
    extracted_ids = set()

    for cluster_id, cluster in extracted_clusters.items():
        add_cluster = True
        for transaction in cluster:
            if transaction[0] in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_clusters[cluster_id] = cluster
            for transaction in cluster:
                extracted_ids.add(transaction[0])

    new_clusters = {}
    for cluster_id, cluster in unique_clusters.items():
        community_extended = []
        for idx in cluster:
            community_extended.append(idx)
        new_clusters[cluster_id] = unique_txs(community_extended)

    new_clusters = filter_clusters(new_clusters, min_cluster_size)

    return new_clusters


def enrich_assignments(data, clusters, text_key):
    cluster_id_map = {}
    cluster_similarity_score_map = {}

    for i, cluster_ids in clusters.items():
        for j, txs in enumerate(cluster_ids):
            txs_id, similarity_score = txs
            cluster_id_map[txs_id] = i
            cluster_similarity_score_map[txs_id] = similarity_score

    cluster_ids = [
        cluster_id_map.get(idx) if idx in cluster_id_map else None for idx in data.index
    ]
    data["cluster_id"] = cluster_ids
    data["cluster_size"] = [
        len(clusters.get(cluster_id, [cluster_id])) for cluster_id in cluster_ids
    ]
    data["cluster_similarity_score"] = [
        cluster_similarity_score_map.get(idx)
        if idx in cluster_similarity_score_map
        else None
        for idx in data.index
    ]

    data = data.sort_values(
        ["cluster_size", "cluster_id", "cluster_similarity_score", text_key],
        ascending=[False, False, False, True],
    )

    return data


def cluster(
    data,
    model_name="all-MiniLM-L6-v2",
    cores=4,
    batch_size=32,
    iterations=20,
    min_cluster_size=5,
    threshold=0.75,
    text_key="description",
):
    ids = list(data.index)

    with print_durations("embedding data"):
        embeddings = embed_data(
            data, text_key, model_name, cores=cores, batch_size=batch_size
        )
    print("Data embedded")

    clusters = {}
    with print_durations("online_community_detection"):
        for i in range(iterations):
            print("Iteration", i)
            clusters = online_community_detection(
                ids,
                embeddings,
                clusters,
                min_cluster_size=min_cluster_size,
                threshold=threshold,
                iterations=1,
            )
            if len(get_unclustured_ids(ids, clusters)) == 0:
                break
    data = enrich_assignments(data, clusters, text_key)
    return clusters, data


@click.command()
@click.option("--path", help="The path to the input file.")
@click.option(
    "--model-name", default="all-MiniLM-L6-v2", help="The encoder model name."
)
@click.option("--cores", default=1, help="Number of CPU cores to use.")
@click.option("--batch_size", default=32)
@click.option("--iterations", default=20, help="Number of iterations of clustering.")
@click.option("--min-cluster-size", default=5, help="Minimum cluster size.")
@click.option(
    "--threshold", default=0.75, help="Minimum cosine similarity within cluster."
)
@click.option("--text_key", default="description")
@click.option("--id_key", default="unique_transaction_id")
def cluster_command(path, id_key, **kwargs):
    data = pd.read_csv(path)
    data = data.set_index(id_key)
    clusters, data = cluster(data, **kwargs)
    root_path = path.replace(".csv", "")
    data.to_csv(f"{root_path}_clustered.csv")


def cluster_txt(path, **kwargs):
    text = load_txt(path)
    data = pd.DataFrame(text, columns=["description"]).drop_duplicates()
    clusters, data = cluster(data, **kwargs)
    root_path = path.replace(".txt", "")
    data.to_csv(f"{root_path}_clustered.csv")


if __name__ == "__main__":
    cluster_command()
