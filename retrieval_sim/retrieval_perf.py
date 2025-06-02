import json
import pandas as pd

from cost import RetrievalCost
from retrieval_params import ScannParams
from systems import System

# Config 1: Large-scale retrieval with 64 billion vectors
retro_db_scam = ScannParams(
    db_nvec=64 * int(1e9),
    db_dim=768,
    # partitioning parameters
    n_partitions_total_root=1,
    n_partitions_total_branch=int(1e6),
    n_partitions_total_leaf=int(250e6),
    # Quantization parameters for each level
    bytes_per_pq_vec_root=int(768 / 8),
    bytes_per_pq_vec_branch=int(768 / 8),
    bytes_per_pq_vec_leaf=int(768 / 4),
    bytes_per_rerank_vec_root=768,
    bytes_per_rerank_vec_branch=768,
    bytes_per_rerank_vec_leaf=768,
    # Memory hierarchy parameters for each level
    memory_hierarchy_pq_vec_root="ddr",
    memory_hierarchy_pq_vec_branch="ddr",
    memory_hierarchy_pq_vec_leaf="ddr",
    memory_hierarchy_rerank_vec_root="ddr",
    memory_hierarchy_rerank_vec_branch="ddr",
    memory_hierarchy_rerank_vec_leaf="ddr",
    # Number of partitions to scan at each stage of the search process
    n_partitions_to_scan_root=1,
    n_partitions_to_scan_branch=int(0.01 * 1e6),
    n_partitions_to_scan_leaf=int(0.001 * 250e6),
    # Number of vectors to rerank at each stage of the search process
    # the reranking is optional, set to None to disable reranking
    n_vecs_to_rerank_root=int(0.01 * 1e6),
    n_vecs_to_rerank_branch=int(0.001 * 250e6),
    n_vecs_to_rerank_leaf=100,
    parallel_mode_pq_scan_root="inter_query",
    parallel_mode_pq_scan_branch="inter_query",
    parallel_mode_pq_scan_leaf="inter_query",
    parallel_mode_rerank_root="inter_query",
    parallel_mode_rerank_branch="inter_query",
    parallel_mode_rerank_leaf="inter_query",
    # Whether to cap the data movement per operator for inter-query parallel
    # mode. The cap is the total amount of data size of the level.
    allow_data_reuse_pq_scan_root=False,
    allow_data_reuse_pq_scan_branch=False,
    allow_data_reuse_pq_scan_leaf=False,
    allow_data_reuse_rerank_root=False,
    allow_data_reuse_rerank_branch=False,
    allow_data_reuse_rerank_leaf=False,
)


# Config 2: Small-scale retrieval with 100K vectors
# For the small database, assume around 1M tokens chunked into 100-token chunks.
# The 100K vectors can then be scanned in brute-force manner.
rag_vanilla_db_small_scam = ScannParams(
    db_nvec=int(1e5),
    db_dim=768,
    # partitioning parameters
    n_partitions_total_root=0,
    n_partitions_total_branch=0,
    n_partitions_total_leaf=1,
    # Here, assuming 2 bytes per dim
    bytes_per_pq_vec_leaf=int(768 * 2),
    bytes_per_rerank_vec_leaf=768 * 2,
    # Memory hierarchy parameters for each level
    memory_hierarchy_pq_vec_leaf="ddr",
    memory_hierarchy_rerank_vec_leaf="ddr",
    # Number of partitions to scan at each stage of the search
    n_partitions_to_scan_root=0,
    n_partitions_to_scan_branch=0,
    n_partitions_to_scan_leaf=1,
    # Number of vectors to rerank at each stage of the search
    # the reranking is optional, set to None to disable reranking
    n_vecs_to_rerank_leaf=100,
    parallel_mode_pq_scan_leaf="inter_query",
    parallel_mode_rerank_leaf="inter_query",
)


# import systems/cpu.json
with open("systems/cpu.json", "r") as f:
    config_data = json.load(f)
    print(config_data)
    sim_sys = System(config_data)

results = []

# Large-scale retrieval, assuming 16 servers to provide enough memory capacity
# Assuming balanced workload across servers
db_name = 'large_64b'
hardware_name = 'cpu_dram'
batch_sizes = [int(2**i) for i in range(0, 8 + 1)]
num_servers_list = [16, 32]

print("========== Large DB ===========")
for num_servers in num_servers_list:
    for batch_size in batch_sizes:
        
        retrieval_cost_analyzer = RetrievalCost(
            scann_params=retro_db_scam,
            sim_sys=sim_sys,
            retrieval_batch_size=batch_size,
        )
        retrieval_cost = retrieval_cost_analyzer.get_total_cost()
        latency_ms = retrieval_cost.roofline / num_servers * 1000
        print(f"{num_servers} servers, batch size: {batch_size}, latency: {latency_ms:.2f} ms")

        results.append({
            "db.name": db_name,
            "hardware.name": hardware_name,
            "hardware.num_servers": num_servers,
            "batch_size": batch_size,
            "time_ms": latency_ms,
        })


# Small-scale retrieval, only one server needed
db_name = 'small_100k'
hardware_name = 'cpu_dram'
batch_sizes = [int(2**i) for i in range(0, 8 + 1)]
num_servers = 1

print("========== Small DB ===========")
for batch_size in batch_sizes:
    
    retrieval_cost_analyzer = RetrievalCost(
        scann_params=rag_vanilla_db_small_scam,
        sim_sys=sim_sys,
        retrieval_batch_size=batch_size,
    )
    retrieval_cost = retrieval_cost_analyzer.get_total_cost()
    latency_ms = retrieval_cost.roofline * 1000
    print(f"1 server, batch size: {batch_size}, latency: {latency_ms:.2f} ms")

    results.append({
        "db.name": db_name,
        "hardware.name": hardware_name,
        "hardware.num_servers": num_servers,
        "batch_size": batch_size,
        "time_ms": latency_ms,
    })


# Create DataFrame
df = pd.DataFrame(results, columns=[
    "db.name", "hardware.name", "hardware.num_servers", "batch_size", "time_ms"
])

# Save to CSV
df.to_csv("perf_results/retrieval_perf.csv", index=False)