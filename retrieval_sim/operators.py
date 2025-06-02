class PQScanOp:
    """Class to compute time costs for the PQ Scan operator."""

    def __init__(
        self,
        n_partitions_total: float,
        n_partitions_to_scan: float,
        n_avg_vec_per_partition: float,
        bytes_per_pq_vec: float,
        memory_hierarchy_pq_vec: str,
        retrieval_batch_size: float,
        parallel_mode: str,
        allow_data_reuse_pq_scan: bool = False,
    ):
        self.n_partitions_total = n_partitions_total
        self.n_partitions_to_scan = n_partitions_to_scan
        self.n_avg_vec_per_partition = n_avg_vec_per_partition
        self.bytes_per_pq_vec = bytes_per_pq_vec
        self.memory_hierarchy_pq_vec = memory_hierarchy_pq_vec
        self.retrieval_batch_size = retrieval_batch_size
        self.parallel_mode = parallel_mode
        self.allow_data_reuse_pq_scan = allow_data_reuse_pq_scan

    def data_movement_pq_bytes(self):
        if self.allow_data_reuse_pq_scan:
            data_movement_pq_bytes = min(
                self.retrieval_batch_size
                * self.n_avg_vec_per_partition
                * self.n_partitions_to_scan
                * self.bytes_per_pq_vec,
                self.n_partitions_total
                * self.n_avg_vec_per_partition
                * self.bytes_per_pq_vec,
            )
        else:
            data_movement_pq_bytes = (
                self.retrieval_batch_size
                * self.n_avg_vec_per_partition
                * self.n_partitions_to_scan
                * self.bytes_per_pq_vec
            )
        data_movement_pq_bytes_dict = {"llc": 0, "ddr": 0, "storage": 0}
        data_movement_pq_bytes_dict[self.memory_hierarchy_pq_vec] = (
            data_movement_pq_bytes
        )
        return data_movement_pq_bytes_dict

    def compute_pq_bytes(self):
        compute_pq_bytes = (
            self.retrieval_batch_size
            * self.n_avg_vec_per_partition
            * self.n_partitions_to_scan
            * self.bytes_per_pq_vec
        )
        compute_rerank_bytes_dict = {"llc": 0, "ddr": 0, "storage": 0}
        compute_rerank_bytes_dict[self.memory_hierarchy_pq_vec] = compute_pq_bytes
        return compute_rerank_bytes_dict

    def retrieval_batch_size(self):
        return self.retrieval_batch_size

    def parallel_mode(self):
        return self.parallel_mode


class RerankOp:
    """Class to compute time costs for the reranking operator."""

    def __init__(
        self,
        n_partitions_total: float,
        n_avg_vec_per_partition: float,
        bytes_per_rerank_vec,
        n_vecs_to_rerank,
        memory_hierarchy_rerank_vec: str,
        retrieval_batch_size: float,
        parallel_mode: str,
        allow_data_reuse_pq_scan: bool = False,
    ):
        self.n_partitions_total = n_partitions_total
        self.n_avg_vec_per_partition = n_avg_vec_per_partition
        self.bytes_per_rerank_vec = bytes_per_rerank_vec
        self.n_vecs_to_rerank = n_vecs_to_rerank
        self.memory_hierarchy_rerank_vec = memory_hierarchy_rerank_vec
        self.retrieval_batch_size = retrieval_batch_size
        self.parallel_mode = parallel_mode
        self.allow_data_reuse_pq_scan = allow_data_reuse_pq_scan

    def data_movement_rerank_bytes(self):
        if self.allow_data_reuse_pq_scan:
            data_movement_rerank_bytes = min(
                self.retrieval_batch_size
                * self.n_vecs_to_rerank
                * self.bytes_per_rerank_vec,
                self.n_partitions_total
                * self.n_avg_vec_per_partition
                * self.bytes_per_rerank_vec,
            )
        else:
            data_movement_rerank_bytes = (
                self.retrieval_batch_size
                * self.n_vecs_to_rerank
                * self.bytes_per_rerank_vec
            )
        data_movement_rerank_bytes_dict = {"llc": 0, "ddr": 0, "storage": 0}
        data_movement_rerank_bytes_dict[self.memory_hierarchy_rerank_vec] = (
            data_movement_rerank_bytes
        )
        return data_movement_rerank_bytes_dict

    def compute_rerank_bytes(self):
        compute_rerank_bytes = (
            self.retrieval_batch_size
            * self.n_vecs_to_rerank
            * self.bytes_per_rerank_vec
        )
        compute_rerank_bytes_dict = {"llc": 0, "ddr": 0, "storage": 0}
        compute_rerank_bytes_dict[self.memory_hierarchy_rerank_vec] = (
            compute_rerank_bytes
        )
        return compute_rerank_bytes_dict

    def retrieval_batch_size(self):
        return self.retrieval_batch_size

    def parallel_mode(self):
        return self.parallel_mode
