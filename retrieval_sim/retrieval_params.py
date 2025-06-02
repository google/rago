import dataclasses


@dataclasses.dataclass(frozen=True)
class ScannParams:
    """Index and search parameters for ScaNN.

    ScaNN supports one-level (brute-force), two-level, and three-level tree search.
    For two-level tree search, leave all branch-level parameters as None.
    For brute-force search, leave all branch-level and leaf-level parameters as
    None.
    """

    # The number of vectors in the database
    db_nvec: int | None = None
    # The number of dimensions in each vector
    db_dim: int | None = None

    # Total number of partitions in the database
    # root has only one partition containing all the branch centroids
    # (three-level) or all leaf centroids (two-level)
    n_partitions_total_root: int = 1
    n_partitions_total_branch: int | None = None
    n_partitions_total_leaf: int | None = None

    # Quantization parameters for each level
    bytes_per_pq_vec_root: int | None = None
    bytes_per_pq_vec_branch: int | None = None
    bytes_per_pq_vec_leaf: int | None = None
    bytes_per_rerank_vec_root: int | None = None
    bytes_per_rerank_vec_branch: int | None = None
    bytes_per_rerank_vec_leaf: int | None = None

    # Memory hierarchy parameters for each level
    memory_hierarchy_pq_vec_root: str | None = None
    memory_hierarchy_pq_vec_branch: str | None = None
    memory_hierarchy_pq_vec_leaf: str | None = None
    memory_hierarchy_rerank_vec_root: str | None = None
    memory_hierarchy_rerank_vec_branch: str | None = None
    memory_hierarchy_rerank_vec_leaf: str | None = None

    # Number of partitions to scan at each stage of the search process
    n_partitions_to_scan_root: int = 1
    n_partitions_to_scan_branch: int | None = None
    n_partitions_to_scan_leaf: int | None = None

    # Number of vectors to rerank at each stage of the search process
    # the reranking is optional, set to None to disable reranking
    n_vecs_to_rerank_root: int | None = None
    n_vecs_to_rerank_branch: int | None = None
    n_vecs_to_rerank_leaf: int | None = None

    # Parallel search mode, either one thread per query (inter_query parallelism)
    # or using all threads to search a single query (intra_query parallelism)
    # For PQ scan, the mode can be either inter_query (for larger batch sizes) or
    # intra_query (for smaller batch sizes)
    parallel_mode_pq_scan_root: str = "inter_query"
    parallel_mode_pq_scan_branch: str = "inter_query"
    parallel_mode_pq_scan_leaf: str = "inter_query"
    # In general the amount of data to be reranked is smaller than the amount of
    # data to be scanned, so we default to inter_query parallelism for reranking.
    parallel_mode_rerank_root: str = "inter_query"
    parallel_mode_rerank_branch: str = "inter_query"
    parallel_mode_rerank_leaf: str = "inter_query"

    # Whether to cap the data movement per operator: if True, the data movement
    # is capped to the total amount of data size of the level. For example, all
    # queries scan the same set of root vectors, so the data movement can be
    # capped to the total number of root vectors.
    allow_data_reuse_pq_scan_root: bool = False
    allow_data_reuse_pq_scan_branch: bool = False
    allow_data_reuse_pq_scan_leaf: bool = False
    allow_data_reuse_rerank_root: bool = False
    allow_data_reuse_rerank_branch: bool = False
    allow_data_reuse_rerank_leaf: bool = False

    @property
    def n_avg_vec_per_partition_root(self) -> float:
        if self.n_partitions_total_branch:
            return self.n_partitions_total_branch
        elif self.n_partitions_total_leaf:
            return self.n_partitions_total_leaf
        else:
            return 0.0

    @property
    def n_avg_vec_per_partition_branch(self) -> float:
        if not self.n_partitions_total_branch:
            return 0.0
        return self.n_partitions_total_leaf / self.n_partitions_total_branch

    @property
    def n_avg_vec_per_partition_leaf(self) -> float:
        return self.db_nvec / self.n_partitions_total_leaf

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
