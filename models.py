# The database parameters are similar to the RETRO paper: [go/retro-paper-pdf],
# although the index setting is likely different (not documented in the paper).
# The LLM itself uses Llama as a proxy, instead of the original RETRO model,
# as RETRO uses special attention mechanisms.
retro_db_scam = rag_params.RAGParams(
    retrieval_policy='interval',
    retrieval_interval=64,
    retrieval_at_this_step=True,
    topk=8,
    retrieval_chunk_length=128,
    search_lib='scam',
    # Whether to use multiple servers -> the latency will be divided by
    # the number of servers
    num_retrieval_servers=1,
    scam_params=rag_params.ScamParams(
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
        memory_hierarchy_pq_vec_root='ddr',
        memory_hierarchy_pq_vec_branch='ddr',
        memory_hierarchy_pq_vec_leaf='ddr',
        memory_hierarchy_rerank_vec_root='ddr',
        memory_hierarchy_rerank_vec_branch='ddr',
        memory_hierarchy_rerank_vec_leaf='ddr',
        # Number of partitions to scan at each stage of the search process
        n_partitions_to_scan_root=1,
        n_partitions_to_scan_branch=int(0.01 * 1e6),
        n_partitions_to_scan_leaf=int(0.001 * 250e6),
        # Number of vectors to rerank at each stage of the search process
        # the reranking is optional, set to None to disable reranking
        n_vecs_to_rerank_root=int(0.01 * 1e6),
        n_vecs_to_rerank_branch=int(0.001 * 250e6),
        n_vecs_to_rerank_leaf=100,
        parallel_mode_pq_scan_root='inter_query',
        parallel_mode_pq_scan_branch='inter_query',
        parallel_mode_pq_scan_leaf='inter_query',
        parallel_mode_rerank_root='inter_query',
        parallel_mode_rerank_branch='inter_query',
        parallel_mode_rerank_leaf='inter_query',
        # Whether to cap the data movement per operator for inter-query parallel
        # mode. The cap is the total amount of data size of the level.
        allow_data_reuse_pq_scan_root=False,
        allow_data_reuse_pq_scan_branch=False,
        allow_data_reuse_pq_scan_leaf=False,
        allow_data_reuse_rerank_root=False,
        allow_data_reuse_rerank_branch=False,
        allow_data_reuse_rerank_leaf=False,
    ),
)

# For the small database, assume around 1M tokens chunked into 100-token chunks.
# The 10K vectors can then be scanned in brute-force manner.
rag_vanilla_db_small_scam = rag_params.RAGParams(
    retrieval_policy='once',
    retrieval_at_this_step=True,
    topk=10,
    retrieval_chunk_length=128,
    search_lib='scam',
    scam_params=rag_params.ScamParams(
        db_nvec=int(1e4),
        db_dim=768,
        # partitioning parameters
        n_partitions_total_root=0,
        n_partitions_total_branch=0,
        n_partitions_total_leaf=1,
        # Quantization parameters for each level
        bytes_per_pq_vec_leaf=int(768 / 4),
        bytes_per_rerank_vec_leaf=768,
        # Memory hierarchy parameters for each level
        memory_hierarchy_pq_vec_leaf='ddr',
        memory_hierarchy_rerank_vec_leaf='storage',
        # Number of partitions to scan at each stage of the search
        n_partitions_to_scan_root=0,
        n_partitions_to_scan_branch=0,
        n_partitions_to_scan_leaf=1,
        # Number of vectors to rerank at each stage of the search
        # the reranking is optional, set to None to disable reranking
        n_vecs_to_rerank_leaf=100,
        parallel_mode_pq_scan_leaf='inter_query',
        parallel_mode_rerank_leaf='inter_query',
    ),
)

# Model: rag_vanilla_v1_inference_decode, retrieval-augmented model (RAG)
# A RAG model reuses llama2's parameters, using a retrieval database that is
# similar to the RETRO paper. Vanilla RAG only retrieves once at the beginning
# of the decoding process.
rag_vanilla_db_large_scam = retro_db_scam.replace(
    retrieval_policy='once',
    retrieval_interval=None,
    retrieval_at_this_step=True,
)

# For the small database, assume around 1M tokens chunked into 100-token chunks.
# The 10K vectors can then be scanned in brute-force manner.
rag_vanilla_db_small_scam = rag_params.RAGParams(
    retrieval_policy='once',
    retrieval_at_this_step=True,
    topk=10,
    retrieval_chunk_length=128,
    search_lib='scam',
    scam_params=rag_params.ScamParams(
        db_nvec=int(1e4),
        db_dim=768,
        # partitioning parameters
        n_partitions_total_root=0,
        n_partitions_total_branch=0,
        n_partitions_total_leaf=1,
        # Quantization parameters for each level
        bytes_per_pq_vec_leaf=int(768 / 4),
        bytes_per_rerank_vec_leaf=768,
        # Memory hierarchy parameters for each level
        memory_hierarchy_pq_vec_leaf='ddr',
        memory_hierarchy_rerank_vec_leaf='storage',
        # Number of partitions to scan at each stage of the search
        n_partitions_to_scan_root=0,
        n_partitions_to_scan_branch=0,
        n_partitions_to_scan_leaf=1,
        # Number of vectors to rerank at each stage of the search
        # the reranking is optional, set to None to disable reranking
        n_vecs_to_rerank_leaf=100,
        parallel_mode_pq_scan_leaf='inter_query',
        parallel_mode_rerank_leaf='inter_query',
    ),
)

@dataclasses.dataclass(frozen=True)
class ScamParams:
  """Index and search parameters for ScaM.

  ScaM supports one-level (brute-force), two-level, and three-level tree search.
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
  parallel_mode_pq_scan_root: str = 'inter_query'
  parallel_mode_pq_scan_branch: str = 'inter_query'
  parallel_mode_pq_scan_leaf: str = 'inter_query'
  # In general the amount of data to be reranked is smaller than the amount of
  # data to be scanned, so we default to inter_query parallelism for reranking.
  parallel_mode_rerank_root: str = 'inter_query'
  parallel_mode_rerank_branch: str = 'inter_query'
  parallel_mode_rerank_leaf: str = 'inter_query'

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

  def replace(self, **changes) -> ScamParams:
    return dataclasses.replace(self, **changes)


@dataclasses.dataclass(frozen=True)
class RAGParams:
  """RAG Parameters, with a focus on the database."""

  # retrieval policy can be either "once" or "interval", for later, we need to
  # specify the interval (retrieved once every N steps). One should then set
  # retrieval_at_this_step to True to enable retrieval at the current step.
  # Note that RAG does not necessarily retrieve content at each decoding step.,
  # thus one must replace the retrieval_at_this_step with False for the
  # non-retrieval steps when analyzing end-to-end generation.
  retrieval_policy: str | None = None
  retrieval_interval: int | None = None
  retrieval_at_this_step: bool | None = None

  # The number of top-k nearest neighbors to return from the search
  topk: int | None = None
  # The length of each retrieval chunk.
  retrieval_chunk_length: int | None = None

  # The search library to use, either "scam" or "faiss"
  # Both uses product Quantization (PQ), a compression technique that reduces
  # the dimensionality of database vectors.
  search_lib: str | None = None
  faiss_params: FaissParams | None = None
  scam_params: ScamParams | None = None

  # Whether to use multiple servers -> the latency will be divided by
  # the number of servers
  num_retrieval_servers: int = 1

  @property
  def db_nvec(self) -> int | None:
    if self.search_lib == 'scam' and self.scam_params:
      return self.scam_params.db_nvec
    elif self.search_lib == 'faiss' and self.faiss_params:
      return self.faiss_params.db_nvec
    else:
      return None

  @property
  def db_dim(self) -> int | None:
    if self.search_lib == 'scam' and self.scam_params:
      return self.scam_params.db_dim
    elif self.search_lib == 'faiss' and self.faiss_params:
      return self.faiss_params.db_dim
    else:
      return None
