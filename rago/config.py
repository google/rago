import dataclasses

# Data classes of retrieval / inference policies


@dataclasses.dataclass(frozen=True)
class RetrievalPolicy:
    # Whether to invoke retrieval at all
    run_retrieval: bool = False

    # Retrieval pattern can be either "once" or "interval" (every n tokens)
    retrieval_pattern: str | None = None
    retrieval_interval: int | None = None

    # Rewriter model
    run_rewrite: bool = False
    rewrite_seq_len_inference_prefill: int | None = None
    rewrite_seq_len_prefill_with_template: int | None = (
        None  # the prefill is the rewrite instruction template + question
    )
    rewrite_dec_steps: int | None = None

    # Reranker model
    run_rerank: bool = False
    rerank_seq_len_inference_prefill: int | None = None
    rerank_topk: int | None = None


@dataclasses.dataclass(frozen=True)
class EncodeDBPolicy:
    # Whether to invoke encode db at all
    run_encode_db: bool = False
    # Chunk length is the seq_len_inference_prefill for the encoder model
    encode_seq_len_inference_prefill: int | None = None
    # number of vectors in the database for encode db
    encode_db_nvec: int | None = None


class PhysicalMapping:
    def __init__(
        self,
        num_retrieval_servers: int = 0,
        num_chips: dict[str, int] = {
            "rewrite_prefill": 0,
            "rewrite_decode": 0,
            "encode": 0,
            "rerank": 0,
            "prefill": 0,
            "decode": 0,
            "e2e": 0,
        },
        placement_policy: str = "disaggregated",  # 'collocated' or 'disaggregated'
        collocation_strategy: (
            list[list[str]] | None
        ) = None,  # [['retrieval'], ['prefill'], ['decode']]
    ):
        self.num_retrieval_servers = num_retrieval_servers
        self.num_chips = num_chips
        self.placement_policy = placement_policy
        self.collocation_strategy = collocation_strategy

    # Print each of the attribute
    def __str__(self):
        return (
            f"PhysicalMapping(num_retrieval_servers={self.num_retrieval_servers},"
            + f" num_chips={self.num_chips}, placement_policy={self.placement_policy},"
            + f" collocation_strategy={self.collocation_strategy})"
        )
