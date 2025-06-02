import dataclasses

from typing import List, Dict, Any, Optional

from retrieval_params import ScannParams
from systems import System
from operators import PQScanOp, RerankOp


class RooflineCost:
    """Class to record costs for the retrieval."""

    def __init__(self):
        self.time_memory_llc = 0
        self.time_memory_ddr = 0
        self.time_memory_storage = 0
        self.time_compute = 0
        self.roofline = 0


class OperatorCost:

    def time_cost_pq_scan(
        self,
        op: PQScanOp,
        sim_sys: System,
        results: RooflineCost,
    ) -> RooflineCost:
        """Compute time costs for the PQ Scan operator."""

        compute_pq_bytes = op.compute_pq_bytes()
        data_movement_pq_bytes = op.data_movement_pq_bytes()
        cpu_per_core_pq_throughput = sim_sys.get_effective_cpu_per_core_pq_throughput()
        results = self.time_cost_scan(
            sim_sys=sim_sys,
            compute_bytes=compute_pq_bytes,
            data_movement_bytes=data_movement_pq_bytes,
            cpu_per_core_throughput=cpu_per_core_pq_throughput,
            retrieval_batch_size=op.retrieval_batch_size,
            parallel_mode=op.parallel_mode,
            results=results,
        )
        return results

    def time_cost_rerank(
        self,
        op: RerankOp,
        sim_sys: System,
        results: RooflineCost,
    ) -> RooflineCost:
        """Compute time costs for the reranking operator."""

        compute_rerank_bytes = op.compute_rerank_bytes()
        data_movement_rerank_bytes = op.data_movement_rerank_bytes()
        cpu_per_core_rerank_throughput = (
            sim_sys.get_effective_cpu_per_core_rerank_throughput()
        )
        results = self.time_cost_scan(
            sim_sys=sim_sys,
            compute_bytes=compute_rerank_bytes,
            data_movement_bytes=data_movement_rerank_bytes,
            cpu_per_core_throughput=cpu_per_core_rerank_throughput,
            retrieval_batch_size=op.retrieval_batch_size,
            parallel_mode=op.parallel_mode,
            results=results,
        )
        return results

    def time_cost_scan(
        self,
        sim_sys: System,
        compute_bytes: dict[str, float],
        data_movement_bytes: dict[str, float],
        cpu_per_core_throughput: float,
        retrieval_batch_size: int,
        parallel_mode: str,
        results: RooflineCost,
    ) -> RooflineCost:
        """Compute time costs for the general data scan operator."""

        # Computation: Flops, bytes transferred for the actual computation.
        memory_hierarchy = self.get_memory_hierarchy(data_movement_bytes)
        if memory_hierarchy is None:
            return results

        # Calculate the time consumption and roofline.
        # As commented above, we assume the stages are executed sequentially.
        # So here we need to do roofline of only one stage, and then sum up the cost
        # in a higher level function.
        cpu_num_cores = sim_sys.cpu_num_cores
        bandwidth = 0.0
        if memory_hierarchy == "llc":
            bandwidth = sim_sys.get_effective_llc_bandwidth()
        elif memory_hierarchy == "ddr":
            bandwidth = sim_sys.get_effective_ddr_bandwidth()
        elif memory_hierarchy == "storage":
            bandwidth = sim_sys.get_effective_storage_bandwidth()

        if parallel_mode == "inter_query":
            results = self.time_cost_scan_inter_query_parallel(
                retrieval_batch_size=retrieval_batch_size,
                total_compute_bytes=compute_bytes,
                total_data_movement_bytes=data_movement_bytes,
                cpu_num_cores=cpu_num_cores,
                cpu_per_core_throughput=cpu_per_core_throughput,
                memory_hierarchy=memory_hierarchy,
                bandwidth=bandwidth,
                results=results,
            )
        elif parallel_mode == "intra_query":
            self.time_cost_scan_intra_query_parallel(
                total_compute_bytes=compute_bytes,
                total_data_movement_bytes=data_movement_bytes,
                cpu_num_cores=cpu_num_cores,
                cpu_per_core_throughput=cpu_per_core_throughput,
                memory_hierarchy=memory_hierarchy,
                bandwidth=bandwidth,
                results=results,
            )
        else:
            raise NotImplementedError()

        return results

    def get_memory_hierarchy(self, data_movement_bytes: dict[str, float]) -> str | None:
        """Return the memory hierarchy of the operator.

        Args:
            data_movement_bytes: a dict of data movement bytes in different memory
                hierarchies.

        Note that the ScaNN search process is sequential, i.e., root PQ -> root
        reranking -> branch PQ -> branch reranking -> leaf PQ -> leaf reranking. So
        we need to do roofline per stage, and then sum up the cost. Thus this
        function only accept one stage with none-zero workload, and the operator is
        only in at most one memory hierarchy.
        """
        memory_hierarchy_list = ["llc", "ddr", "storage"]
        memory_hierarchy = None
        num_none_zero_workloads = 0
        for key in memory_hierarchy_list:
            if data_movement_bytes[key] > 0:
                num_none_zero_workloads += 1
                memory_hierarchy = key
        assert (
            num_none_zero_workloads == 1 or num_none_zero_workloads == 0
        ), "Only one memory hierarchy is allowed for CPU"
        return memory_hierarchy

    def time_cost_scan_inter_query_parallel(
        self,
        retrieval_batch_size: int,
        total_compute_bytes: dict[str, float],
        total_data_movement_bytes: dict[str, float],
        cpu_num_cores: int,
        cpu_per_core_throughput: float,
        memory_hierarchy: str,
        bandwidth: float,
        results: RooflineCost,
    ) -> RooflineCost:
        """Inter-query parallel scan: each query is processed by one core."""

        # Calculate the time consumption on data movement (llc / ddr / storage)
        data_movement_bytes = total_data_movement_bytes[memory_hierarchy]
        time_memory_llc = 0.0
        time_memory_ddr = 0.0
        time_memory_storage = 0.0
        data_movement_time = data_movement_bytes / bandwidth
        if memory_hierarchy == "llc":
            time_memory_llc = data_movement_time
            results.time_memory_llc += time_memory_llc
        elif memory_hierarchy == "ddr":
            time_memory_ddr = data_movement_time
            results.time_memory_ddr += time_memory_ddr
        elif memory_hierarchy == "storage":
            time_memory_storage = data_movement_time
            results.time_memory_storage += time_memory_storage

        # Calculate the time consumption on compute.
        # If each query is processed by a thread, then small batches cannot use
        # all the cores. For large batches, we can use all the cores and assume
        # the OS can schedule the thread management without additional overhead.
        compute_bytes = total_compute_bytes[memory_hierarchy]
        if retrieval_batch_size < cpu_num_cores:
            num_queries_per_core = 1
        else:
            num_queries_per_core = retrieval_batch_size / cpu_num_cores
        compute_bytes_per_core = (
            compute_bytes * num_queries_per_core / retrieval_batch_size
        )
        compute_time_cpu = compute_bytes_per_core / cpu_per_core_throughput
        time_compute = compute_time_cpu
        results.time_compute += time_compute

        # update results and get roofline per round
        # print(
        #     ""
        #     f"time compute: {time_compute}, time memory llc: {time_memory_llc}, "
        #     f"time memory ddr: {time_memory_ddr}, time memory storage: {time_memory_storage}"
        # )
        results.roofline += max(
            time_compute,
            time_memory_llc,
            time_memory_ddr,
            time_memory_storage,
        )

        return results

    def time_cost_scan_intra_query_parallel(
        self,
        total_compute_bytes: dict[str, float],
        total_data_movement_bytes: dict[str, float],
        cpu_num_cores: int,
        cpu_per_core_throughput: float,
        memory_hierarchy: str,
        bandwidth: float,
        results: RooflineCost,
    ) -> RooflineCost:
        """Intra-query parallel PQ scan: each query is processed by all cores."""

        # For compute, the compute time of the batch on all cores is equivalent
        # to the compute time of one query on one core, assuming the same amount
        # of data is scanned per query
        # For memory or storage, the memory time of the batch is the sum of the
        # memory/storage time of all queries, as each accesses different data.
        time_memory_llc = 0.0
        time_memory_ddr = 0.0
        time_memory_storage = 0.0
        data_movement_bytes = total_data_movement_bytes[memory_hierarchy]
        data_movement_time = data_movement_bytes / bandwidth
        if memory_hierarchy == "llc":
            time_memory_llc = data_movement_time
            results.time_memory_llc += time_memory_llc
        elif memory_hierarchy == "ddr":
            time_memory_ddr = data_movement_time
            results.time_memory_ddr += time_memory_ddr
        elif memory_hierarchy == "storage":
            time_memory_storage = data_movement_time
            results.time_memory_storage += time_memory_storage

        def time_cpu_sync(cpu_num_cores: int) -> float:
            """The synchronization overhead for multi-core CPU."""
            # Assuming a simple model of linear cost to the number of cores.
            per_core_overhead = 1e-6  # 1 us
            return per_core_overhead * cpu_num_cores

        compute_bytes = total_compute_bytes[memory_hierarchy]
        compute_time_pq = compute_bytes / (cpu_per_core_throughput * cpu_num_cores)
        compute_time_pq += time_cpu_sync(cpu_num_cores)
        time_compute_this_round = compute_time_pq
        results.time_compute += time_compute_this_round

        results.roofline += max(
            time_compute_this_round,
            time_memory_llc,
            time_memory_ddr,
            time_memory_storage,
        )

        return results


class RetrievalCost:
    """End-to-end retrieval cost"""

    def __init__(
        self,
        scann_params: ScannParams,
        sim_sys: System,
        retrieval_batch_size: int,
    ):
        self.scann_params = scann_params
        self.sim_sys = sim_sys
        self.retrieval_batch_size = retrieval_batch_size

    def get_operators_from_retrieval_params(
        self,
    ) -> List[Any]:
        """
        Given ScannParams, return a list of operators to be used for retrieval.
        """

        scann_params = self.scann_params
        retrieval_batch_size = self.retrieval_batch_size

        operators = []

        # Root Level
        if scann_params.n_partitions_total_root > 0:
            operators.append(
                PQScanOp(
                    n_partitions_total=scann_params.n_partitions_total_root,
                    n_partitions_to_scan=scann_params.n_partitions_to_scan_root,
                    n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_root,
                    bytes_per_pq_vec=scann_params.bytes_per_pq_vec_root,
                    memory_hierarchy_pq_vec=scann_params.memory_hierarchy_pq_vec_root,
                    retrieval_batch_size=retrieval_batch_size,
                    parallel_mode=scann_params.parallel_mode_pq_scan_root,
                    allow_data_reuse_pq_scan=scann_params.allow_data_reuse_pq_scan_root,
                )
            )
            operators.append(
                RerankOp(
                    n_partitions_total=scann_params.n_partitions_total_root,
                    n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_root,
                    bytes_per_rerank_vec=scann_params.bytes_per_rerank_vec_root,
                    memory_hierarchy_rerank_vec=scann_params.memory_hierarchy_rerank_vec_root,
                    n_vecs_to_rerank=scann_params.n_vecs_to_rerank_root,
                    retrieval_batch_size=retrieval_batch_size,
                    parallel_mode=scann_params.parallel_mode_rerank_root,
                    allow_data_reuse_pq_scan=scann_params.allow_data_reuse_rerank_root,
                )
            )
        # Branch Level
        if scann_params.n_partitions_total_branch > 0:
            operators.append(
                PQScanOp(
                    n_partitions_total=scann_params.n_partitions_total_branch,
                    n_partitions_to_scan=scann_params.n_partitions_to_scan_branch,
                    n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_branch,
                    bytes_per_pq_vec=scann_params.bytes_per_pq_vec_branch,
                    memory_hierarchy_pq_vec=scann_params.memory_hierarchy_pq_vec_branch,
                    retrieval_batch_size=retrieval_batch_size,
                    parallel_mode=scann_params.parallel_mode_pq_scan_branch,
                    allow_data_reuse_pq_scan=scann_params.allow_data_reuse_pq_scan_branch,
                )
            )
            operators.append(
                RerankOp(
                    n_partitions_total=scann_params.n_partitions_total_branch,
                    n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_branch,
                    bytes_per_rerank_vec=scann_params.bytes_per_rerank_vec_branch,
                    n_vecs_to_rerank=scann_params.n_vecs_to_rerank_branch,
                    memory_hierarchy_rerank_vec=scann_params.memory_hierarchy_rerank_vec_branch,
                    retrieval_batch_size=retrieval_batch_size,
                    parallel_mode=scann_params.parallel_mode_rerank_branch,
                    allow_data_reuse_pq_scan=scann_params.allow_data_reuse_rerank_branch,
                )
            )
        # Leaf Level
        operators.append(
            PQScanOp(
                n_partitions_total=scann_params.n_partitions_total_leaf,
                n_partitions_to_scan=scann_params.n_partitions_to_scan_leaf,
                n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_leaf,
                bytes_per_pq_vec=scann_params.bytes_per_pq_vec_leaf,
                memory_hierarchy_pq_vec=scann_params.memory_hierarchy_pq_vec_leaf,
                retrieval_batch_size=retrieval_batch_size,
                parallel_mode=scann_params.parallel_mode_pq_scan_leaf,
                allow_data_reuse_pq_scan=scann_params.allow_data_reuse_pq_scan_leaf,
            )
        )
        operators.append(
            RerankOp(
                n_partitions_total=scann_params.n_partitions_total_leaf,
                n_avg_vec_per_partition=scann_params.n_avg_vec_per_partition_leaf,
                bytes_per_rerank_vec=scann_params.bytes_per_rerank_vec_leaf,
                n_vecs_to_rerank=scann_params.n_vecs_to_rerank_leaf,
                memory_hierarchy_rerank_vec=scann_params.memory_hierarchy_rerank_vec_leaf,
                retrieval_batch_size=retrieval_batch_size,
                parallel_mode=scann_params.parallel_mode_rerank_leaf,
                allow_data_reuse_pq_scan=scann_params.allow_data_reuse_rerank_leaf,
            )
        )

        return operators

    def get_total_cost(
        self,
    ) -> RooflineCost:
        """Compute the total cost for the retrieval."""

        operators = self.get_operators_from_retrieval_params()

        # Initialize the cost results
        results = RooflineCost()

        # Compute the costs for each operator
        for op in operators:
            if isinstance(op, PQScanOp):
                results = OperatorCost().time_cost_pq_scan(
                    op=op,
                    sim_sys=self.sim_sys,
                    results=results,
                )
            elif isinstance(op, RerankOp):
                results = OperatorCost().time_cost_rerank(
                    op=op,
                    sim_sys=self.sim_sys,
                    results=results,
                )
            else:
                raise NotImplementedError()

        return results
