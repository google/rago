 @property
 def vpu_flops(self) -> float:
   return self._get_compute_config().vpu.flops


 @property
 def cpu_per_core_flops(self) -> float:
   return self._get_compute_config().cpu.per_core_flops


 @property
 def cpu_num_cores(self) -> int:
   return self._get_compute_config().cpu.num_cores


 @property
 def cpu_per_core_pq_throughput(self) -> float:
   return self._get_compute_config().cpu.per_core_pq_throughput


 @property
 def cpu_per_core_rerank_throughput(self) -> float:
   return self._get_compute_config().cpu.per_core_rerank_throughput


 @property
 def external_memory_bandwidth(self) -> float | None:
   return self._get_memory_config().emem.bandwidth or None

 @property
 def wmem_write_bandwidth(self) -> float:
   return self._get_memory_config().wmem.write_bandwidth


 @property
 def llc_bandwidth(self) -> float:
   return self._get_memory_config().llc.bandwidth


 @property
 def ddr_bandwidth(self) -> float:
   return self._get_memory_config().ddr.bandwidth


 @property
 def storage_bandwidth(self) -> float:
   return self._get_storage_config().bandwidth


 @property

   if override_flops_multiplier is not None:
     return self.vpu_flops * override_flops_multiplier
   else:
     return self.vpu_flops * self.vpu_flops_hw_multiplier


 def get_effective_cpu_per_core_flops(
     self,
     override_flops_multiplier: Optional[float] = None,
 ) -> float:
   return self.cpu_per_core_flops * (
       override_flops_multiplier if override_flops_multiplier else 1.0
   )


 def get_effective_cpu_per_core_pq_throughput(
     self,
     override_throughput_multiplier: Optional[float] = None,
 ) -> float:
   return self.cpu_per_core_pq_throughput * (
       override_throughput_multiplier
       if override_throughput_multiplier
       else 1.0
   )


 def get_effective_cpu_per_core_rerank_throughput(
     self,
     override_throughput_multiplier: Optional[float] = None,
 ) -> float:
   return self.cpu_per_core_rerank_throughput * (
       override_throughput_multiplier
       if override_throughput_multiplier
       else 1.0
   )


 def get_effective_memory_bandwidth(
     self,
     override_bandwidth_multiplier: Optional[float] = None,
     bandwidth_from_lookup_table: bool = False,
     average_dma_size_in_bytes: Optional[float] = None,

         self.wmem_write_bandwidth
         if self.wmem_write_bandwidth > 0
         else float('inf')
     )


 def get_effective_ddr_bandwidth(self) -> float:
   """Returns effective LLC bandwidth of system."""
   return self.ddr_bandwidth


 def get_effective_llc_bandwidth(self) -> float:
   """Returns effective LLC bandwidth of system."""
   return self.llc_bandwidth

 optional GPU gpu = 5;


 // CPU message includes parameters that are specific to CPU systems for
 // modeling their compute performance in detail.
 message CPU {






   // Number of physical cores per CPU server
   optional int32 num_cores = 1;
   // Per-physcial-core CPU performance in PQ operations per second.
   optional double per_core_pq_throughput = 2;
   // PPer-physcial-core CPU performance in rerank operations per second.
   optional double per_core_rerank_throughput = 3;
   // Per-physcial-core CPU performance in floating-point operations per second
   optional double per_core_flops = 4;
 }
 optional CPU cpu = 6;


 // The number of logical chiplets in a system package. This is number of
 // compute devices per package that have their own dedicated memory and can

   optional double hw_bandwidth_multiplier = 2;
   // Per-chip EMEM capacity in bytes.
   optional double capacity = 3;
 }
 optional ExternalMemory emem = 3;


 // WeightsMemory represents the parameters of the Static RAM that can hold
 // weights. WMEM is faster than HBM and slower than VMEM.
 message WeightsMemory {
   // Per-chip WMEM read bandwidth in bytes per second.
   optional double read_bandwidth = 1;
   // Per-chip WMEM write bandwidth in bytes per second.
   optional double write_bandwidth = 2;
   // Per-chip WMEM capacity in bytes.
   optional double capacity = 3;
   // Per-chip WMEM reserved capacity in bytes.
   optional double reserve_capacity = 4;
 }
 optional WeightsMemory wmem = 4;


 // CPU's last level cache (LLC)
 message CPULLCMemory {
   // Per-server LLC bandwidth in bytes per second.
   optional double bandwidth = 1;
   // Per-server LLC capacity in bytes.
   optional double capacity = 2;
 }
 optional CPULLCMemory llc = 5;


 // CPU's DRAM
 message CPUDDRMemory {
   // Per-server DDR bandwidth in bytes per second.
   optional double bandwidth = 1;
   // Per-server DDR capacity in bytes.
   optional double capacity = 2;
 }
 optional CPUDDRMemory ddr = 6;

