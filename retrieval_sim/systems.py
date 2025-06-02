from typing import Optional

class System:

    def __init__(self, config_data, num_servers : int =1):
        self.config_data = config_data
        self.num_servers = num_servers

    def _get_compute_config(self):
        return self.config_data["cpu"]

    def _get_llc_config(self):
        return self.config_data["llc"]

    def _get_memory_config(self):
        return self.config_data["ddr"]

    def _get_storage_config(self):
        return self.config_data["storage"]

    @property
    def cpu_num_cores(self) -> int:
        cpu_config = self._get_compute_config()
        return cpu_config["num_cores"]

    @property
    def cpu_per_core_pq_throughput(self) -> float:
        cpu_config = self._get_compute_config()
        return cpu_config["per_core_pq_throughput"]

    @property
    def cpu_per_core_rerank_throughput(self) -> float:
        cpu_config = self._get_compute_config()
        return cpu_config["per_core_rerank_throughput"]

    @property
    def llc_bandwidth(self) -> float:
        llc_config = self._get_llc_config()
        return llc_config["bandwidth"] * self.num_servers

    @property
    def llc_capacity(self) -> float:
        llc_config = self._get_llc_config()
        return llc_config["capacity"] * self.num_servers

    @property
    def ddr_bandwidth(self) -> float:
        ddr_config = self._get_memory_config()
        return ddr_config["bandwidth"] * self.num_servers

    @property
    def ddr_capacity(self) -> float:
        ddr_config = self._get_memory_config()
        return ddr_config["capacity"] * self.num_servers

    @property
    def storage_bandwidth(self) -> float:
        storage_config = self._get_storage_config()
        return storage_config["bandwidth"] * self.num_servers

    @property
    def storage_capacity(self) -> float:
        storage_config = self._get_storage_config()
        return storage_config["capacity"] * self.num_servers

    def get_effective_cpu_per_core_pq_throughput(
        self,
        override_throughput_multiplier: Optional[float] = None,
    ) -> float:
        return self.cpu_per_core_pq_throughput * (
            override_throughput_multiplier if override_throughput_multiplier else 1.0
        )

    def get_effective_cpu_per_core_rerank_throughput(
        self,
        override_throughput_multiplier: Optional[float] = None,
    ) -> float:
        return self.cpu_per_core_rerank_throughput * (
            override_throughput_multiplier if override_throughput_multiplier else 1.0
        )

    def get_effective_llc_bandwidth(
        self,
        override_throughput_multiplier: Optional[float] = None,
    ) -> float:
        """Returns effective LLC bandwidth of system."""
        return self.llc_bandwidth * (
            override_throughput_multiplier if override_throughput_multiplier else 1.0
        )

    def get_effective_ddr_bandwidth(
        self,
        override_throughput_multiplier: Optional[float] = None,
    ) -> float:
        """Returns effective LLC bandwidth of system."""
        return self.ddr_bandwidth * (
            override_throughput_multiplier if override_throughput_multiplier else 1.0
        )

    def get_effective_storage_bandwidth(
        self,
        override_throughput_multiplier: Optional[float] = None,
    ) -> float:
        """Returns effective storage bandwidth of system."""
        return self.storage_bandwidth * (
            override_throughput_multiplier if override_throughput_multiplier else 1.0
        )

if __name__ == "__main__":

    # Example usage
    config_data = {
        "cpu": {
            "num_cores": 32,
            "per_core_pq_throughput": 20 * 1e9,
            "per_core_rerank_throughput": 10 * 1e9,
        },
        "llc": {
            "bandwidth": 1e12,
            "capacity": 64e6,
        },
        "memory": {
            "bandwidth": 400e9,
            "capacity": 512e9,
        },
        "storage": {
            "bandwidth": 10e9,
            "capacity": 2e12,
        },
    }

    system = System(config_data)
    print("CPU Num Cores:", system.cpu_num_cores)
    print("CPU Per Core PQ Throughput:", system.cpu_per_core_pq_throughput)
    print("CPU Per Core Rerank Throughput:", system.cpu_per_core_rerank_throughput)
    print("LLC Bandwidth:", system.llc_bandwidth)
    print("LLC Capacity:", system.llc_capacity)
    print("DDR Bandwidth:", system.ddr_bandwidth)
    print("DDR Capacity:", system.ddr_capacity)
    print("Storage Bandwidth:", system.storage_bandwidth)
    print("Storage Capacity:", system.storage_capacity)
    