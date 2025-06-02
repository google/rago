# Retrieval Performance

This folder implements a performance model for the [ScaNN](https://github.com/google-research/google-research/tree/master/scann) vector search library.

The inputs of the modeling can be break down to (a) the CPU system used to serve the workload, and (b) the database configuration (e.g., sizes, dimensionality, search scope, etc.). The systems configuration can be set in [systems/cpu.json](systems/cpu.json) while the retrieval configuration is specified in [retrieval_perf.py](retrieval_perf.py).

To run the performance modeling:

```
python retrieval_perf.py 
```

The outputs can be found in [perf_results/retrieval_perf.csv](perf_results/retrieval_perf.csv).