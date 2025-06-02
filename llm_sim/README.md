# LLM Simulation

We loop in the [GenZ LLM simulator](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer). Note that this is **not the LLM simulator we used in the RAGO paper**, as Google's production LLM simulator is not open-sourced yet. So the performance number generated here is slightly different to the RAGO paper. 

## Estimate LLM serving performance

The example scripts can be executed by:

```
cd genz_scripts
python llm_perf.py
```

To configure the models and hardware to use, edit [llm_perf.py](llm_perf.py).

The generated results are saved in [genz_scripts/perf_results](genz_scripts/perf_results). For example, the performance of the main LLM can be found in [genz_scripts/perf_results/main_llm_perf.csv](genz_scripts/perf_results/main_llm_perf.csv).