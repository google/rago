# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving

RAGO is a tool for modeling and evaluating the performance and efficiency of retrieval-augmented generation (RAG) serving. This repository is an open-source implementation for the ISCA'25 paper [RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving](https://arxiv.org/pdf/2503.14649). 

RAG is a genAI paradigm that combines generative LLMs and retrievals:

![RAG Example](figures/RAG_example.png)

RAGO is a system performance optimization framework. It searches for the optimal system configurations (task placement, resource allocation, and batching policy) based on the specific RAG algorithms and underlying hardware:  

![RAGO Overview](figures/paper_workflow.png)

RAGO takes two sets of performance files as inputs: (a) LLM inference performance results and (b) retrieval performance results. Both LLM and retrieval performance can be either (a) profiled on real machines or (b) modeled by simulators such as [Generative LLM Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer). It then evaluates the end-to-end performance Pareto frontier (e.g., time-to-first-token latency, throughput, etc.) by assembling inference and retrieval. 

## Getting Started

### Install

```
conda create --name rago python=3.11
conda activate rago

pip install -r requirements.txt
```

### Example RAG performance analysis 

We provided some example RAG pipelines. The description of these RAG workloads can be found in [examples/README.md](examples/README.md).

To run some examples of RAG pipelines:

```
cd examples
python example_case1.py
```

The output performance figures can be found in [examples/img](examples/img).

### Inference Performance

Here we use [GenZ](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer) as an example simulator to produce some performance results based on different hardware and models. You can find a more detailed description in [llm_sim/README.md](llm_sim/README.md).

To use them: 

```
# Initialization
git submodule update --init --recursive
cd llm_sim/genz
git checkout cb2448332a1a83eec52cd6e3b7919d56eaff380c
pip install -r requirements.txt

# Run the analysis
cd ../genz_scripts
python llm_perf.py
```

The results can be found in [llm_sim/genz_scripts/perf_results](llm_sim/genz_scripts/perf_results).

Any other simulators or real profiles can be looped in, as long as they produce the same performance csv format. 

### Retrieval Performance

The retrieval performance model and its usage is decribed in [retrieval_sim/README.md](retrieval_sim/README.md). The performance model is based on the [ScaNN](https://github.com/google-research/google-research/tree/master/scann) vector search library.

```
cd retrieval_sim
python retrieval_perf.py 
```

## 🙏 Acknowledgements

We extend our gratitude towards Cliff Young, David Culler, and Eugene Le for reviewing the paper and providing insightful feedback.
We also thank the extended team at Google DeepMind and System Research@Google who enabled and supported this research direction.

## 📄 License

Code in this Github repository is licensed under a [APACHE 2.0 License](./LICENSE).

## 🎓 Citing RAGO

```
@inproceedings{rago:isca:2025,
  title={RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving},
  author={Jiang, Wenqi and Subramanian, Suvinay and Graves, Cat and Alonso, Gustavo and Yazdanbakhsh, Amir and Dadu, Vidushi},
  booktitle = {Proceedings of the 52th Annual International Symposium on Computer Architecture}
  year={2025}
}
```

In addition to RAGO, there are some related works about improving RAG serving performance, specifically:

* [KDD'25] [PipeRAG: Fast retrieval-augmented generation via adaptive pipeline parallelism](https://www.amazon.science/publications/piperag-fast-retrieval-augmented-generation-via-adaptive-pipeline-parallelism)

PipeRAG addresses performance optimization for RAG with iterative retrieval by algorithm- and system-level improvements.

* [VLDB'25] [Chameleon: A Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models](https://arxiv.org/pdf/2310.09949)

Chameleon is a heterogeneous accelerator system for RAG serving. It prototypes FPGA-based accelerators for retrieval and runs LLM inference on GPUs.

* [SC'23] [Co-design Hardware and Algorithm for Vector Search](https://arxiv.org/pdf/2306.11182)

FANNS accelerates product-quantization-based vector search.

* [VLDB'25] [Fast Graph Vector Search via Hardware Acceleration and Delayed-Synchronization Traversal](https://arxiv.org/abs/2406.12385)

Falcon accelerates graph-based vector search.


*This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*
