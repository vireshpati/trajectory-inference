(WIP) This repository includes cross-comparison performance testing for the trajectory inference methods put forward by the papers below on their respective datasets.

For up-to-date documents on overleaf : https://github.com/vireshpati/AI4TS-2025-Spring 

Relevant scripts are:
- for MFL native: mfl/fig1_batch/run_local.sh
- for MFL eot benchmark: mfl/fig1_batch/eot_mfl_benchmark.py
- for schrodinger plug-in native: SinkhornBridges/examples/Guschin_benchmark.ipynb
- for schrodinger plug-in on multi-time dataset: implementation TBD. need to ensure marginals are aligned between steps.

Link EOT Bench: https://github.com/ngushchin/EntropicOTBenchmark

## References ##

Aram-Alexandre Pooladian and Jonathan Niles-Weed. ”Plug-in estimation of
Schrodinger bridges” arXiv.2024.

https://github.com/APooladian/SinkhornBridges


Chizat, Lénaïc, et al. "Trajectory inference via mean-field langevin in path space." Advances in Neural Information Processing Systems 35 (2022): 16731-16742.

https://github.com/zsteve/mfl

