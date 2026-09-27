# CPU task-count latency benchmark

Measured locally on an Apple M1 Max (macOS ARM64, 10 logical CPUs, 32 GiB RAM). Both models ran on CPU in FP32, one model process at a time. Two/four refers to PyTorch inference threads, not an enforced container CPU quota.

The workload preserves the evaluated policy: GLiNER runs its exact frozen five-head positive schema and, separately, five actual negative heads. Balanced Laya runs one task-count question and its actual opposition. These are task-count timings, not all eight default questions.

A deterministic hash-selected subset contains eight paired scenarios / 16 English–French messages, balanced by routine/suspicious framing and multiple/single-task labels. Each message ran three times (48 timings per setting); two warmups per language were discarded. Message state sizes are 170–424 characters when serialized with ASCII escapes. Timing includes schema preparation/tokenization and inference; it excludes loading, report serialization, HTTP, queuing and pod overhead.

| Policy | Threads | Positive median | Positive p95 | Including opposition median |
| --- | ---: | ---: | ---: | ---: |
| GLiNER binary policy | 2 | 1019 ms | 1127 ms | 2010 ms |
| GLiNER binary policy | 4 | 761 ms | 845 ms | 1514 ms |
| Balanced Laya | 2 | 143 ms | 182 ms | 271 ms |
| Balanced Laya | 4 | 134 ms | 167 ms | 262 ms |

Four threads reduced GLiNER median latency by about 25%, while Laya improved by about 6%. At four threads, the evaluated GLiNER policy took approximately 5.7 times as long as Laya. No policy decisions changed across CPU/GPU or repetitions on these 16 messages; this is a stability check, not a fresh full accuracy evaluation.

| Policy / threads | English median | French median |
| --- | ---: | ---: | ---: |
| gliner / 2 | 986 ms | 1077 ms |
| gliner / 4 | 737 ms | 806 ms |
| laya / 2 | 163 ms | 73 ms |
| laya / 4 | 157 ms | 69 ms |

At four threads, observed whole-process peak resident memory was 4.38 GB for GLiNER and 4.51 GB for Laya with both language checkpoints loaded. These include runtime and temporary allocations and are not model-only sizes or established pod memory limits. Construction plus checkpoint/package validation took about 9.8 seconds for GLiNER and 5.9 seconds for Laya; Python process startup is excluded.

## Repeat on the OpenShift CPU

Use the evaluated package versions and locally cached checkpoints, then run the two commands sequentially in the intended pod. Keep thread settings aligned with the test configuration and record the actual CPU model, CPU request/limit, architecture and worker count. The harness validates checkpoint identity and schemas, permits relocated caches and records the PyTorch build, actual CPU FP32 parameters and process memory.

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-gliner/bin/python -m system_one_models.action_cpu_benchmark \
  --backend gliner --threads 4 --repeats 3 --output results/cpu-gliner.json

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m system_one_models.action_cpu_benchmark \
  --backend laya --threads 4 --repeats 3 --output results/cpu-laya.json
```

The tracked `schemas/laya-cpu-benchmark-lock-v1.json` supplies the Laya checkpoint/schema lock; `schemas/task-count-gliner-binary-v2.json` supplies GLiNER. No downloads are performed in offline mode. Set `--threads 2` and the environment variables to 2 to repeat that setting. Dataset and policy paths can be supplied explicitly with `--dataset`, `--policy` and `--laya-selection`.

These Mac ARM results do not establish latency on Linux x86 servers, quota-constrained pods or concurrent production traffic. Run this benchmark on the target CPU, followed by service-level concurrency and representative message-length tests, before assigning an SLA or capacity.

Raw timings and answers: `results/action-cpu-{gliner,laya}-{2,4}threads-v1.json`. Combined summary: `results/action-cpu-comparison-v1.json`. The first GLiNER two-thread run preceded provenance-only harness additions; its inference schema and timing loop match subsequent runs, but peak RSS was not recorded. Subsequent runs used the saved harness in `results/action-cpu-benchmark-harness-v1.py`.
