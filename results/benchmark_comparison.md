## AutoMR / MRI / Ours benchmark comparison

Subject programs: Apache Commons Math 2.2 (8 numerical functions)
Source: Automatic Discovery of Accurate and Succinct Metamorphic Relationships (https://github.com/bolzzzz/AutoMR)

### Aggregate

| Metric | MRI (baseline) | AutoMR (upper bound) | Ours (LLM-guided PSO) |
|---|---|---|---|
| Mutants killed | 94/625 | 374/625 | 142/625 |
| Kill rate | 15.04% | 59.84% | 22.7% |
| False detections | 0 | 0 | 0 |
| Runtime per MR (s) | 3.9 | 4.7 | 1.9 |
| PSO runs x iters | 500x350 | 500x350 | 8x157 (50 part.) |

### Per-function

| Function | AutoMR killed/faults | AutoMR kill % | Ours kill %* | Ours #MR |
|---|---|---|---|---|
| abs | 10/14 | 71.4 | 50.0 | 13 |
| asinh | 70/297 | 23.6 | 1.6 | 1 |
| atan | 96/188 | 51.1 | 16.1 | 1 |
| cos | 16/38 | 42.1 | 25.0 | 8 |
| log1p | 92/230 | 40.0 | 27.4 | 3 |
| log10 | 40/116 | 34.5 | 41.2 | 2 |
| sin | 18/34 | 52.9 | 50.8 | 6 |
| tan | 32/36 | 88.9 | 54.2 | 2 |

\* Ours kills source-level mutants of the Commons Math 2.2 port, built the same way as AutoMR's mutants but not the identical Java set; AutoMR's per-function denominators are its own.
- The 625 mutants aren't in the AutoMR repo. They are mutated Apache Commons Math 2.2 Java source files, made by applying mutation operators to the source, and came from the MRI paper [16] (Zhang et al.). The repo ships only the inferred-MR results (.npz/.pkl) and the algorithm.
- Paper inconsistency, kept verbatim: per-program seeded_faults in Table V sum to 953, but the headline (RQ2, Fig. 5, Table VI) says 625 total. Both are left unchanged here.
- The paper gives MRI kills only in aggregate (94), not per function, so MRI appears only in the aggregate block.
