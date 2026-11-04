# Agent B paired comparison matrix

Produced by `python scripts\compare_matrix.py --hands 2000 --out runs\b_matrix.md`
on the six committed-format checkpoints (git-ignored, reproducible from the
training commands in the README). 2,000 shared deals per cell, base seed 10000.

Modes: `history-vs-control` (memory vs capacity-matched control, same initial
weights), `agent-a-vs-b` (frozen Agent A vs B with history), `ablation`
(same B weights, history on vs off). Positive always means the *first* policy
out-earned the second. The `mean` row is the unweighted mean across seeds with
its standard deviation and a t-based 95 % half-width (df = 2).

"resolved" means the per-seed paired 95 % interval excludes zero -- it is a
statement about that cell, not about the seed population.

"random" is a uniform-random legal action; it was never a training or evaluation
opponent and is included only as an out-of-distribution probe.
Paired comparison matrix (bb/hand; positive = first minus second)

## history-vs-control

| opponent | seed | first - second | 95 % | resolved |
|---|---|---|---|---|
| passive | 0 | +0.0845 | +/-0.1086 | no |
| passive | 1 | +0.0103 | +/-0.1124 | no |
| passive | 2 | -0.0740 | +/-0.1034 | no |
| passive | **mean** | **+0.0069** | sd 0.0793, +/-0.1970 | 2/3 seeds same sign |
| raiser_strong | 0 | +0.1020 | +/-0.1435 | no |
| raiser_strong | 1 | -0.0483 | +/-0.1793 | no |
| raiser_strong | 2 | -0.0275 | +/-0.1342 | no |
| raiser_strong | **mean** | **+0.0087** | sd 0.0814, +/-0.2023 | 1/3 seeds same sign |
| raiser_loose | 0 | +0.1105 | +/-0.1854 | no |
| raiser_loose | 1 | +0.0617 | +/-0.2032 | no |
| raiser_loose | 2 | -0.1125 | +/-0.1715 | no |
| raiser_loose | **mean** | **+0.0199** | sd 0.1172, +/-0.2912 | 2/3 seeds same sign |
| random | 0 | -0.2895 | +/-0.1536 | yes |
| random | 1 | -0.0177 | +/-0.1497 | no |
| random | 2 | -0.2805 | +/-0.1462 | yes |
| random | **mean** | **-0.1959** | sd 0.1544, +/-0.3835 | 3/3 seeds same sign |

## agent-a-vs-b

| opponent | seed | first - second | 95 % | resolved |
|---|---|---|---|---|
| passive | 0 | -0.1505 | +/-0.1084 | yes |
| passive | 1 | +0.0030 | +/-0.0854 | no |
| passive | 2 | +0.0660 | +/-0.1056 | no |
| passive | **mean** | **-0.0272** | sd 0.1114, +/-0.2766 | 1/3 seeds same sign |
| raiser_strong | 0 | -0.0970 | +/-0.1122 | no |
| raiser_strong | 1 | +0.0015 | +/-0.0876 | no |
| raiser_strong | 2 | +0.0965 | +/-0.1155 | no |
| raiser_strong | **mean** | **+0.0003** | sd 0.0968, +/-0.2404 | 2/3 seeds same sign |
| raiser_loose | 0 | -0.1555 | +/-0.1495 | yes |
| raiser_loose | 1 | -0.0675 | +/-0.1292 | no |
| raiser_loose | 2 | -0.0050 | +/-0.1537 | no |
| raiser_loose | **mean** | **-0.0760** | sd 0.0756, +/-0.1878 | 3/3 seeds same sign |
| random | 0 | +0.1675 | +/-0.1143 | yes |
| random | 1 | -0.2185 | +/-0.1184 | yes |
| random | 2 | -0.2660 | +/-0.1319 | yes |
| random | **mean** | **-0.1057** | sd 0.2378, +/-0.5906 | 2/3 seeds same sign |

## ablation

| opponent | seed | first - second | 95 % | resolved |
|---|---|---|---|---|
| passive | 0 | +0.0050 | +/-0.0238 | no |
| passive | 1 | -0.0130 | +/-0.0159 | no |
| passive | 2 | -0.0080 | +/-0.0092 | no |
| passive | **mean** | **-0.0053** | sd 0.0093, +/-0.0231 | 2/3 seeds same sign |
| raiser_strong | 0 | +0.0120 | +/-0.0253 | no |
| raiser_strong | 1 | -0.0250 | +/-0.0214 | yes |
| raiser_strong | 2 | -0.0095 | +/-0.0154 | no |
| raiser_strong | **mean** | **-0.0075** | sd 0.0186, +/-0.0462 | 2/3 seeds same sign |
| raiser_loose | 0 | +0.0135 | +/-0.0452 | no |
| raiser_loose | 1 | +0.0110 | +/-0.0311 | no |
| raiser_loose | 2 | -0.0060 | +/-0.0275 | no |
| raiser_loose | **mean** | **+0.0062** | sd 0.0106, +/-0.0264 | 2/3 seeds same sign |
| random | 0 | +0.0360 | +/-0.0370 | no |
| random | 1 | -0.0145 | +/-0.0327 | no |
| random | 2 | -0.0095 | +/-0.0339 | no |
| random | **mean** | **+0.0040** | sd 0.0278, +/-0.0691 | 1/3 seeds same sign |

