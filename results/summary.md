# Results summary

Made by `scripts/summarize_eval.py`. Mean over seeds per arm, range over seeds in brackets. Rollouts use the baselines' fixed subsample of at most 1,000 cubes per set; the optimal-move rate uses every cube with known optimal moves. d1_4_all is in the training pool by design.

## Primary: step-100,000 checkpoints

Checkpoint steps: {"true_distance (20 stages)": {"0": 100000, "1": 100000, "2": 100000, "3": 100000, "4": 100000, "5": 100000}, "kociemba (20 stages)": {"0": 100000, "1": 100000, "2": 100000, "3": 100000, "4": 100000, "5": 100000}, "true_distance (2 stages)": {"0": 100000, "1": 100000, "2": 100000, "3": 100000, "4": 100000, "5": 100000}, "kociemba (2 stages)": {"0": 100000, "1": 100000, "2": 100000, "3": 100000, "4": 100000, "5": 100000}, "mixed": {"0": 100000, "1": 100000, "2": 100000, "3": 100000, "4": 100000, "5": 100000}}

### Key comparisons

Difference of means over seeds; 95% Welch t-interval with each seed as one observation; 'Diff/SE' is the difference divided by its standard error; 'Excludes 0' marks intervals that do not contain zero. The same test cubes are used for every run.

| Comparison | Measure | Difference | 95% interval | Diff/SE | Excludes 0 | Seeds |
|---|---|---|---|---|---|---|
| mixed minus true_distance (20 stages) | greedy solve rate, d10 | +34.4 pts | +32.2 pts to +36.7 pts | 35.0 | yes | 6/6 |
| mixed minus true_distance (20 stages) | greedy solve rate, d13 | +47.5 pts | +45.2 pts to +49.8 pts | 45.9 | yes | 6/6 |
| mixed minus true_distance (20 stages) | greedy solve rate, random | +45.3 pts | +42.3 pts to +48.2 pts | 34.8 | yes | 6/6 |
| mixed minus true_distance (20 stages) | optimal-move rate, d10 | -4.5 pts | -5.8 pts to -3.2 pts | 7.7 | yes | 6/6 |
| mixed minus true_distance (20 stages) | optimal-move rate, d13 | -4.1 pts | -5.4 pts to -2.8 pts | 7.3 | yes | 6/6 |
| mixed minus true_distance (20 stages) | beam excess over exact, d13 | +2.86 | +2.74 to +2.99 | 54.2 | yes | 6/6 |
| mixed minus true_distance (20 stages) | beam mean length, random | +0.82 | +0.78 to +0.85 | 53.3 | yes | 6/6 |
| mixed minus true_distance (20 stages) | beam mean length, d20_known | +0.52 | +0.18 to +0.85 | 4.0 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, d10 | +31.6 pts | +29.2 pts to +33.9 pts | 30.6 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, d13 | +42.4 pts | +39.7 pts to +45.0 pts | 37.2 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, random | +41.8 pts | +38.6 pts to +44.9 pts | 29.7 | yes | 6/6 |
| mixed minus kociemba (20 stages) | optimal-move rate, d10 | -4.6 pts | -5.8 pts to -3.5 pts | 9.7 | yes | 6/6 |
| mixed minus kociemba (20 stages) | optimal-move rate, d13 | -3.6 pts | -5.2 pts to -2.0 pts | 5.1 | yes | 6/6 |
| mixed minus kociemba (20 stages) | beam excess over exact, d13 | +2.64 | +2.54 to +2.74 | 58.1 | yes | 6/6 |
| mixed minus kociemba (20 stages) | beam mean length, random | +0.76 | +0.72 to +0.79 | 46.6 | yes | 6/6 |
| mixed minus kociemba (20 stages) | beam mean length, d20_known | +0.52 | +0.21 to +0.82 | 4.4 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, d10 | +2.9 pts | +1.0 pts to +4.8 pts | 3.4 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, d13 | +5.1 pts | +2.2 pts to +8.0 pts | 4.0 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, random | +3.5 pts | +0.5 pts to +6.5 pts | 2.6 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | optimal-move rate, d10 | +0.1 pts | -0.9 pts to +1.1 pts | 0.3 | no | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | optimal-move rate, d13 | -0.5 pts | -1.9 pts to +0.9 pts | 0.8 | no | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam excess over exact, d13 | +0.23 | +0.10 to +0.35 | 4.0 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam mean length, random | +0.06 | +0.02 to +0.10 | 3.2 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam mean length, d20_known | +0.00 | -0.39 to +0.39 | 0.0 | no | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, d10 | +28.4 pts | +24.6 pts to +32.1 pts | 18.6 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, d13 | +39.2 pts | +35.1 pts to +43.2 pts | 22.8 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, random | +40.6 pts | +36.5 pts to +44.7 pts | 23.4 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | optimal-move rate, d10 | +0.6 pts | -1.0 pts to +2.2 pts | 0.9 | no | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | optimal-move rate, d13 | -1.2 pts | -2.2 pts to -0.3 pts | 3.0 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam excess over exact, d13 | +0.30 | +0.18 to +0.43 | 5.4 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam mean length, random | -0.46 | -0.51 to -0.41 | 21.8 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam mean length, d20_known | -0.42 | -0.82 to -0.03 | 2.4 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, d10 | +25.4 pts | +22.2 pts to +28.5 pts | 19.2 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, d13 | +34.3 pts | +31.1 pts to +37.4 pts | 24.3 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, random | +36.0 pts | +32.7 pts to +39.4 pts | 24.2 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | optimal-move rate, d10 | -0.7 pts | -1.4 pts to -0.0 pts | 2.3 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | optimal-move rate, d13 | -0.7 pts | -2.0 pts to +0.7 pts | 1.2 | no | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam excess over exact, d13 | +0.21 | +0.11 to +0.32 | 4.8 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam mean length, random | -0.45 | -0.49 to -0.41 | 25.0 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam mean length, d20_known | -0.58 | -0.88 to -0.27 | 4.5 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, d10 | -0.1 pts | -4.3 pts to +4.0 pts | 0.1 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, d13 | +0.2 pts | -4.0 pts to +4.4 pts | 0.1 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, random | -1.1 pts | -5.3 pts to +3.1 pts | 0.6 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | optimal-move rate, d10 | -1.2 pts | -2.8 pts to +0.3 pts | 2.0 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | optimal-move rate, d13 | +0.1 pts | -0.7 pts to +0.9 pts | 0.3 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam excess over exact, d13 | +0.14 | +0.04 to +0.24 | 3.2 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam mean length, random | +0.07 | +0.02 to +0.12 | 3.3 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam mean length, d20_known | -0.15 | -0.47 to +0.17 | 1.1 | no | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, d10 | +6.1 pts | +2.2 pts to +10.0 pts | 3.7 | yes | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, d13 | +8.3 pts | +4.4 pts to +12.3 pts | 5.2 | yes | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, random | +4.7 pts | +0.5 pts to +8.8 pts | 2.6 | yes | 6/6 |
| mixed minus true_distance (2 stages) | optimal-move rate, d10 | -5.1 pts | -6.8 pts to -3.5 pts | 7.2 | yes | 6/6 |
| mixed minus true_distance (2 stages) | optimal-move rate, d13 | -2.8 pts | -4.1 pts to -1.5 pts | 5.1 | yes | 6/6 |
| mixed minus true_distance (2 stages) | beam excess over exact, d13 | +2.56 | +2.46 to +2.66 | 57.8 | yes | 6/6 |
| mixed minus true_distance (2 stages) | beam mean length, random | +1.28 | +1.23 to +1.32 | 67.1 | yes | 6/6 |
| mixed minus true_distance (2 stages) | beam mean length, d20_known | +0.94 | +0.62 to +1.26 | 7.5 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, d10 | +6.2 pts | +3.0 pts to +9.5 pts | 4.4 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, d13 | +8.1 pts | +5.4 pts to +10.9 pts | 6.8 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, random | +5.7 pts | +2.4 pts to +9.0 pts | 3.9 | yes | 6/6 |
| mixed minus kociemba (2 stages) | optimal-move rate, d10 | -3.9 pts | -5.1 pts to -2.8 pts | 7.8 | yes | 6/6 |
| mixed minus kociemba (2 stages) | optimal-move rate, d13 | -2.9 pts | -4.2 pts to -1.7 pts | 5.8 | yes | 6/6 |
| mixed minus kociemba (2 stages) | beam excess over exact, d13 | +2.42 | +2.33 to +2.51 | 60.2 | yes | 6/6 |
| mixed minus kociemba (2 stages) | beam mean length, random | +1.21 | +1.18 to +1.24 | 82.3 | yes | 6/6 |
| mixed minus kociemba (2 stages) | beam mean length, d20_known | +1.09 | +0.94 to +1.25 | 16.6 | yes | 6/6 |

### Optimal-move rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba | baseline: shallow_search_d2 | baseline: random |
|---|---|---|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 91.4% | 31.9% | 6.2% |
| holdout_d5 | 100.0% | 100.0% (99.9%-100.0%) | 100.0% | 100.0% (99.9%-100.0%) | 99.9% | 90.4% | 30.3% | 6.1% |
| holdout_d6 | 99.0% | 98.6% (98.5%-98.6%) | 99.1% | 98.9% (98.8%-98.9%) | 98.8% (98.8%-98.9%) | 90.5% | 25.1% | 6.2% |
| d7 | 90.9% (90.5%-91.6%) | 90.0% (89.6%-90.3%) | 91.6% (91.0%-92.3%) | 90.7% (90.1%-91.0%) | 90.8% (90.5%-91.3%) | 90.4% | 21.3% | 5.6% |
| d8 | 72.5% (72.0%-73.4%) | 71.3% (69.7%-72.9%) | 72.2% (71.5%-72.9%) | 72.9% (71.4%-74.8%) | 71.1% (69.5%-72.1%) | 89.3% | 19.3% | 6.0% |
| d9 | 55.2% (54.2%-56.6%) | 53.7% (52.6%-55.0%) | 54.5% (52.7%-56.8%) | 53.9% (52.8%-54.7%) | 52.7% (51.8%-53.9%) | 88.1% | 15.6% | 6.7% |
| d10 | 39.5% (38.4%-41.1%) | 39.6% (38.9%-40.1%) | 40.2% (38.3%-41.6%) | 38.9% (38.0%-39.8%) | 35.0% (33.8%-36.2%) | 77.3% | 11.6% | 5.3% |
| d11 | 28.8% (26.4%-30.4%) | 28.5% (26.1%-30.5%) | 28.7% (28.1%-29.3%) | 27.8% (26.8%-28.5%) | 24.9% (22.5%-26.6%) | 56.3% | 9.1% | 6.6% |
| d12 | 25.8% (24.6%-26.9%) | 25.4% (24.5%-26.4%) | 25.2% (24.4%-26.2%) | 24.2% (23.0%-25.4%) | 21.7% (20.0%-23.2%) | 37.5% | 10.1% | 6.2% |
| d13 | 19.4% (18.3%-20.1%) | 18.9% (17.1%-20.6%) | 18.1% (17.0%-18.9%) | 18.2% (17.5%-18.8%) | 15.3% (13.3%-16.4%) | 26.7% | 9.4% | 5.4% |
| random | - | - | - | - | - | - | - | - |
| d20_known | - | - | - | - | - | - | - | - |

### Greedy solve rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: shallow_search_d2 |
|---|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 13.4% |
| holdout_d5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.4% |
| holdout_d6 | 99.9% (99.6%-100.0%) | 99.6% (99.3%-99.9%) | 99.8% (99.8%-99.9%) | 99.8% (99.7%-99.9%) | 99.9% (99.9%-100.0%) | 0.4% |
| d7 | 96.5% (96.1%-96.7%) | 96.6% (96.1%-97.1%) | 98.6% (98.3%-98.9%) | 98.2% (98.0%-98.9%) | 98.9% (98.5%-99.3%) | 0.0% |
| d8 | 84.0% (83.1%-84.7%) | 83.7% (83.0%-84.7%) | 91.6% (90.1%-93.4%) | 91.7% (89.7%-93.3%) | 93.8% (92.4%-94.7%) | 0.0% |
| d9 | 64.7% (62.8%-65.6%) | 65.8% (64.3%-67.4%) | 82.0% (79.4%-85.6%) | 83.5% (80.5%-84.9%) | 88.1% (87.0%-89.3%) | 0.0% |
| d10 | 48.1% (46.2%-49.7%) | 50.9% (49.2%-53.3%) | 76.4% (70.1%-79.7%) | 76.3% (72.6%-79.5%) | 82.5% (80.4%-85.3%) | 0.0% |
| d11 | 37.7% (35.0%-40.6%) | 41.8% (38.6%-44.3%) | 71.7% (69.1%-76.2%) | 72.3% (68.2%-74.5%) | 80.1% (78.3%-82.1%) | 0.0% |
| d12 | 33.4% (30.5%-35.9%) | 37.2% (35.3%-38.8%) | 71.2% (65.8%-74.2%) | 71.0% (65.3%-72.7%) | 79.2% (76.6%-82.4%) | 0.0% |
| d13 | 30.7% (28.9%-33.9%) | 35.8% (32.7%-38.4%) | 69.8% (66.3%-73.7%) | 70.1% (66.1%-73.0%) | 78.2% (75.6%-79.6%) | 0.0% |
| random | 29.7% (26.9%-33.0%) | 33.2% (29.9%-36.6%) | 70.3% (63.7%-73.4%) | 69.3% (65.0%-72.9%) | 75.0% (71.7%-77.5%) | 0.0% |
| d20_known | 27.3% (18.2%-36.4%) | 31.8% (18.2%-45.5%) | 62.1% (45.5%-72.7%) | 68.2% (54.5%-72.7%) | 74.2% (54.5%-90.9%) | 0.0% |

### Beam (4096) solve rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout_d5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout_d6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d7 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d11 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d13 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| random | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d20_known | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### Beam (4096) mean solution length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba |
|---|---|---|---|---|---|---|
| d1_4_all | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 | 4.77 |
| holdout_d5 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 6.29 |
| holdout_d6 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 7.67 |
| d7 | 7.00 | 7.00 | 7.00 | 7.00 | 7.00 | 9.15 |
| d8 | 8.00 (8.00-8.01) | 8.00 (8.00-8.02) | 8.01 (8.00-8.02) | 8.01 (8.00-8.02) | 8.03 (8.02-8.04) | 10.49 |
| d9 | 9.06 (9.04-9.07) | 9.04 (9.03-9.06) | 9.18 (9.14-9.20) | 9.11 (9.07-9.13) | 9.48 (9.39-9.58) | 11.78 |
| d10 | 10.25 (10.18-10.29) | 10.21 (10.15-10.24) | 10.57 (10.45-10.62) | 10.52 (10.40-10.65) | 12.05 (11.98-12.15) | 13.71 |
| d11 | 12.07 (11.96-12.13) | 12.11 (11.96-12.19) | 12.64 (12.46-12.81) | 12.68 (12.61-12.74) | 15.53 (15.33-15.67) | 16.37 |
| d12 | 14.66 (14.62-14.68) | 14.73 (14.68-14.85) | 15.21 (14.92-15.38) | 15.24 (15.14-15.34) | 18.32 (18.23-18.40) | 18.28 |
| d13 | 17.51 (17.32-17.59) | 17.74 (17.60-17.82) | 17.81 (17.71-17.91) | 17.95 (17.87-18.05) | 20.38 (20.28-20.48) | 19.40 |
| random | 21.89 (21.84-21.92) | 21.95 (21.92-22.01) | 21.43 (21.36-21.47) | 21.50 (21.45-21.53) | 22.71 (22.69-22.74) | 20.79 |
| d20_known | 23.02 (22.55-23.45) | 23.02 (22.64-23.27) | 22.59 (22.09-22.91) | 22.44 (22.18-22.55) | 23.53 (23.45-23.64) | 21.55 |

### Beam (4096) mean excess over exact distance (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| holdout_d5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| holdout_d6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| d7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| d8 | 0.00 (0.00-0.01) | 0.00 (0.00-0.02) | 0.01 (0.00-0.02) | 0.01 (0.00-0.02) | 0.03 (0.02-0.04) |
| d9 | 0.06 (0.04-0.07) | 0.04 (0.03-0.06) | 0.18 (0.14-0.20) | 0.11 (0.07-0.13) | 0.48 (0.39-0.58) |
| d10 | 0.25 (0.18-0.29) | 0.21 (0.15-0.24) | 0.57 (0.45-0.62) | 0.52 (0.40-0.65) | 2.05 (1.98-2.15) |
| d11 | 1.07 (0.96-1.13) | 1.11 (0.96-1.19) | 1.64 (1.46-1.81) | 1.68 (1.61-1.74) | 4.53 (4.33-4.67) |
| d12 | 2.66 (2.62-2.68) | 2.73 (2.68-2.85) | 3.21 (2.92-3.38) | 3.24 (3.14-3.34) | 6.32 (6.23-6.40) |
| d13 | 4.51 (4.32-4.59) | 4.74 (4.60-4.82) | 4.81 (4.71-4.91) | 4.95 (4.87-5.05) | 7.38 (7.28-7.48) |
| random | - | - | - | - | - |
| d20_known | 3.02 (2.55-3.45) | 3.02 (2.64-3.27) | 2.59 (2.09-2.91) | 2.44 (2.18-2.55) | 3.53 (3.45-3.64) |

### Beam (4096) mean excess over Kociemba length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | -0.93 | -0.93 | -0.93 | -0.93 | -0.93 |
| holdout_d5 | -1.20 | -1.20 | -1.20 | -1.20 | -1.20 |
| holdout_d6 | -1.64 | -1.64 | -1.64 | -1.64 | -1.64 |
| d7 | -2.16 | -2.16 | -2.16 | -2.16 | -2.16 |
| d8 | -2.49 (-2.49--2.48) | -2.49 (-2.49--2.47) | -2.48 (-2.49--2.47) | -2.48 (-2.49--2.47) | -2.46 (-2.47--2.45) |
| d9 | -2.73 (-2.75--2.71) | -2.74 (-2.75--2.72) | -2.61 (-2.64--2.59) | -2.67 (-2.71--2.65) | -2.30 (-2.39--2.20) |
| d10 | -3.46 (-3.53--3.42) | -3.50 (-3.56--3.47) | -3.15 (-3.26--3.09) | -3.19 (-3.31--3.06) | -1.67 (-1.73--1.56) |
| d11 | -4.31 (-4.41--4.25) | -4.27 (-4.41--4.18) | -3.74 (-3.92--3.57) | -3.70 (-3.76--3.63) | -0.85 (-1.04--0.70) |
| d12 | -3.62 (-3.67--3.60) | -3.55 (-3.60--3.43) | -3.07 (-3.36--2.90) | -3.04 (-3.14--2.95) | 0.04 (-0.05-0.12) |
| d13 | -1.88 (-2.08--1.80) | -1.66 (-1.80--1.57) | -1.58 (-1.69--1.48) | -1.44 (-1.52--1.35) | 0.98 (0.89-1.09) |
| random | 1.10 (1.06-1.13) | 1.16 (1.13-1.23) | 0.64 (0.57-0.68) | 0.71 (0.66-0.74) | 1.92 (1.90-1.96) |
| d20_known | 1.47 (1.00-1.91) | 1.47 (1.09-1.73) | 1.05 (0.55-1.36) | 0.89 (0.64-1.00) | 1.98 (1.91-2.09) |

### Greedy mean solution length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba |
|---|---|---|---|---|---|---|
| d1_4_all | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 (3.92-3.93) | 4.77 |
| holdout_d5 | 5.00 (5.00-5.01) | 5.00 | 5.00 | 5.01 (5.00-5.05) | 5.00 (5.00-5.01) | 6.29 |
| holdout_d6 | 6.07 (6.01-6.18) | 6.09 (6.01-6.16) | 6.09 (6.03-6.15) | 6.08 (6.00-6.17) | 6.18 (6.05-6.35) | 7.67 |
| d7 | 7.94 (7.73-8.21) | 8.13 (8.01-8.37) | 8.43 (8.22-8.54) | 8.54 (8.36-8.68) | 8.48 (8.29-8.60) | 9.15 |
| d8 | 13.27 (13.00-13.73) | 13.97 (13.60-14.46) | 16.11 (14.98-17.28) | 15.51 (14.20-16.54) | 16.01 (15.46-16.76) | 10.49 |
| d9 | 22.06 (21.50-22.77) | 23.76 (22.39-25.13) | 27.67 (25.84-29.71) | 27.25 (25.99-28.22) | 27.10 (26.42-27.40) | 11.78 |
| d10 | 34.33 (32.55-35.69) | 34.91 (33.79-36.97) | 38.30 (36.66-40.85) | 37.25 (34.94-38.51) | 35.82 (34.31-38.40) | 13.71 |
| d11 | 46.81 (45.54-48.14) | 47.68 (46.46-49.58) | 46.59 (44.60-48.86) | 44.68 (43.35-45.73) | 41.26 (40.61-42.13) | 16.37 |
| d12 | 53.39 (51.81-54.88) | 52.65 (51.24-54.90) | 49.44 (47.48-51.14) | 47.56 (46.49-48.81) | 43.67 (42.02-45.36) | 18.28 |
| d13 | 57.82 (56.31-61.41) | 56.60 (53.89-58.27) | 51.77 (49.65-53.51) | 49.19 (47.67-50.14) | 44.75 (43.28-45.77) | 19.40 |
| random | 61.76 (60.69-63.50) | 61.02 (60.05-62.78) | 52.90 (50.86-54.87) | 51.04 (50.23-52.22) | 47.31 (46.42-48.71) | 20.79 |
| d20_known | 59.72 (48.50-69.00) | 55.15 (40.50-68.00) | 51.38 (45.50-63.38) | 55.02 (46.00-62.12) | 44.61 (41.56-49.29) | 21.55 |

### Greedy, unsolved cubes: wrong pieces (of 20)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | - | - | - | - | - |
| holdout_d5 | - | - | - | - | - |
| holdout_d6 | 13.95 (10.50-17.25) | 15.34 (13.25-18.00) | 13.75 (10.00-18.50) | 14.25 (10.67-19.00) | 16.67 (12.00-19.00) |
| d7 | 15.26 (14.51-15.79) | 15.24 (14.52-16.23) | 15.71 (14.00-16.57) | 15.80 (14.09-17.50) | 15.86 (15.18-16.58) |
| d8 | 15.40 (14.93-15.61) | 15.27 (15.01-15.64) | 14.91 (14.32-15.39) | 15.46 (14.78-16.16) | 15.52 (15.16-16.47) |
| d9 | 15.53 (15.23-15.97) | 15.25 (14.83-15.61) | 15.28 (14.81-15.71) | 15.18 (14.53-15.54) | 15.05 (14.42-15.88) |
| d10 | 15.47 (15.33-15.66) | 15.07 (14.72-15.44) | 15.10 (14.50-15.46) | 15.16 (14.85-15.59) | 15.22 (14.30-16.01) |
| d11 | 15.41 (15.29-15.64) | 15.18 (14.68-15.50) | 15.29 (14.77-15.73) | 15.30 (14.95-15.59) | 15.36 (14.74-15.84) |
| d12 | 15.41 (15.29-15.56) | 15.22 (14.92-15.40) | 14.80 (14.13-15.29) | 15.16 (14.95-15.41) | 14.97 (14.51-15.37) |
| d13 | 15.47 (15.28-15.68) | 15.21 (14.93-15.46) | 14.97 (14.64-15.67) | 15.12 (14.67-15.40) | 15.28 (14.88-15.71) |
| random | 15.48 (15.32-15.67) | 15.18 (14.94-15.59) | 15.18 (14.75-15.61) | 15.16 (14.66-15.65) | 15.06 (14.53-15.58) |
| d20_known | 15.21 (14.11-17.38) | 14.56 (12.86-16.38) | 13.94 (10.40-16.67) | 14.66 (10.67-16.33) | 15.93 (12.00-19.00) |

### Greedy, unsolved cubes: correct stickers (of 48)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | - | - | - | - | - |
| holdout_d5 | - | - | - | - | - |
| holdout_d6 | 23.90 (18.00-31.50) | 21.49 (20.00-24.00) | 24.75 (15.00-31.00) | 23.33 (16.67-29.33) | 16.00 (6.00-26.00) |
| d7 | 20.83 (19.60-22.31) | 21.32 (18.57-23.10) | 19.90 (18.07-22.73) | 19.89 (16.00-23.91) | 20.09 (18.42-22.45) |
| d8 | 20.66 (20.11-21.56) | 20.84 (20.21-21.42) | 21.57 (20.12-23.04) | 20.91 (19.58-22.29) | 20.84 (18.68-21.72) |
| d9 | 20.26 (19.73-20.69) | 20.95 (20.13-21.77) | 21.22 (20.78-22.15) | 21.27 (20.47-22.87) | 21.86 (19.97-23.52) |
| d10 | 20.56 (20.15-20.81) | 21.36 (20.62-22.10) | 21.37 (20.66-22.79) | 21.54 (20.65-22.46) | 21.47 (20.05-23.52) |
| d11 | 20.61 (20.17-20.91) | 21.11 (20.32-21.91) | 21.07 (20.22-22.20) | 21.24 (20.37-22.26) | 21.21 (19.84-22.60) |
| d12 | 20.58 (20.37-20.89) | 21.04 (20.70-21.85) | 22.03 (21.25-23.11) | 21.50 (21.01-21.90) | 22.10 (21.33-23.30) |
| d13 | 20.47 (20.07-20.90) | 21.04 (20.42-21.67) | 21.81 (20.60-22.83) | 21.52 (21.12-22.07) | 21.37 (20.17-22.00) |
| random | 20.42 (20.20-20.65) | 21.10 (20.26-21.69) | 21.42 (20.51-22.36) | 21.50 (20.16-22.78) | 21.87 (20.58-23.23) |
| d20_known | 20.37 (17.00-23.86) | 21.86 (17.38-25.17) | 23.86 (18.50-30.80) | 22.70 (18.33-31.00) | 21.28 (17.00-28.00) |

### Beam-width sweep

#### Beam solve rate, d8

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 84.0% (83.1%-84.7%) | 91.3% (91.0%-91.6%) | 96.0% (95.8%-96.3%) | 99.9% (99.8%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 83.7% (83.0%-84.7%) | 90.1% (89.5%-90.4%) | 95.4% (95.1%-95.7%) | 100.0% (99.9%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 93.8% (92.4%-94.7%) | 95.4% (94.9%-96.3%) | 98.3% (97.8%-99.2%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d9

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 64.7% (62.8%-65.6%) | 77.7% (77.1%-78.6%) | 87.2% (85.6%-88.7%) | 99.6% (99.3%-99.8%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 65.8% (64.3%-67.4%) | 77.1% (75.3%-78.5%) | 88.0% (87.6%-88.5%) | 99.4% (99.0%-99.6%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 88.1% (87.0%-89.3%) | 89.8% (88.8%-90.4%) | 95.4% (93.9%-96.4%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d10

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 48.1% (46.2%-49.7%) | 63.4% (61.8%-66.5%) | 77.8% (76.2%-79.1%) | 98.8% (98.6%-99.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 50.9% (49.2%-53.3%) | 65.1% (61.8%-67.3%) | 79.2% (77.4%-81.4%) | 98.7% (97.8%-99.2%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 82.5% (80.4%-85.3%) | 85.2% (84.4%-86.4%) | 93.0% (91.7%-95.0%) | 100.0% (99.9%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d11

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 37.7% (35.0%-40.6%) | 51.1% (50.0%-53.2%) | 68.1% (65.7%-71.2%) | 97.5% (97.2%-97.8%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 41.8% (38.6%-44.3%) | 54.6% (53.7%-56.4%) | 70.8% (68.2%-72.2%) | 98.4% (98.3%-98.4%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 80.1% (78.3%-82.1%) | 82.1% (80.8%-84.3%) | 91.7% (88.8%-94.4%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d12

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 33.4% (30.5%-35.9%) | 46.2% (44.6%-49.1%) | 63.7% (61.4%-66.8%) | 96.7% (96.2%-97.8%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 37.2% (35.3%-38.8%) | 48.4% (46.4%-50.9%) | 65.8% (64.8%-67.5%) | 97.3% (96.8%-98.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 79.2% (76.6%-82.4%) | 80.8% (78.4%-84.7%) | 91.2% (89.3%-93.3%) | 100.0% (99.9%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d13

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 30.7% (28.9%-33.9%) | 43.1% (41.8%-45.2%) | 59.4% (57.7%-61.4%) | 96.8% (96.5%-97.1%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 35.8% (32.7%-38.4%) | 46.4% (45.6%-47.4%) | 63.7% (59.6%-66.7%) | 97.0% (96.7%-97.4%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 78.2% (75.6%-79.6%) | 80.4% (78.4%-82.0%) | 90.5% (86.9%-93.0%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, random

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 29.7% (26.9%-33.0%) | 42.2% (39.0%-47.2%) | 57.5% (55.7%-60.6%) | 96.0% (95.3%-97.1%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 33.2% (29.9%-36.6%) | 43.7% (43.4%-44.1%) | 59.2% (56.6%-62.7%) | 96.4% (96.1%-96.5%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 75.0% (71.7%-77.5%) | 78.2% (74.7%-81.3%) | 89.8% (87.1%-91.7%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam solve rate, d20_known

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 27.3% (18.2%-36.4%) | 42.4% (36.4%-54.5%) | 54.5% (27.3%-81.8%) | 93.9% (90.9%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| kociemba (20 stages) (n=3) | 31.8% (18.2%-45.5%) | 48.5% (36.4%-54.5%) | 54.5% (45.5%-63.6%) | 93.9% (90.9%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% |
| mixed (n=3) | 74.2% (54.5%-90.9%) | 72.7% (45.5%-90.9%) | 90.9% (81.8%-100.0%) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

#### Beam mean solution length (solved), d8

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 13.27 (13.00-13.73) | 12.28 (11.85-12.91) | 10.57 (10.20-10.80) | 8.86 (8.63-9.06) | 8.14 (8.13-8.14) | 8.03 (8.02-8.05) | 8.02 | 8.00 (8.00-8.01) |
| kociemba (20 stages) (n=3) | 13.97 (13.60-14.46) | 12.60 (12.25-12.88) | 10.75 (10.60-10.90) | 8.78 (8.70-8.86) | 8.13 (8.11-8.17) | 8.04 (8.02-8.06) | 8.01 (8.00-8.03) | 8.00 (8.00-8.02) |
| mixed (n=3) | 16.01 (15.46-16.76) | 14.08 (13.93-14.29) | 12.19 (11.98-12.52) | 9.88 (9.74-10.09) | 8.79 (8.72-8.91) | 8.28 (8.25-8.32) | 8.10 (8.08-8.11) | 8.03 (8.02-8.04) |

#### Beam mean solution length (solved), d9

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 22.06 (21.50-22.77) | 20.40 (20.01-21.11) | 17.21 (16.79-18.04) | 12.65 (12.32-12.90) | 10.00 (9.97-10.05) | 9.36 (9.29-9.43) | 9.10 (9.08-9.12) | 9.06 (9.04-9.07) |
| kociemba (20 stages) (n=3) | 23.76 (22.39-25.13) | 20.33 (19.80-20.60) | 17.37 (16.78-18.18) | 12.43 (12.13-12.66) | 10.00 (9.94-10.09) | 9.39 (9.33-9.47) | 9.10 (9.07-9.11) | 9.04 (9.03-9.06) |
| mixed (n=3) | 27.10 (26.42-27.40) | 21.83 (21.53-22.35) | 19.03 (17.93-20.29) | 14.72 (14.67-14.79) | 12.12 (12.08-12.18) | 10.71 (10.63-10.75) | 9.89 (9.81-10.01) | 9.48 (9.39-9.58) |

#### Beam mean solution length (solved), d10

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 34.33 (32.55-35.69) | 30.19 (28.69-32.13) | 25.48 (24.25-26.32) | 18.77 (18.21-19.05) | 13.22 (13.11-13.39) | 11.43 (11.34-11.59) | 10.58 (10.50-10.64) | 10.25 (10.18-10.29) |
| kociemba (20 stages) (n=3) | 34.91 (33.79-36.97) | 30.85 (30.36-31.82) | 25.45 (24.70-26.46) | 18.08 (17.99-18.20) | 13.28 (13.17-13.37) | 11.46 (11.37-11.52) | 10.58 (10.56-10.62) | 10.21 (10.15-10.24) |
| mixed (n=3) | 35.82 (34.31-38.40) | 29.61 (29.00-30.17) | 26.27 (24.83-28.32) | 20.35 (20.16-20.59) | 16.92 (16.79-17.14) | 14.67 (14.64-14.71) | 13.20 (13.09-13.43) | 12.05 (11.98-12.15) |

#### Beam mean solution length (solved), d11

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 46.81 (45.54-48.14) | 41.60 (41.07-41.88) | 35.42 (34.19-36.89) | 25.97 (25.07-26.74) | 18.11 (17.96-18.28) | 15.01 (14.94-15.11) | 13.09 (13.04-13.17) | 12.07 (11.96-12.13) |
| kociemba (20 stages) (n=3) | 47.68 (46.46-49.58) | 41.21 (39.98-42.73) | 34.41 (32.81-37.31) | 24.62 (24.38-24.80) | 18.10 (18.01-18.26) | 15.23 (15.22-15.26) | 13.44 (13.33-13.53) | 12.11 (11.96-12.19) |
| mixed (n=3) | 41.26 (40.61-42.13) | 34.54 (33.78-35.50) | 30.72 (29.27-32.31) | 24.27 (24.16-24.36) | 20.95 (20.81-21.07) | 18.66 (18.52-18.75) | 16.91 (16.75-17.12) | 15.53 (15.33-15.67) |

#### Beam mean solution length (solved), d12

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 53.39 (51.81-54.88) | 49.03 (47.74-50.81) | 42.28 (41.50-43.10) | 30.53 (30.28-30.98) | 21.83 (21.78-21.86) | 18.50 (18.44-18.57) | 16.34 (16.23-16.44) | 14.66 (14.62-14.68) |
| kociemba (20 stages) (n=3) | 52.65 (51.24-54.90) | 46.54 (44.08-48.68) | 40.23 (38.06-43.34) | 28.88 (28.25-29.80) | 21.83 (21.69-21.94) | 18.68 (18.60-18.78) | 16.45 (16.36-16.55) | 14.73 (14.68-14.85) |
| mixed (n=3) | 43.67 (42.02-45.36) | 36.62 (35.95-37.64) | 33.31 (31.65-35.39) | 26.40 (25.98-26.65) | 23.18 (23.01-23.28) | 21.17 (20.99-21.28) | 19.62 (19.48-19.73) | 18.32 (18.23-18.40) |

#### Beam mean solution length (solved), d13

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 57.82 (56.31-61.41) | 51.83 (49.97-54.81) | 45.49 (44.93-46.16) | 33.72 (32.65-34.78) | 24.58 (24.36-24.79) | 21.29 (21.16-21.38) | 19.13 (18.91-19.30) | 17.51 (17.32-17.59) |
| kociemba (20 stages) (n=3) | 56.60 (53.89-58.27) | 50.80 (49.90-52.30) | 43.35 (42.15-45.43) | 31.79 (31.59-31.90) | 24.13 (24.10-24.17) | 21.42 (21.39-21.49) | 19.36 (19.33-19.38) | 17.74 (17.60-17.82) |
| mixed (n=3) | 44.75 (43.28-45.77) | 37.92 (37.14-39.16) | 34.65 (32.94-36.81) | 27.70 (27.58-27.79) | 24.62 (24.59-24.64) | 22.78 (22.73-22.84) | 21.43 (21.39-21.50) | 20.38 (20.28-20.48) |

#### Beam mean solution length (solved), random

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 61.76 (60.69-63.50) | 57.44 (55.39-58.80) | 49.15 (47.50-50.53) | 36.41 (35.31-36.98) | 27.07 (27.03-27.12) | 24.37 (24.33-24.41) | 22.87 (22.82-22.92) | 21.89 (21.84-21.92) |
| kociemba (20 stages) (n=3) | 61.02 (60.05-62.78) | 54.48 (54.30-54.70) | 45.83 (44.37-47.92) | 33.98 (33.23-34.81) | 26.63 (26.54-26.74) | 24.21 (24.18-24.27) | 22.86 (22.83-22.93) | 21.95 (21.92-22.01) |
| mixed (n=3) | 47.31 (46.42-48.71) | 40.45 (39.33-41.56) | 36.13 (34.69-37.99) | 29.54 (29.27-29.70) | 26.44 (26.43-26.44) | 24.75 (24.73-24.77) | 23.57 (23.54-23.61) | 22.71 (22.69-22.74) |

#### Beam mean solution length (solved), d20_known

| Arm | greedy | width 2 | width 4 | width 16 | width 64 | width 256 | width 1024 | width 4096 |
|---|---|---|---|---|---|---|---|---|
| true_distance (20 stages) (n=3) | 59.72 (48.50-69.00) | 50.31 (42.50-62.17) | 54.04 (47.67-61.33) | 37.99 (34.60-42.20) | 28.73 (28.55-28.91) | 26.27 (25.82-26.73) | 24.24 (24.09-24.36) | 23.02 (22.55-23.45) |
| kociemba (20 stages) (n=3) | 55.15 (40.50-68.00) | 58.83 (54.50-63.00) | 50.31 (41.14-60.80) | 32.31 (30.90-33.73) | 27.67 (27.18-28.00) | 25.52 (25.18-25.82) | 24.39 (24.00-25.00) | 23.02 (22.64-23.27) |
| mixed (n=3) | 44.61 (41.56-49.29) | 41.91 (38.22-46.30) | 38.27 (36.33-41.18) | 31.48 (30.36-32.91) | 27.48 (27.00-28.18) | 25.27 (25.00-25.45) | 24.18 (24.00-24.45) | 23.53 (23.45-23.64) |

### Beam (4096) seconds per cube, clean timing

Seeds 0 to 2 of the 20-stage curricula and mixed; one process per H200, same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a time); wall time / cubes.

| Set | cubes timed | true_distance (n=3) | kociemba (n=3) | mixed (n=3) |
|---|---|---|---|---|
| d1_4_all | 200 | 0.011 | 0.011 | 0.011 |
| holdout_d5 | 200 | 0.025 | 0.025 | 0.025 |
| holdout_d6 | 200 | 0.038 | 0.038 | 0.038 |
| d7 | 200 | 0.051 | 0.051 | 0.051 |
| d8 | 200 | 0.064 (0.064-0.065) | 0.064 | 0.066 (0.065-0.066) |
| d9 | 200 | 0.078 (0.078-0.079) | 0.078 (0.078-0.079) | 0.084 (0.081-0.085) |
| d10 | 200 | 0.094 (0.092-0.095) | 0.094 (0.092-0.096) | 0.122 (0.121-0.123) |
| d11 | 200 | 0.122 (0.121-0.124) | 0.125 (0.122-0.128) | 0.172 (0.170-0.175) |
| d12 | 200 | 0.150 (0.148-0.151) | 0.153 (0.150-0.159) | 0.202 (0.200-0.204) |
| d13 | 200 | 0.189 (0.186-0.191) | 0.192 (0.190-0.195) | 0.227 (0.222-0.235) |
| random | 200 | 0.248 (0.246-0.249) | 0.250 (0.249-0.251) | 0.260 |
| d20_known | 11 | 0.271 (0.265-0.277) | 0.271 (0.266-0.274) | 0.278 (0.277-0.280) |

### Greedy seconds per cube, clean timing

Seeds 0 to 2 of the 20-stage curricula and mixed; one process per H200, same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a time); wall time / cubes.

| Set | cubes timed | true_distance (n=3) | kociemba (n=3) | mixed (n=3) |
|---|---|---|---|---|
| d1_4_all | 200 | 0.0001 | 0.0001 | 0.0001 |
| holdout_d5 | 200 | 0.0001 | 0.0001 | 0.0001 |
| holdout_d6 | 200 | 0.0005 (0.0003-0.0009) | 0.0016 (0.0013-0.0018) | 0.0012 (0.0006-0.0017) |
| d7 | 200 | 0.0018 (0.0017-0.0018) | 0.0017 (0.0017-0.0018) | 0.0017 (0.0016-0.0018) |
| d8 | 200 | 0.0019 | 0.0018 | 0.0018 |
| d9 | 200 | 0.0019 | 0.0019 | 0.0019 |
| d10 | 200 | 0.0019 | 0.0019 | 0.0018 (0.0018-0.0019) |
| d11 | 200 | 0.0019 | 0.0019 | 0.0019 (0.0018-0.0019) |
| d12 | 200 | 0.0019 | 0.0019 | 0.0019 |
| d13 | 200 | 0.0019 | 0.0019 | 0.0019 (0.0018-0.0019) |
| random | 200 | 0.0019 | 0.0019 | 0.0019 (0.0018-0.0019) |
| d20_known | 11 | 0.0314 (0.0312-0.0316) | 0.0313 (0.0309-0.0317) | 0.0317 (0.0315-0.0318) |

## Secondary: best-validation checkpoints

Checkpoint steps: {"true_distance (20 stages)": {"0": 100000, "1": 100000, "2": 100000, "3": 90000, "4": 90000, "5": 90000}, "kociemba (20 stages)": {"0": 100000, "1": 90000, "2": 100000, "3": 100000, "4": 90000, "5": 100000}, "true_distance (2 stages)": {"0": 70000, "1": 80000, "2": 70000, "3": 70000, "4": 70000, "5": 70000}, "kociemba (2 stages)": {"0": 70000, "1": 70000, "2": 70000, "3": 70000, "4": 70000, "5": 70000}, "mixed": {"0": 50000, "1": 40000, "2": 40000, "3": 40000, "4": 40000, "5": 50000}}

### Key comparisons

Difference of means over seeds; 95% Welch t-interval with each seed as one observation; 'Diff/SE' is the difference divided by its standard error; 'Excludes 0' marks intervals that do not contain zero. The same test cubes are used for every run.

| Comparison | Measure | Difference | 95% interval | Diff/SE | Excludes 0 | Seeds |
|---|---|---|---|---|---|---|
| mixed minus true_distance (20 stages) | greedy solve rate, d10 | +32.1 pts | +28.7 pts to +35.6 pts | 21.9 | yes | 6/6 |
| mixed minus true_distance (20 stages) | greedy solve rate, d13 | +43.6 pts | +39.7 pts to +47.4 pts | 28.0 | yes | 6/6 |
| mixed minus true_distance (20 stages) | greedy solve rate, random | +42.9 pts | +36.9 pts to +48.8 pts | 17.6 | yes | 6/6 |
| mixed minus true_distance (20 stages) | optimal-move rate, d10 | -0.9 pts | -1.9 pts to +0.2 pts | 1.8 | no | 6/6 |
| mixed minus true_distance (20 stages) | optimal-move rate, d13 | -1.2 pts | -2.7 pts to +0.2 pts | 2.0 | no | 6/6 |
| mixed minus true_distance (20 stages) | beam excess over exact, d13 | +0.14 | -0.17 to +0.46 | 1.1 | no | 6/6 |
| mixed minus true_distance (20 stages) | beam mean length, random | -0.72 | -0.81 to -0.63 | 19.0 | yes | 6/6 |
| mixed minus true_distance (20 stages) | beam mean length, d20_known | -1.11 | -1.63 to -0.58 | 5.4 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, d10 | +29.3 pts | +25.9 pts to +32.8 pts | 20.7 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, d13 | +38.5 pts | +34.4 pts to +42.5 pts | 21.6 | yes | 6/6 |
| mixed minus kociemba (20 stages) | greedy solve rate, random | +37.7 pts | +31.6 pts to +43.8 pts | 14.7 | yes | 6/6 |
| mixed minus kociemba (20 stages) | optimal-move rate, d10 | -1.0 pts | -2.0 pts to -0.0 pts | 2.4 | yes | 6/6 |
| mixed minus kociemba (20 stages) | optimal-move rate, d13 | -0.9 pts | -2.6 pts to +0.8 pts | 1.2 | no | 6/6 |
| mixed minus kociemba (20 stages) | beam excess over exact, d13 | -0.10 | -0.41 to +0.22 | 0.7 | no | 6/6 |
| mixed minus kociemba (20 stages) | beam mean length, random | -0.77 | -0.86 to -0.68 | 18.9 | yes | 6/6 |
| mixed minus kociemba (20 stages) | beam mean length, d20_known | -1.23 | -1.48 to -0.97 | 11.4 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, d10 | +2.8 pts | +0.9 pts to +4.6 pts | 3.4 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, d13 | +5.1 pts | +2.3 pts to +7.8 pts | 4.3 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | greedy solve rate, random | +5.2 pts | +2.0 pts to +8.3 pts | 3.8 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | optimal-move rate, d10 | +0.2 pts | -0.6 pts to +1.0 pts | 0.5 | no | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | optimal-move rate, d13 | -0.3 pts | -1.7 pts to +1.0 pts | 0.6 | no | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam excess over exact, d13 | +0.24 | +0.05 to +0.43 | 2.9 | yes | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam mean length, random | +0.05 | -0.02 to +0.12 | 1.7 | no | 6/6 |
| kociemba (20 stages) minus true_distance (20 stages) | beam mean length, d20_known | +0.12 | -0.40 to +0.64 | 0.6 | no | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, d10 | +25.3 pts | +23.4 pts to +27.1 pts | 30.6 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, d13 | +36.1 pts | +33.4 pts to +38.8 pts | 31.3 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | greedy solve rate, random | +36.6 pts | +34.4 pts to +38.7 pts | 39.4 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | optimal-move rate, d10 | +2.5 pts | +1.6 pts to +3.4 pts | 6.2 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | optimal-move rate, d13 | +0.1 pts | -0.8 pts to +1.0 pts | 0.2 | no | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam excess over exact, d13 | -0.40 | -0.60 to -0.20 | 4.4 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam mean length, random | -0.74 | -0.81 to -0.67 | 23.7 | yes | 6/6 |
| true_distance (2 stages) minus true_distance (20 stages) | beam mean length, d20_known | -0.77 | -1.30 to -0.25 | 3.8 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, d10 | +25.2 pts | +22.8 pts to +27.6 pts | 24.5 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, d13 | +33.3 pts | +29.5 pts to +37.1 pts | 19.9 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | greedy solve rate, random | +34.7 pts | +29.2 pts to +40.2 pts | 15.0 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | optimal-move rate, d10 | +1.3 pts | +0.6 pts to +2.0 pts | 4.2 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | optimal-move rate, d13 | +0.2 pts | -1.1 pts to +1.6 pts | 0.4 | no | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam excess over exact, d13 | -0.66 | -0.82 to -0.50 | 9.9 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam mean length, random | -0.77 | -0.83 to -0.71 | 30.6 | yes | 6/6 |
| kociemba (2 stages) minus kociemba (20 stages) | beam mean length, d20_known | -0.86 | -1.12 to -0.61 | 8.0 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, d10 | +2.7 pts | +0.3 pts to +5.1 pts | 2.6 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, d13 | +2.3 pts | -1.5 pts to +6.0 pts | 1.4 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | greedy solve rate, random | +3.3 pts | -2.1 pts to +8.7 pts | 1.6 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | optimal-move rate, d10 | -1.0 pts | -1.8 pts to -0.2 pts | 2.7 | yes | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | optimal-move rate, d13 | -0.2 pts | -1.0 pts to +0.7 pts | 0.4 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam excess over exact, d13 | -0.02 | -0.20 to +0.16 | 0.3 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam mean length, random | +0.02 | -0.04 to +0.09 | 0.9 | no | 6/6 |
| kociemba (2 stages) minus true_distance (2 stages) | beam mean length, d20_known | +0.03 | -0.12 to +0.18 | 0.4 | no | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, d10 | +6.8 pts | +3.4 pts to +10.3 pts | 4.8 | yes | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, d13 | +7.4 pts | +3.3 pts to +11.5 pts | 4.2 | yes | 6/6 |
| mixed minus true_distance (2 stages) | greedy solve rate, random | +6.3 pts | +0.2 pts to +12.4 pts | 2.6 | yes | 6/6 |
| mixed minus true_distance (2 stages) | optimal-move rate, d10 | -3.3 pts | -4.4 pts to -2.3 pts | 7.1 | yes | 6/6 |
| mixed minus true_distance (2 stages) | optimal-move rate, d13 | -1.3 pts | -2.8 pts to +0.1 pts | 2.2 | no | 6/6 |
| mixed minus true_distance (2 stages) | beam excess over exact, d13 | +0.54 | +0.22 to +0.87 | 4.0 | yes | 6/6 |
| mixed minus true_distance (2 stages) | beam mean length, random | +0.02 | -0.07 to +0.12 | 0.5 | no | 6/6 |
| mixed minus true_distance (2 stages) | beam mean length, d20_known | -0.33 | -0.49 to -0.18 | 4.9 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, d10 | +4.2 pts | +0.5 pts to +7.8 pts | 2.6 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, d13 | +5.1 pts | +0.7 pts to +9.6 pts | 2.6 | yes | 6/6 |
| mixed minus kociemba (2 stages) | greedy solve rate, random | +3.0 pts | -4.0 pts to +10.0 pts | 1.0 | no | 6/6 |
| mixed minus kociemba (2 stages) | optimal-move rate, d10 | -2.3 pts | -3.4 pts to -1.3 pts | 5.1 | yes | 6/6 |
| mixed minus kociemba (2 stages) | optimal-move rate, d13 | -1.2 pts | -2.6 pts to +0.3 pts | 1.9 | no | 6/6 |
| mixed minus kociemba (2 stages) | beam excess over exact, d13 | +0.56 | +0.25 to +0.88 | 4.6 | yes | 6/6 |
| mixed minus kociemba (2 stages) | beam mean length, random | -0.00 | -0.09 to +0.09 | 0.0 | no | 6/6 |
| mixed minus kociemba (2 stages) | beam mean length, d20_known | -0.36 | -0.52 to -0.20 | 5.1 | yes | 6/6 |

### Optimal-move rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba | baseline: shallow_search_d2 | baseline: random |
|---|---|---|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 91.4% | 31.9% | 6.2% |
| holdout_d5 | 100.0% | 100.0% (99.9%-100.0%) | 100.0% | 100.0% | 99.9% | 90.4% | 30.3% | 6.1% |
| holdout_d6 | 99.0% | 98.6% (98.5%-98.6%) | 99.1% | 98.9% (98.8%-98.9%) | 98.7% (98.6%-98.8%) | 90.5% | 25.1% | 6.2% |
| d7 | 90.9% (90.5%-91.7%) | 89.9% (89.6%-90.3%) | 91.9% (91.1%-92.7%) | 90.8% (90.2%-91.2%) | 91.3% (90.9%-91.6%) | 90.4% | 21.3% | 5.6% |
| d8 | 72.8% (72.2%-73.4%) | 71.5% (70.5%-72.9%) | 73.5% (71.4%-74.8%) | 73.6% (71.8%-75.5%) | 73.1% (71.3%-74.0%) | 89.3% | 19.3% | 6.0% |
| d9 | 55.2% (53.9%-56.4%) | 53.6% (52.6%-55.0%) | 55.5% (54.2%-57.3%) | 56.0% (54.9%-56.8%) | 55.9% (54.2%-58.4%) | 88.1% | 15.6% | 6.7% |
| d10 | 39.4% (38.1%-40.1%) | 39.6% (38.9%-40.0%) | 41.9% (41.0%-42.5%) | 40.9% (40.1%-41.6%) | 38.6% (37.8%-40.1%) | 77.3% | 11.6% | 5.3% |
| d11 | 28.9% (27.2%-30.4%) | 28.6% (26.1%-30.5%) | 30.2% (29.2%-31.9%) | 29.1% (28.4%-30.3%) | 28.2% (26.4%-29.8%) | 56.3% | 9.1% | 6.6% |
| d12 | 25.8% (24.6%-27.0%) | 25.5% (25.0%-26.2%) | 26.7% (24.6%-28.6%) | 25.9% (24.9%-27.7%) | 24.8% (23.1%-25.7%) | 37.5% | 10.1% | 6.2% |
| d13 | 19.2% (18.0%-20.1%) | 18.9% (17.1%-20.4%) | 19.3% (18.6%-20.1%) | 19.1% (18.3%-20.0%) | 17.9% (16.5%-20.3%) | 26.7% | 9.4% | 5.4% |
| random | - | - | - | - | - | - | - | - |
| d20_known | - | - | - | - | - | - | - | - |

### Greedy solve rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: shallow_search_d2 |
|---|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 13.4% |
| holdout_d5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.4% |
| holdout_d6 | 99.9% (99.6%-100.0%) | 99.7% (99.3%-99.8%) | 99.9% (99.8%-100.0%) | 99.8% (99.8%-99.9%) | 99.9% (99.8%-100.0%) | 0.4% |
| d7 | 96.7% (96.2%-97.4%) | 96.5% (96.0%-97.1%) | 98.4% (97.9%-98.7%) | 98.3% (97.8%-98.5%) | 98.8% (98.1%-99.3%) | 0.0% |
| d8 | 84.5% (84.0%-85.2%) | 84.0% (83.0%-84.9%) | 91.6% (90.8%-92.7%) | 92.2% (90.2%-92.9%) | 94.0% (93.2%-96.5%) | 0.0% |
| d9 | 65.2% (62.8%-67.6%) | 65.4% (64.3%-67.5%) | 81.0% (79.0%-82.6%) | 83.0% (80.1%-86.7%) | 86.7% (82.9%-89.5%) | 0.0% |
| d10 | 47.8% (45.6%-49.5%) | 50.5% (49.2%-52.3%) | 73.0% (71.3%-74.5%) | 75.7% (73.0%-78.5%) | 79.9% (75.8%-84.6%) | 0.0% |
| d11 | 37.6% (35.9%-40.6%) | 41.0% (38.6%-43.4%) | 67.7% (66.5%-69.7%) | 71.0% (64.9%-74.2%) | 76.2% (70.6%-80.6%) | 0.0% |
| d12 | 33.8% (30.5%-36.4%) | 37.9% (35.3%-40.9%) | 67.7% (65.8%-70.4%) | 70.2% (67.4%-73.3%) | 74.4% (68.8%-79.7%) | 0.0% |
| d13 | 30.5% (28.9%-32.6%) | 35.6% (32.7%-39.6%) | 66.7% (62.7%-70.1%) | 68.9% (64.1%-71.8%) | 74.1% (68.9%-78.2%) | 0.0% |
| random | 28.5% (26.5%-31.0%) | 33.7% (29.9%-36.7%) | 65.1% (62.9%-66.3%) | 68.4% (60.2%-73.5%) | 71.4% (62.6%-78.4%) | 0.0% |
| d20_known | 28.8% (18.2%-45.5%) | 33.3% (27.3%-36.4%) | 68.2% (45.5%-90.9%) | 60.6% (36.4%-81.8%) | 80.3% (72.7%-100.0%) | 0.0% |

### Beam (4096) solve rate

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout_d5 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| holdout_d6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d7 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d10 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d11 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d13 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| random | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| d20_known | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### Beam (4096) mean solution length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba |
|---|---|---|---|---|---|---|
| d1_4_all | 3.92 | 3.92 | 3.92 | 3.92 | 3.92 | 4.77 |
| holdout_d5 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 6.29 |
| holdout_d6 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 7.67 |
| d7 | 7.00 | 7.00 | 7.00 | 7.00 | 7.00 | 9.15 |
| d8 | 8.00 (8.00-8.01) | 8.00 | 8.00 (8.00-8.02) | 8.00 (8.00-8.01) | 8.00 (8.00-8.01) | 10.49 |
| d9 | 9.06 (9.04-9.07) | 9.04 (9.03-9.06) | 9.05 (9.02-9.12) | 9.04 (9.01-9.06) | 9.06 (9.02-9.15) | 11.78 |
| d10 | 10.27 (10.18-10.35) | 10.22 (10.19-10.24) | 10.28 (10.22-10.47) | 10.23 (10.17-10.35) | 10.34 (10.22-10.44) | 13.71 |
| d11 | 12.13 (12.06-12.22) | 12.15 (12.02-12.30) | 12.08 (11.93-12.34) | 11.99 (11.86-12.08) | 12.45 (12.14-12.96) | 16.37 |
| d12 | 14.69 (14.62-14.78) | 14.79 (14.68-15.00) | 14.43 (14.27-14.86) | 14.44 (14.35-14.50) | 15.02 (14.84-15.33) | 18.28 |
| d13 | 17.55 (17.32-17.73) | 17.79 (17.60-18.01) | 17.15 (16.93-17.44) | 17.13 (17.03-17.24) | 17.70 (17.44-18.08) | 19.40 |
| random | 21.92 (21.86-21.97) | 21.97 (21.92-22.07) | 21.18 (21.12-21.29) | 21.20 (21.16-21.22) | 21.20 (21.12-21.31) | 20.79 |
| d20_known | 22.98 (22.27-23.45) | 23.11 (22.73-23.27) | 22.21 (22.09-22.36) | 22.24 (22.09-22.36) | 21.88 (21.73-22.09) | 21.55 |

### Beam (4096) mean excess over exact distance (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| holdout_d5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| holdout_d6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| d7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| d8 | 0.00 (0.00-0.01) | 0.00 | 0.00 (0.00-0.02) | 0.00 (0.00-0.01) | 0.00 (0.00-0.01) |
| d9 | 0.06 (0.04-0.07) | 0.04 (0.03-0.06) | 0.05 (0.02-0.12) | 0.04 (0.01-0.06) | 0.06 (0.02-0.15) |
| d10 | 0.27 (0.18-0.35) | 0.22 (0.19-0.24) | 0.28 (0.22-0.47) | 0.23 (0.17-0.34) | 0.34 (0.22-0.44) |
| d11 | 1.13 (1.06-1.22) | 1.15 (1.02-1.30) | 1.08 (0.93-1.34) | 0.99 (0.86-1.08) | 1.45 (1.14-1.97) |
| d12 | 2.69 (2.62-2.78) | 2.79 (2.68-3.00) | 2.43 (2.27-2.86) | 2.44 (2.35-2.50) | 3.02 (2.84-3.33) |
| d13 | 4.55 (4.32-4.73) | 4.79 (4.60-5.01) | 4.15 (3.93-4.44) | 4.13 (4.03-4.24) | 4.70 (4.44-5.08) |
| random | - | - | - | - | - |
| d20_known | 2.98 (2.27-3.45) | 3.11 (2.73-3.27) | 2.21 (2.09-2.36) | 2.24 (2.09-2.36) | 1.88 (1.73-2.09) |

### Beam (4096) mean excess over Kociemba length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | -0.93 | -0.93 | -0.93 | -0.93 | -0.93 |
| holdout_d5 | -1.20 | -1.20 | -1.20 | -1.20 | -1.20 |
| holdout_d6 | -1.64 | -1.64 | -1.64 | -1.64 | -1.64 |
| d7 | -2.16 | -2.16 | -2.16 | -2.16 | -2.16 |
| d8 | -2.49 (-2.49--2.48) | -2.49 | -2.49 (-2.49--2.47) | -2.49 (-2.49--2.48) | -2.49 (-2.49--2.48) |
| d9 | -2.72 (-2.75--2.71) | -2.74 (-2.75--2.72) | -2.73 (-2.77--2.66) | -2.75 (-2.77--2.73) | -2.73 (-2.76--2.63) |
| d10 | -3.44 (-3.53--3.36) | -3.49 (-3.52--3.47) | -3.43 (-3.50--3.24) | -3.48 (-3.54--3.37) | -3.37 (-3.49--3.27) |
| d11 | -4.24 (-4.32--4.15) | -4.22 (-4.36--4.08) | -4.30 (-4.44--4.03) | -4.38 (-4.51--4.29) | -3.92 (-4.24--3.41) |
| d12 | -3.60 (-3.67--3.51) | -3.49 (-3.60--3.28) | -3.85 (-4.02--3.43) | -3.84 (-3.93--3.79) | -3.26 (-3.44--2.96) |
| d13 | -1.84 (-2.08--1.66) | -1.60 (-1.80--1.39) | -2.24 (-2.46--1.96) | -2.26 (-2.37--2.16) | -1.70 (-1.95--1.32) |
| random | 1.13 (1.07-1.19) | 1.18 (1.13-1.28) | 0.39 (0.33-0.51) | 0.41 (0.38-0.44) | 0.41 (0.34-0.52) |
| d20_known | 1.44 (0.73-1.91) | 1.56 (1.18-1.73) | 0.67 (0.55-0.82) | 0.70 (0.55-0.82) | 0.33 (0.18-0.55) |

### Greedy mean solution length (solved)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) | baseline: kociemba |
|---|---|---|---|---|---|---|
| d1_4_all | 3.92 | 3.92 | 3.92 | 3.92 | 3.93 (3.92-3.97) | 4.77 |
| holdout_d5 | 5.00 (5.00-5.01) | 5.00 | 5.00 (5.00-5.01) | 5.01 (5.00-5.03) | 5.01 (5.00-5.06) | 6.29 |
| holdout_d6 | 6.05 (6.01-6.11) | 6.10 (6.01-6.23) | 6.09 (6.04-6.16) | 6.13 (6.03-6.25) | 6.21 (6.13-6.34) | 7.67 |
| d7 | 8.01 (7.73-8.21) | 8.11 (7.71-8.37) | 8.21 (7.66-8.84) | 8.60 (8.49-8.74) | 8.40 (8.06-8.82) | 9.15 |
| d8 | 13.37 (13.00-13.91) | 13.91 (13.60-14.11) | 15.32 (14.81-15.88) | 15.44 (14.28-16.89) | 15.30 (14.84-16.09) | 10.49 |
| d9 | 22.17 (20.87-23.36) | 23.48 (22.39-24.74) | 26.94 (26.06-28.25) | 26.48 (25.37-27.83) | 24.80 (22.88-26.23) | 11.78 |
| d10 | 34.03 (32.55-35.46) | 34.86 (33.79-36.44) | 37.12 (35.65-38.66) | 36.64 (34.61-38.92) | 33.58 (32.49-34.75) | 13.71 |
| d11 | 46.47 (45.49-48.14) | 47.57 (46.55-49.58) | 45.74 (43.43-47.65) | 44.60 (43.43-45.74) | 39.24 (36.96-40.39) | 16.37 |
| d12 | 53.01 (51.81-54.88) | 53.18 (51.24-54.90) | 48.99 (48.26-50.28) | 47.54 (45.98-49.94) | 41.90 (40.11-43.84) | 18.28 |
| d13 | 58.29 (56.31-60.99) | 56.62 (53.89-57.85) | 50.89 (49.80-52.26) | 50.44 (48.85-51.57) | 43.52 (41.69-46.80) | 19.40 |
| random | 61.14 (59.24-63.50) | 60.81 (58.94-62.78) | 52.67 (51.35-54.33) | 52.29 (50.04-54.24) | 44.95 (44.07-46.76) | 20.79 |
| d20_known | 55.93 (44.67-69.00) | 60.69 (51.50-71.25) | 49.76 (43.00-62.50) | 51.78 (47.75-62.12) | 45.96 (36.67-54.88) | 21.55 |

### Greedy, unsolved cubes: wrong pieces (of 20)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | - | - | - | - | - |
| holdout_d5 | - | - | - | - | - |
| holdout_d6 | 14.60 (11.00-18.00) | 15.56 (13.25-19.00) | 17.90 (15.50-20.00) | 15.58 (10.00-20.00) | 17.33 (16.00-19.00) |
| d7 | 15.20 (14.51-15.79) | 15.21 (14.72-16.23) | 15.48 (13.38-16.62) | 14.95 (13.06-16.31) | 13.91 (12.57-14.58) |
| d8 | 15.43 (15.11-15.61) | 15.36 (15.02-15.86) | 14.64 (14.08-14.92) | 14.69 (14.29-15.20) | 15.43 (14.74-16.11) |
| d9 | 15.33 (14.90-15.73) | 15.25 (15.17-15.34) | 14.95 (14.63-15.51) | 14.86 (13.87-15.44) | 15.16 (14.79-15.65) |
| d10 | 15.37 (15.09-15.66) | 15.12 (14.90-15.44) | 15.02 (14.71-15.32) | 14.95 (14.27-15.41) | 15.20 (14.66-15.87) |
| d11 | 15.35 (15.15-15.49) | 15.20 (14.99-15.32) | 15.03 (14.77-15.18) | 15.06 (14.20-15.47) | 15.00 (14.72-15.44) |
| d12 | 15.43 (15.29-15.56) | 15.21 (14.83-15.40) | 14.96 (14.73-15.27) | 14.95 (14.28-15.36) | 15.09 (14.16-15.82) |
| d13 | 15.40 (15.19-15.68) | 15.22 (14.93-15.54) | 15.07 (14.83-15.48) | 14.92 (14.01-15.43) | 15.12 (14.61-15.93) |
| random | 15.50 (15.40-15.67) | 15.23 (14.95-15.49) | 14.95 (14.71-15.18) | 15.10 (13.71-15.53) | 15.22 (14.70-15.79) |
| d20_known | 16.03 (14.14-17.38) | 14.31 (12.14-16.38) | 16.93 (14.50-20.00) | 15.23 (13.00-17.50) | 15.53 (13.50-18.00) |

### Greedy, unsolved cubes: correct stickers (of 48)

| Set | true_distance (20 stages) (n=6) | kociemba (20 stages) (n=6) | true_distance (2 stages) (n=6) | kociemba (2 stages) (n=6) | mixed (n=6) |
|---|---|---|---|---|---|
| d1_4_all | - | - | - | - | - |
| holdout_d5 | - | - | - | - | - |
| holdout_d6 | 22.05 (9.00-30.00) | 20.62 (15.00-24.00) | 15.50 (13.00-19.00) | 19.08 (6.00-30.50) | 17.50 (14.00-21.50) |
| d7 | 20.82 (19.60-22.31) | 21.29 (18.57-22.46) | 20.55 (17.85-24.05) | 21.79 (18.81-24.75) | 23.78 (22.25-26.29) |
| d8 | 20.71 (20.31-21.39) | 20.60 (19.45-21.42) | 22.26 (21.52-23.35) | 22.27 (21.00-23.07) | 20.81 (19.02-22.24) |
| d9 | 20.75 (19.73-21.71) | 21.03 (20.60-21.46) | 21.63 (20.52-22.42) | 22.11 (21.21-23.96) | 21.38 (20.37-22.25) |
| d10 | 20.70 (20.15-21.24) | 21.25 (20.62-21.66) | 21.60 (21.19-22.17) | 21.86 (21.15-23.32) | 21.55 (20.31-22.70) |
| d11 | 20.71 (20.42-21.05) | 21.12 (20.83-21.53) | 21.56 (21.23-22.01) | 21.56 (20.76-23.18) | 21.69 (20.82-22.47) |
| d12 | 20.53 (20.37-20.89) | 20.96 (20.49-21.73) | 21.60 (21.23-21.95) | 21.94 (21.31-23.27) | 21.59 (19.91-23.11) |
| d13 | 20.63 (20.07-21.17) | 21.01 (20.34-21.62) | 21.58 (21.01-22.34) | 21.94 (21.06-23.89) | 21.53 (19.83-22.16) |
| random | 20.41 (20.20-20.66) | 21.04 (20.44-21.68) | 21.64 (21.18-22.50) | 21.70 (20.61-24.36) | 21.31 (20.17-22.28) |
| d20_known | 19.34 (17.00-23.86) | 22.33 (17.38-26.57) | 19.48 (13.00-24.00) | 21.96 (18.00-26.00) | 19.83 (14.33-24.00) |

### Beam (4096) seconds per cube, clean timing

Seeds 0 to 2 of the 20-stage curricula and mixed; one process per H200, same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a time); wall time / cubes.

| Set | cubes timed | true_distance (n=3) | kociemba (n=3) | mixed (n=3) |
|---|---|---|---|---|
| d1_4_all | 200 | 0.011 | 0.011 | 0.011 |
| holdout_d5 | 200 | 0.025 | 0.025 | 0.025 |
| holdout_d6 | 200 | 0.038 | 0.038 | 0.038 |
| d7 | 200 | 0.051 | 0.051 | 0.051 |
| d8 | 200 | 0.064 (0.064-0.065) | 0.064 | 0.065 |
| d9 | 200 | 0.078 (0.078-0.079) | 0.078 (0.078-0.079) | 0.078 (0.078-0.080) |
| d10 | 200 | 0.094 (0.092-0.095) | 0.095 (0.094-0.096) | 0.097 |
| d11 | 200 | 0.122 (0.121-0.124) | 0.126 (0.124-0.128) | 0.131 (0.125-0.140) |
| d12 | 200 | 0.150 (0.148-0.151) | 0.154 (0.150-0.160) | 0.161 (0.159-0.163) |
| d13 | 200 | 0.189 (0.186-0.191) | 0.192 (0.190-0.195) | 0.192 (0.188-0.197) |
| random | 200 | 0.248 (0.246-0.249) | 0.250 (0.249-0.251) | 0.239 (0.237-0.242) |
| d20_known | 11 | 0.271 (0.265-0.277) | 0.273 (0.270-0.274) | 0.256 (0.255-0.258) |

### Greedy seconds per cube, clean timing

Seeds 0 to 2 of the 20-stage curricula and mixed; one process per H200, same batching as the evaluation (greedy: the whole subsample at once; beam: 32 cubes at a time); wall time / cubes.

| Set | cubes timed | true_distance (n=3) | kociemba (n=3) | mixed (n=3) |
|---|---|---|---|---|
| d1_4_all | 200 | 0.0001 | 0.0001 | 0.0001 |
| holdout_d5 | 200 | 0.0001 | 0.0001 | 0.0001 |
| holdout_d6 | 200 | 0.0005 (0.0003-0.0009) | 0.0018 (0.0017-0.0018) | 0.0014 (0.0011-0.0018) |
| d7 | 200 | 0.0018 (0.0017-0.0018) | 0.0018 | 0.0018 (0.0018-0.0019) |
| d8 | 200 | 0.0019 | 0.0018 (0.0018-0.0019) | 0.0020 (0.0019-0.0020) |
| d9 | 200 | 0.0019 | 0.0019 | 0.0020 |
| d10 | 200 | 0.0019 | 0.0019 | 0.0020 |
| d11 | 200 | 0.0019 | 0.0019 | 0.0020 |
| d12 | 200 | 0.0019 | 0.0019 | 0.0020 (0.0020-0.0021) |
| d13 | 200 | 0.0019 | 0.0019 | 0.0020 |
| random | 200 | 0.0019 | 0.0019 | 0.0020 |
| d20_known | 11 | 0.0314 (0.0312-0.0316) | 0.0315 (0.0313-0.0317) | 0.0335 (0.0326-0.0341) |
