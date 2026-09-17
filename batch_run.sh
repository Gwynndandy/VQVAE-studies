#!/usr/bin/env bash

set -e

values=(
100
51.79
26.83
13.89
7.20
3.73
1.93
1.00
0.8
0.7
0.6
0.518
0.268
0.139
0.072
0.0373
0.0193
0.01
)

NUM_RUNS=3
for value in "${values[@]}"; do
    echo
    echo "========================================"
    echo "Training models with value: $value"
    echo "========================================"

    for ((i=1; i<=NUM_RUNS; i++)); do
        echo "Run $i/$NUM_RUNS for value: $value"

        python VQ-VAE.py "$value"
        python VTB.py "$value"
        #python VTB-ROI-Finder.py "$value"
    done
done

echo
echo "All runs completed."