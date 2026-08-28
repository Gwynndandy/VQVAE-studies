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

for value in "${values[@]}"; do
    echo
    echo "========================================"
    echo "Running VQ-VAE.py with value: $value"
    echo "========================================"

    #python VQ-VAE.py "$value"
    python VTB.py "$value"


done

echo
echo "All runs completed."