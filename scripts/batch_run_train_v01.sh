#!/usr/bin/env bash

set -e

hyperperameters=("  'batch_size': " "  'hid': " "  'z_ch': " "  'n_codes': " "  'lr': " "  'weight_decay': " "  'min_lr': ")

cat batch_v01_nucc.yaml > batch_nucc.yaml
cat batch_v01_nues.yaml > batch_nues.yaml
mkdir -p $1

NUM_LOOPS=$1
NUM_RUNS=$2
for ((j=1; j<=NUM_LOOPS; j++)); do
    echo
    echo "========================================"
    echo "      Training models in loop $j        "
    echo "========================================"

    sed -i "s/  'target_snr':.*/  'target_snr': $value/g"  batch.yaml
    echo "'target_snr': $value" >> $1/log.txt

    for ((i=1; i<=NUM_RUNS; i++)); do
        hyperperameter=${hyperperameters[ $RANDOM % ${#hyperperameters[@]} ]}
        current_value=$(grep -F "$hyperperameter" batch.yaml | head -n1 | sed -E "s/.*: *([0-9.eE+-]+).*/\1/")
        updated_value=$(awk -v base="$current_value" -v seed="$RANDOM" 'BEGIN{
                            srand(seed);
                            factor = 0.80 + rand() * 0.2;
                            printf "%.6g", base * factor
                        }')
        sed -i "s|${hyperperameter}.*|${hyperperameter}${updated_value}|g" batch.yaml
        echo "   ${hyperperameter}${updated_value}" >> "$1/log.txt"

        echo "========================================"
        echo "Run $i/$NUM_RUNS for value: $value"
        echo "Changed ${hyperperameter} : ${updated_value}"
        echo "========================================"
        

        python VQ_VAE.py batch_nues.yaml 0
        python VQ_VAE.py batch_nucc.yaml 0
    done
    cat batch_init.yaml > batch.yaml
done

echo
echo "All runs completed."
