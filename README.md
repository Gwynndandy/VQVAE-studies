# VQVAE_studies
Studies for vector quantized variational autoencoders for denoising.

## Overview of this directory
- post_processing: python scripts to process training run data for reporting and hyperparameter tuning
- run_data: where run information is stored in folders:
    - Experiment
        - run
            - config.yaml (defining the run parameters)
            - model_dict.pth
            - model.pth
            - output_n.png (from runtime training)
            - history.csv
            - test_stats.csv
- sample_data_detector
    - sample_data_signal.npy
    - sample_data_clnsig.npy
- scripts
    - src (python libraries for use in model training)
        - autoencoders.py (autoencoder models and tools)
        - cnns.py (CNNs models and tools)
        - data_handling.py (tools for handling data)
        - training.py (training loops and the like)
    - yamls
        - config.yaml (define models and dataset for runs)
    - batch_run*.sh (bash script to automate runs)
    - Sys-Test.py (Simple test to make sure things work)

## To Dos:
- Finish refactoring everything so I can be free of code debt (cope)
- Finish pytorch ROI finder
    - Mostly just 
- Write H5 stuff if it does not exist
    - Need to read .npys efficiently
- Once approved merge with main branch
