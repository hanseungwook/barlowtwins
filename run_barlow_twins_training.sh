#!/bin/bash

# Barlow Twins Training (2 GPU) - Script Version
# Converted from VS Code launch configuration

# Set PYTHONPATH to include WiLoR-mini workspace
export PYTHONPATH="/home/swhan/projects/WiLoR-mini"

# Run the Barlow Twins training script
python main_vla_egoexodataloader.py \
    "/mnt/nfs_csail/hamer_diffusion_policy_project/datasets/egoexo/" \
    "/home/swhan/projects/barlowtwins/configs/barlow_cfg.yaml" \
    --debug-max-takes=600
