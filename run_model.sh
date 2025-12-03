#!/bin/bash
# Workaround for TensorFlow mutex issues on macOS
export TF_ENABLE_ONEDNN_OPTS=0
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

python3 -u nba_model.py
