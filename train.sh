#!/usr/bin/env bash

for model_cfg in "$@"; do 
  name="${model_cfg%.*}"
  name="${model_cfg##*/}"
  python3 main.py train "$name" --config "$model_cfg"
done
