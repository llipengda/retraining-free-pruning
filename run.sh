#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi
DIR="$1"
DIR_NAME=$(basename "$DIR")

for folder in "$DIR"/*; do
  if [ -d "$folder" ]; then
    task_name=$(basename "$folder")
    echo "Running task: $task_name"
    python3 main.py --model_name $DIR_NAME \
                --task_name $task_name \
                --ckpt_dir $folder \
                --constraint 0.6 \
                --seed 0
  else
    echo "Skipping non-directory: $folder"
  fi
done