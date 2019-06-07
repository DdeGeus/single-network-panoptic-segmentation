#!/bin/bash

base_dir="examples/params/"

for filename in "$base_dir"* ; do
    echo "$(basename "$filename")"
    echo $filename

    nvidia-smi -q -d temperature | grep GPU

    python train.py --json_path=$filename


done
