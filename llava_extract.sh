#!/bin/bash

# Download LLaVA-Instruct-150K dataset
wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json

# Download COCO dataset
wget -c http://images.cocodataset.org/zips/train2017.zip

# Create coco directory if it doesn't exist
mkdir -p data/coco

# Extract COCO dataset
unzip -q train2017.zip -d data/coco/

echo "Download and extraction complete."