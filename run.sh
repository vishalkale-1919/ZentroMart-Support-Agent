#!/usr/bin/env bash
# VoiceOps 2026 Batch Runner Script
set -e

INPUT_DIR=${1:-"./sample_inputs"}
OUTPUT_DIR=${2:-"./sample_outputs"}
LANGUAGE=${3:-"en-IN"}

echo "Running VoiceOps 2026 Batch Runner..."
python run.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --language "$LANGUAGE"
