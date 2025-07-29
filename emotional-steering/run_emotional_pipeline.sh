#!/bin/bash

# Complete emotional steering pipeline
# Usage: bash run_emotional_pipeline.sh [model] [emotion] [n_examples]

MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
EMOTION=${2:-"depressive"}
N_EXAMPLES=${3:-100}

echo "Running emotional steering pipeline for $MODEL with $EMOTION emotion ($N_EXAMPLES examples)"

# Step 1: Generate emotional examples
echo "Step 1: Generating emotional examples..."
python generate_emotional_examples.py \
    --emotion_category $EMOTION \
    --n_examples $N_EXAMPLES \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate emotional examples"
    exit 1
fi

# Step 2: Train emotional vectors
echo "Step 2: Training emotional vectors..."
python train_emotional_vectors.py \
    --model $MODEL \
    --target_emotion $EMOTION \
    --n_samples $N_EXAMPLES \
    --batch_size 4 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Failed to train emotional vectors"
    exit 1
fi

# Step 3: Evaluate emotional steering
echo "Step 3: Evaluating emotional steering..."
python evaluate_emotional_steering.py \
    --model $MODEL \
    --emotions $EMOTION \
    --n_examples 20 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate emotional steering"
    exit 1
fi

echo "Emotional steering pipeline completed successfully!"
echo "Results saved in results/ directory"