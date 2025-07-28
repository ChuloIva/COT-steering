# %%
# This script trains steering vectors for language models by analyzing their reasoning patterns
# It processes model responses, extracts thinking processes, and computes mean vectors for different
# reasoning components across the model's layers

import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer
import torch
import re
from nnsight import NNsight
from collections import defaultdict
import os
import random
import json
import utils
from utils import process_saved_responses_batch
import math
import gc
from tqdm import tqdm

# Command line argument setup for configuring the training process
parser = argparse.ArgumentParser(description="Generate annotations and train steering vectors for model reasoning")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--save_every", type=int, default=1, 
                    help="Save checkpoints every n batches")
parser.add_argument("--responses_path", type=str, default=None,
                    help="Path to JSON file containing responses")
parser.add_argument("--n_samples", type=int, default=100,
                    help="Number of samples to process")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for processing messages")
args, _ = parser.parse_known_args()

# Example command:
# python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n_samples 500 --max_tokens 1000 --batch_size 4 --save_every 1 --load_from_json --update_annotation

# %%
def extract_thinking_process(response):
    """
    Extracts the thinking process from a model's response by finding content between <think> tags
    
    Args:
        response (str): The full response text containing thinking process
        
    Returns:
        str: The extracted thinking process text
    """
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """
    Updates the running mean vectors for different reasoning components across model layers
    
    Args:
        mean_vectors (dict): Dictionary storing mean vectors for each label and overall thinking
        layer_outputs (torch.Tensor): Model's layer outputs for current batch
        label_positions (dict): Start and end positions for each reasoning label in the text
        index (int): Current sample index for debugging
    """
    # First calculate the overall thinking section boundaries by finding min/max positions
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update the mean vector for the overall thinking process
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        # Use running average formula: new_mean = old_mean + (new_value - old_mean)/(count + 1)
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        # Check for numerical instability
        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {index}")
    
    # Update mean vectors for individual reasoning components/labels
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            # Take mean of vectors for the label, limiting to 10 tokens after start to avoid long spans
            vectors = layer_outputs[:, start-1:min(end-1, start+10)].mean(dim=1)
            
            # Extensive error checking for numerical stability
            if torch.isnan(vectors).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")
                print(f"Layer outputs: {layer_outputs[:, start-1:min(end-1, start+2)]}")
                print(f"Layer outputs shape: {layer_outputs.shape}")
                print(f"Positions: {positions}")
                print(f"Index: {index}")
                print(f"Label: {label}")
                print(f"Start: {start}")
                print(f"End: {end}")
                print(f"Vectors: {vectors}")
                print(f"Current count: {mean_vectors[label]['count']}")
                print(f"Current mean: {mean_vectors[label]['mean']}")
                
                continue
            
            # Update running mean for this label
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1

def compute_emotional_feature_vectors(mean_vectors_dict):
    """
    Compute feature vectors by subtracting normal-thinking mean from negative emotional category means
    This implements the depressive-normal dichotomy approach
    
    Args:
        mean_vectors_dict (dict): Dictionary containing mean vectors for each label
        
    Returns:
        dict: Feature vectors for emotional categories
    """
    feature_vectors = {}
    
    # Check if we have normal-thinking vectors to use as baseline
    if "normal-thinking" not in mean_vectors_dict:
        print("Warning: No normal-thinking vectors found. Using overall mean as baseline.")
        baseline_mean = mean_vectors_dict.get("overall", {}).get("mean")
        if baseline_mean is None:
            print("Error: No baseline vectors available for feature computation")
            return {}
    else:
        baseline_mean = mean_vectors_dict["normal-thinking"]["mean"]
        print(f"Using normal-thinking as baseline with {mean_vectors_dict['normal-thinking']['count']} samples")
    
    # Add the baseline to feature vectors
    feature_vectors["baseline"] = baseline_mean
    
    # Compute differential vectors for negative emotional categories
    negative_emotional_labels = ["depressive-thinking", "anxious-thinking", "negative-attribution", "pessimistic-projection"]
    
    for label in negative_emotional_labels:
        if label in mean_vectors_dict:
            label_mean = mean_vectors_dict[label]["mean"]
            feature_vectors[label] = label_mean - baseline_mean
            print(f"Computed feature vector for {label} (samples: {mean_vectors_dict[label]['count']})")
        else:
            print(f"Warning: No vectors found for {label}")
    
    # Also include cognitive labels if they exist (subtract from overall mean instead)
    cognitive_labels = ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]
    
    if "overall" in mean_vectors_dict:
        overall_mean = mean_vectors_dict["overall"]["mean"]
        for label in cognitive_labels:
            if label in mean_vectors_dict:
                label_mean = mean_vectors_dict[label]["mean"]
                feature_vectors[label] = label_mean - overall_mean
                print(f"Computed cognitive feature vector for {label}")
    
    return feature_vectors

# %% Main execution
model_name = args.model

# Setup directory structure
os.makedirs('results/vars', exist_ok=True)

# Configure paths for saving checkpoints
save_every = args.save_every
save_path = f"results/vars/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

# Set up path for responses data
responses_json_path = args.responses_path or f"results/vars/responses_{model_name.split('/')[-1].lower()}.json"

if not os.path.exists(responses_json_path):
    raise FileNotFoundError(f"Responses file not found at {responses_json_path}. Please generate responses first.")

# Initialize model and tokenizer
print(f"Loading model {model_name}...")
model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name, load_in_8bit=args.load_in_8bit)

# Initialize dictionary to store mean vectors for each reasoning component
# Structure: {label: {'mean': tensor of shape [num_layers, hidden_size], 'count': number of samples}}
mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# Load and shuffle the pre-generated responses
print(f"Loading responses from {responses_json_path}")
with open(responses_json_path, 'r') as f:
    responses_data = json.load(f)

random.seed(args.seed)
random.shuffle(responses_data)

# Process responses in batches
num_batches = math.ceil(min(len(responses_data), args.n_samples) / args.batch_size)

# Main training loop
for batch_idx in tqdm(range(num_batches), desc="Processing responses"):
    # Prepare batch data
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, min(len(responses_data), args.n_samples))
    
    batch_responses = responses_data[start_idx:end_idx]
    thinking_processes = [data["thinking_process"] for data in batch_responses]
    batch_full_responses = [data["full_response"] for data in batch_responses]
    batch_indices = list(range(start_idx, end_idx))
    
    # Generate annotations for the thinking processes
    annotated_responses = utils.process_batch_annotations(thinking_processes)
    
    # Update the responses data with new annotations
    for i, (response_data, annotated) in enumerate(zip(batch_responses, annotated_responses)):
        responses_data[start_idx + i]["annotated_thinking"] = annotated
    
    # Get model's layer outputs for the batch
    batch_layer_outputs = process_saved_responses_batch(batch_full_responses, tokenizer, model)
    
    # Update mean vectors based on annotations
    for i, (response_data, layer_outputs) in enumerate(zip(batch_responses, batch_layer_outputs)):
        if annotated_responses[i]:  # Only process if valid annotations exist
            label_positions = utils.get_label_positions(annotated_responses[i], response_data["full_response"], tokenizer)
            update_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i])
            
    # Clean up to prevent memory issues
    del batch_layer_outputs
    
    # Save checkpoints periodically
    if batch_idx % save_every == 0:
        # Save updated responses with annotations
        with open(responses_json_path, "w") as f:
            json.dump(responses_data, f, indent=2)
        # Save current state of mean vectors
        save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
        torch.save(save_dict, save_path)

    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Save final results
with open(responses_json_path, "w") as f:
    json.dump(responses_data, f, indent=2)
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)

# Compute and save emotional feature vectors using the new approach
print("Computing emotional feature vectors...")
feature_vectors = compute_emotional_feature_vectors(save_dict)
if feature_vectors:
    feature_vectors_path = f"results/vars/feature_vectors_{model_name.split('/')[-1].lower()}.pt"
    torch.save(feature_vectors, feature_vectors_path)
    print(f"Saved feature vectors to {feature_vectors_path}")
    print(f"Available feature vectors: {list(feature_vectors.keys())}")
else:
    print("No feature vectors computed")

print("Saved final annotations and vectors")