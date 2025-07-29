import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
import random
import json
import math
import gc
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
import utils
from utils import process_saved_responses_batch, get_label_positions
from emotional_annotation import process_emotional_batch_annotations, get_emotional_annotation_labels

# Parse arguments
parser = argparse.ArgumentParser(description="Train emotional steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to train emotional steering vectors for")
parser.add_argument("--emotional_data_path", type=str, default=None,
                    help="Path to emotional examples JSON file")
parser.add_argument("--target_emotion", type=str, default="depressive",
                    help="Primary emotion to focus on (depressive, anxious, hopeful)")
parser.add_argument("--n_samples", type=int, default=100,
                    help="Number of samples to process")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for processing")
parser.add_argument("--save_every", type=int, default=5,
                    help="Save checkpoints every n batches")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args, _ = parser.parse_known_args()

def update_emotional_mean_vectors(mean_vectors, layer_outputs, label_positions, index, target_emotion):
    """Update mean vectors for emotional categories"""
    
    # Calculate overall emotional section boundaries (all emotions combined)
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update overall emotional mean
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN detected in overall mean at index {index}")
    
    # Update individual emotional labels with enhanced focus on target emotion
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            
            # Use more tokens for target emotion, fewer for others
            if label == target_emotion:
                token_window = min(15, end - start)  # Larger window for target emotion
            else:
                token_window = min(8, end - start)   # Smaller window for other emotions
            
            vectors = layer_outputs[:, start-1:min(end-1, start+token_window)].mean(dim=1)
            
            if torch.isnan(vectors).any():
                print(f"NaN detected in {label} vectors at index {index}")
                continue
            
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            
            # Give extra weight to target emotion examples
            weight = 1.5 if label == target_emotion else 1.0
            weighted_vectors = vectors * weight
            
            mean_vectors[label]['mean'] = current_mean + (weighted_vectors - current_mean) / (current_count + weight)
            mean_vectors[label]['count'] += weight

def process_emotional_examples(emotional_data_path, n_samples, target_emotion):
    """Process emotional examples from the generated data"""
    
    print(f"Loading emotional examples from {emotional_data_path}")
    with open(emotional_data_path, 'r') as f:
        emotional_data = json.load(f)
    
    # Filter and prepare examples
    processed_examples = []
    
    for example in emotional_data[:n_samples]:
        # Extract thinking process from the response
        response = example['response']
        
        # Find thinking section
        if '<think>' in response and '</think>' in response:
            think_start = response.index('<think>') + len('<think>')
            think_end = response.index('</think>')
            thinking_process = response[think_start:think_end].strip()
        else:
            thinking_process = response
        
        processed_examples.append({
            "original_prompt": example['prompt'],
            "full_response": response,
            "thinking_process": thinking_process,
            "emotion_category": example['emotion_category'],
            "annotated_thinking": ""  # Will be filled during processing
        })
    
    return processed_examples

def main():
    model_name = args.model
    target_emotion = args.target_emotion
    
    # Create directories
    os.makedirs('results/vars', exist_ok=True)
    
    # Set paths
    if args.emotional_data_path:
        emotional_data_path = args.emotional_data_path
    else:
        emotional_data_path = f"results/vars/emotional_examples_{target_emotion}.json"
    
    if not os.path.exists(emotional_data_path):
        raise FileNotFoundError(f"Emotional data not found at {emotional_data_path}. Generate examples first.")
    
    save_path = f"results/vars/emotional_vectors_{target_emotion}_{model_name.split('/')[-1].lower()}.pt"
    
    # Load model
    print(f"Loading model {model_name}...")
    model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name, load_in_8bit=args.load_in_8bit)
    
    # Initialize mean vectors for emotional categories
    emotional_labels = get_emotional_annotation_labels()
    mean_vectors = defaultdict(lambda: {
        'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
        'count': 0
    })
    
    # Add overall category
    mean_vectors['overall'] = {
        'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
        'count': 0
    }
    
    # Process emotional examples
    emotional_examples = process_emotional_examples(emotional_data_path, args.n_samples, target_emotion)
    
    random.seed(args.seed)
    random.shuffle(emotional_examples)
    
    # Process in batches
    num_batches = math.ceil(len(emotional_examples) / args.batch_size)
    
    print(f"Processing {len(emotional_examples)} emotional examples in {num_batches} batches")
    print(f"Target emotion: {target_emotion}")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing emotional examples"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(emotional_examples))
        
        batch_examples = emotional_examples[start_idx:end_idx]
        thinking_processes = [example["thinking_process"] for example in batch_examples]
        batch_full_responses = [example["full_response"] for example in batch_examples]
        batch_indices = list(range(start_idx, end_idx))
        
        # Generate emotional annotations
        annotated_responses = process_emotional_batch_annotations(thinking_processes, target_emotion)
        
        # Update annotation fields
        for i, (example, annotated) in enumerate(zip(batch_examples, annotated_responses)):
            emotional_examples[start_idx + i]["annotated_thinking"] = annotated
        
        # Process responses to get layer activations
        batch_layer_outputs = process_saved_responses_batch(batch_full_responses, tokenizer, model)
        
        # Update emotional vectors
        for i, (example, layer_outputs) in enumerate(zip(batch_examples, batch_layer_outputs)):
            if annotated_responses[i]:  # Use the new emotional annotations
                label_positions = get_label_positions(annotated_responses[i], example["full_response"], tokenizer)
                if label_positions:  # Only process if we found emotional patterns
                    update_emotional_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i], target_emotion)
        
        del batch_layer_outputs
        
        # Save checkpoints
        if batch_idx % args.save_every == 0 or batch_idx == num_batches - 1:
            # Save annotated examples
            examples_path = f"results/vars/emotional_examples_{target_emotion}_annotated.json"
            with open(examples_path, "w") as f:
                json.dump(emotional_examples, f, indent=2)
            
            # Save vectors
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)
            
            print(f"Saved checkpoint after batch {batch_idx+1}/{num_batches}")
            
            # Print statistics
            print("Current vector counts:")
            for label, vectors in mean_vectors.items():
                print(f"  {label}: {vectors['count']:.1f} examples")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save final results
    examples_path = f"results/vars/emotional_examples_{target_emotion}_annotated.json"
    with open(examples_path, "w") as f:
        json.dump(emotional_examples, f, indent=2)
    
    save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
    torch.save(save_dict, save_path)
    
    print(f"\nFinal emotional vector training completed!")
    print(f"Saved {len(emotional_examples)} annotated examples to {examples_path}")
    print(f"Saved emotional vectors to {save_path}")
    
    print("\nFinal vector statistics:")
    for label, vectors in mean_vectors.items():
        if vectors['count'] > 0:
            print(f"  {label}: {vectors['count']:.1f} examples")

if __name__ == "__main__":
    main()