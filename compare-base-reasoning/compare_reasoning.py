# %%
import dotenv
dotenv.load_dotenv("../.env")

import argparse
import json
import random
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import torch

from utils import chat, steering_config, process_batch_annotations
from messages import messages

# Model Configuration
MODEL_CONFIG = {
    # API Models: model_id to display name mapping
    'API_MODELS': {
        'gpt-4o': 'GPT-4o',
        'claude-3-opus': 'Claude-3-Opus',
        'claude-3-7-sonnet': 'Claude-3-7-Sonnet',
        'gemini-2-0-think': 'Gemini-2-0-Think',
        'gemini-2-0-flash': 'Gemini-2-0-Flash',
        'deepseek-v3': 'DeepSeek-V3',
        'deepseek-r1': 'DeepSeek-R1',
        'deepseek/deepseek-r1-distill-llama-8b': 'DeepSeek-R1-Llama-8B',
        'deepseek/deepseek-r1-distill-llama-70b': 'DeepSeek-R1-Llama-70B',
        'deepseek/deepseek-r1-distill-qwen-1.5b': 'DeepSeek-R1-Qwen-1.5B',
        'deepseek/deepseek-r1-distill-qwen-14b': 'DeepSeek-R1-Qwen-14B',
        'deepseek/deepseek-r1-distill-qwen-32b': 'DeepSeek-R1-Qwen-32B',
        'meta-llama/llama-3.1-8b-instruct': 'Llama-3.1-8B',
        'meta-llama/llama-3.3-70b-instruct': 'Llama-3.3-70B',
    },
    
    # Local Models: model_id to display name mapping
    'LOCAL_MODELS': {
        'Qwen/Qwen2.5-14B-Instruct': 'Qwen-2.5-14B',
        'Qwen/Qwen2.5-Math-1.5B': 'Qwen-2.5-Math-1.5B',
        'Qwen/Qwen2.5-32B-Instruct': 'Qwen-2.5-32B',
    },
    
    # Thinking Models (for visualization grouping)
    'THINKING_MODELS': [
        'deepseek-r1-distill-llama-8b',
        'deepseek-r1-distill-llama-70b',
        'deepseek-r1-distill-qwen-1.5b',
        'deepseek-r1-distill-qwen-14b',
        'deepseek-r1-distill-qwen-32b',
        'claude-3-7-sonnet',
        'gemini-2-0-think',
        'deepseek-r1'
    ]
}

def get_model_display_name(model_id):
    """Convert model ID to display name using configuration"""
    # Check API models first
    if model_id in MODEL_CONFIG['API_MODELS']:
        return MODEL_CONFIG['API_MODELS'][model_id]
    
    # Check local models
    for local_id, display_name in MODEL_CONFIG['LOCAL_MODELS'].items():
        if local_id in model_id:
            return display_name
    
    # Default case: format the model ID
    return model_id.title()

def is_api_model(model_name):
    """Check if the model is an API model"""
    return model_name in MODEL_CONFIG['API_MODELS']

def is_thinking_model(model_name):
    """Check if the model is a thinking model"""
    # Convert model_name to lowercase for case-insensitive comparison
    model_name = model_name.lower()
    
    return model_name in MODEL_CONFIG['THINKING_MODELS']

def is_local_model(model_name):
    """Check if the model is a local model"""
    return model_name in MODEL_CONFIG['LOCAL_MODELS']

def extract_thinking_process(response_text):
    """Extracts the thinking process from between <think> and </think> tags."""
    think_start = 0
    think_end = len(response_text)
    if '<think>' in response_text:
        think_start = response_text.index('<think>') + len('<think>')
    if '</think>' in response_text:
        think_end = response_text.index('</think>')
    return response_text[think_start:think_end].strip()

# Parse arguments
parser = argparse.ArgumentParser(description="Compare reasoning abilities between models")
parser.add_argument("--model", type=str, default="gemini-2-0-think", 
                    help="Model to evaluate (e.g., 'gpt-4o', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')")
parser.add_argument("--n_examples", type=int, default=10, 
                    help="Number of examples to use for evaluation")
parser.add_argument("--compute_from_json", action="store_true", 
                    help="Recompute scores from existing json instead of generating new responses")
parser.add_argument("--re_compute_scores", action="store_true", 
                    help="Recompute scores from existing json instead of generating new responses")
parser.add_argument("--re_annotate_responses", action="store_true", 
                    help="Re-annotate responses with new annotations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_tokens", type=int, default=100, help="Number of max tokens")
parser.add_argument("--skip_viz", action="store_true", help="Skip visualization at the end")
parser.add_argument("--ignore-common-labels", action="store_true", help="Ignore initializing and deduction labels")
args, _ = parser.parse_known_args()

# %%
def get_label_counts(thinking_process, labels, existing_annotated_response=None):
    if existing_annotated_response is None:
        # Get annotated version using chat function
        annotated_response = process_batch_annotations([thinking_process])[0]
    else:
        annotated_response = existing_annotated_response
    
    # Initialize token counts for each label
    label_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    # Get tokens for the entire thinking process
    total = 0
    
    # Count tokens for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in labels:
            # Count tokens in this section
            label_counts[label] += 1
            total += 1
    
    return label_counts, annotated_response

def process_chat_response(message, model_name, model, tokenizer, labels):
    """Process a single message through chat function or model"""
    if is_api_model(model_name) and not is_thinking_model(model_name):
        # API model case (OpenAI models)
        response = chat(f"""Please answer the following question:

Question:
{message["content"]}

Please format your response like this:
<think>
...
</think>
[Your answer here]
""",
        model=model_name,
        max_tokens=args.max_tokens
        )

        print(response)

    elif is_api_model(model_name) and is_thinking_model(model_name):
        response = chat(message["content"], model=model_name, max_tokens=args.max_tokens)
        print(response)

    elif is_local_model(model_name):       
        input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
                        
        with model.generate(
            {
                "input_ids": input_ids, 
                "attention_mask": (input_ids != tokenizer.pad_token_id).long()
            },
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ) as tracer:
            outputs = model.generator.output.save()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract thinking process
    thinking_process = extract_thinking_process(response)
    
    label_counts, annotated_response = get_label_counts(thinking_process, labels, existing_annotated_response=None)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_counts": label_counts,
        "annotated_response": annotated_response
    }

def plot_comparison(results_dict, labels):
    """Plot comparison between multiple models' results"""
    os.makedirs('results/figures', exist_ok=True)
    print(labels)
    
    # Get model names and prepare data
    model_names = list(results_dict.keys())
    print(model_names)
    means_dict = {}
    
    # Separate models into thinking and non-thinking groups
    thinking_names = [name for name in model_names if is_thinking_model(name)]
    non_thinking_names = [name for name in model_names if not is_thinking_model(name)]
    
    # Calculate means for each model and label
    for model_name in model_names:
        means_dict[model_name] = []
        for label in labels:
            label_counts = [ex["label_counts"].get(label, 0) for ex in results_dict[model_name]]
            means_dict[model_name].append(np.mean(label_counts))
    
    # Calculate average performance across all labels for each model
    model_avg_performance = {
        model_name: np.mean(means_dict[model_name]) 
        for model_name in model_names
    }
    
    # Sort models within each group by their average performance
    thinking_names = sorted(thinking_names, key=lambda x: model_avg_performance[x], reverse=True)
    non_thinking_names = sorted(non_thinking_names, key=lambda x: model_avg_performance[x], reverse=True)
    
    # Create bar plot with wider aspect ratio
    plt.style.use('seaborn-v0_8-paper')  # Use a clean scientific style
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))
    
    # Enhanced black box around the plot with slightly thicker lines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Define color palettes for thinking and non-thinking models with more distinct shades
    # Colors are ordered from darkest to brightest
    thinking_colors = [
        '#1565C0',  # Darkest blue
        '#1976D2',  # Dark blue
        '#1E88E5',  # Medium blue
        '#64B5F6',  # Lightest blue
        '#BBDEFB'   # Very light blue
    ]
    non_thinking_colors = [
        '#E65100',  # Darkest orange
        '#F57C00',  # Dark orange
        '#FF9800',  # Medium orange
        '#FFA726',  # Lightest orange
        '#FFE0B2'   # Very light orange
    ]
    
    # Calculate width based on number of models
    width = min(0.35, 0.8 / len(model_names))
    
    # Plot bars for each model, grouped by thinking/non-thinking
    bars_list = []
    
    # Add a small gap between groups if both types of models are present
    n_thinking = len(thinking_names)
    n_non_thinking = len(non_thinking_names)
    gap = width * 0.5 if n_thinking > 0 and n_non_thinking > 0 else 0
    
    # First plot thinking models
    for i, model_name in enumerate(thinking_names):
        # Use darker shades for higher performing models
        color_idx = i if i < len(thinking_colors) else len(thinking_colors) - 1
        bars = ax.bar(x + width * i, means_dict[model_name], width, 
                     label=model_name,
                     color=thinking_colors[color_idx], 
                     alpha=0.85, 
                     edgecolor='black', 
                     linewidth=1)
        bars_list.append(bars)
    
    # Then plot non-thinking models
    for i, model_name in enumerate(non_thinking_names):
        # Use darker shades for higher performing models
        color_idx = i if i < len(non_thinking_colors) else len(non_thinking_colors) - 1
        bars = ax.bar(x + width * (i + n_thinking) + gap, means_dict[model_name], width, 
                     label=model_name,
                     color=non_thinking_colors[color_idx], 
                     alpha=0.85, 
                     edgecolor='black', 
                     linewidth=1)
        bars_list.append(bars)
    
    # Add mean text over each group of bars
    if n_thinking > 0:
        for i in range(len(labels)):
            thinking_means_for_label = [means_dict[name][i] for name in thinking_names]
            group_mean = np.mean(thinking_means_for_label)
            group_center_x = x[i] + width * (n_thinking - 1) / 2
            max_bar_height = max(thinking_means_for_label)
            ax.text(group_center_x, max_bar_height + 0.02, f"Mean: {group_mean:.1f}",
                    ha='center', va='bottom', fontsize=14, color='black')

    if n_non_thinking > 0:
        for i in range(len(labels)):
            non_thinking_means_for_label = [means_dict[name][i] for name in non_thinking_names]
            group_mean = np.mean(non_thinking_means_for_label)
            group_center_x = x[i] + width * (n_thinking + (n_non_thinking - 1) / 2) + gap
            max_bar_height = max(non_thinking_means_for_label)
            ax.text(group_center_x, max_bar_height + 0.02, f"Mean: {group_mean:.1f}",
                    ha='center', va='bottom', fontsize=14, color='black')
    
    # Improve grid and ticks
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)  # Put grid below bars
    
    # Set y-axis limit with more headroom and add label
    ymax = max([max(means) for means in means_dict.values()]) if means_dict else 1
    ax.set_ylim(0, ymax * 1.75)  # Add 75% headroom
    ax.set_ylabel('Average Sentence Count Per Response', fontsize=16)  # Add y-axis label
    ax.set_xlabel("Behavioral patterns", fontsize=16)
    
    # Format x-axis labels more professionally
    tick_pos = x + (n_thinking * width + n_non_thinking * width + gap) / 2 - width / 2
    ax.set_xticks(tick_pos)
    formatted_labels = [label.replace('-', ' ').title() for label in labels]
    formatted_labels = [label.replace(' ', '\n') for label in formatted_labels]
    ax.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add vertical lines to separate thinking and non-thinking groups
    for label_idx in x:
        group_separator = label_idx + width * len(thinking_names) + gap/2
        ax.axvline(x=group_separator, color='gray', linestyle='--', alpha=0.3, zorder=0)
    
    # Enhance legend
    ax.legend(fontsize=16, frameon=True, framealpha=1, 
             edgecolor='black', bbox_to_anchor=(1, 1.02), 
             loc='upper right', ncol=2)
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig(f'results/figures/reasoning_comparison_all_models.pdf', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

def plot_avg_sentences_per_response(results_dict):
    """Plots the average number of sentences per response for each model."""
    os.makedirs('results/figures', exist_ok=True)
    
    model_names = list(results_dict.keys())
    
    # Separate models into thinking and non-thinking groups
    thinking_names = [name for name in model_names if is_thinking_model(name)]
    non_thinking_names = [name for name in model_names if not is_thinking_model(name)]

    avg_sentences = {}
    std_sentences = {}
    for model_name in model_names:
        total_sentences_per_response = []
        for result in results_dict[model_name]:
            # Sum of sentence counts for all labels in one response
            num_sentences = sum(result['label_counts'].values())
            total_sentences_per_response.append(num_sentences)
        
        # Average over all responses for the model
        if total_sentences_per_response:
            avg_sentences[model_name] = np.mean(total_sentences_per_response)
            std_sentences[model_name] = np.std(total_sentences_per_response)
        else:
            avg_sentences[model_name] = 0
            std_sentences[model_name] = 0
    
    # Sort models based on average sentence count
    sorted_thinking_names = sorted(thinking_names, key=lambda x: avg_sentences[x], reverse=True)
    sorted_non_thinking_names = sorted(non_thinking_names, key=lambda x: avg_sentences[x], reverse=True)
    
    sorted_model_names = sorted_thinking_names + sorted_non_thinking_names
    
    # Create the plot
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    thinking_colors = [
        '#1565C0', '#1976D2', '#1E88E5', '#64B5F6', '#BBDEFB'
    ]
    non_thinking_colors = [
        '#E65100', '#F57C00', '#FF9800', '#FFA726', '#FFE0B2'
    ]

    colors = []
    # Assign colors to thinking models
    for i, model_name in enumerate(sorted_thinking_names):
        color_idx = i if i < len(thinking_colors) else len(thinking_colors) - 1
        colors.append(thinking_colors[color_idx])
    
    # Assign colors to non-thinking models
    for i, model_name in enumerate(sorted_non_thinking_names):
        color_idx = i if i < len(non_thinking_colors) else len(non_thinking_colors) - 1
        colors.append(non_thinking_colors[color_idx])

    y_pos = np.arange(len(sorted_model_names))
    performance = [avg_sentences[name] for name in sorted_model_names]
    errors = [std_sentences[name] for name in sorted_model_names]

    bars = ax.barh(y_pos, performance, xerr=errors, align='center', color=colors, edgecolor='black', linewidth=1, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_model_names, fontsize=14)
    ax.invert_yaxis()  # models with highest values at the top
    ax.set_xlabel('Average Sentences', fontsize=16)
    ax.set_title('Average Number of Sentences in Thinking Process', fontsize=18)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                    ha='left', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('results/figures/avg_sentences_per_response.pdf', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# %% Parameters
n_examples = args.n_examples
random.seed(args.seed)
model_name = args.model
compute_from_json = args.compute_from_json
re_compute_scores = args.re_compute_scores
re_annotate_responses = args.re_annotate_responses

# Get a shorter model_id for file naming
model_id = model_name.split('/')[-1].lower()

labels = list(list(steering_config.values())[0].keys())
labels.append('initializing')
labels.append('deduction')

if args.ignore_common_labels:
    labels.remove('initializing')
    labels.remove('deduction')

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %% Load model and evaluate
model = None
tokenizer = None

results = []

# %%
if compute_from_json:
    # Load existing results and recompute scores
    print(f"Loading existing results for {model_name}...")
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'r') as f:
        results = json.load(f)
    
    # Re-compute label fractions from loaded results
    if re_compute_scores:
        print("Re-computing label counts for all loaded responses...")
        for result in tqdm(results, desc="Re-computing scores from JSON"):
            thinking_process = result.get('thinking_process', '')
            if thinking_process == '' or thinking_process == 'None':
                thinking_process = extract_thinking_process(result.get('response', ''))
                result['thinking_process'] = thinking_process

            assert thinking_process != '', f"**ERROR** No thinking process found for {result['response']}"
                
            label_counts, annotated_response = get_label_counts(
                thinking_process, 
                labels,
                existing_annotated_response=result.get('annotated_response', None) if not re_annotate_responses else None
            )
            if 'label_fractions' in result:
                del result['label_fractions']
            result['label_counts'] = label_counts
            result['annotated_response'] = annotated_response
        # Save updated results
        with open(f'results/vars/reasoning_comparison_{model_id}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Re-computed {len(results)} examples for {model_name}")
else:
    # Run new evaluation
    print(f"Running evaluation for {model_name}...")

    if not is_api_model(model_name):
        # Load model using the utils function
        import utils
        print(f"Loading model {model_name}...")
        model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name)
    
    # Randomly sample evaluation examples
    eval_indices = random.sample(range(len(messages)), n_examples)
    selected_messages = [messages[i] for i in eval_indices]
    
    # Process responses
    for message in tqdm(selected_messages, desc=f"Processing examples for {model_name}"):
        # Process response
        result = process_chat_response(message, model_name, model, tokenizer, labels)
        results.append(result)
    
    # Save results
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up model to free memory
    if model is not None:
        del model
        torch.cuda.empty_cache()

# %% Generate visualization with all available models
if not args.skip_viz:
    # Load results for all models
    all_results = {}
    result_files = glob.glob('results/vars/reasoning_comparison_*.json')

    # Filter Llama 8B and Qwen Math 1.5B: the responses are too messy
    result_files = [file for file in result_files if 'llama-8b' not in file and 'llama-3.1-8b' not in file and '1.5b' not in file]

    print(f"Found {len(result_files)} model results for visualization")
    
    for file_path in result_files:
        model_id = os.path.basename(file_path).replace('reasoning_comparison_', '').replace('.json', '')
        display_name = get_model_display_name(model_id)
        
        with open(file_path, 'r') as f:
            all_results[display_name] = json.load(f)
    
    # Generate visualization with all models
    if all_results:
        plot_comparison(all_results, labels)
        plot_avg_sentences_per_response(all_results)
    else:
        print("No results found for visualization")

# %%
