import dotenv
dotenv.load_dotenv("../.env")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import time
import random
import torch.nn as nn
import openai
import anthropic
import os
from openai import OpenAI
import json
import re
import numpy as np
import traceback

def chat(prompt, model="gpt-4o-mini", max_tokens=15000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet", "claude-3.5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    else:
        raise ValueError(f"Model {model} is not supported. Please use OpenAI (gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo) or Anthropic (claude-3.5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-haiku) models.")

    # Ensure max_tokens stays within a safe range for the provider to prevent
    # `BadRequestError: max_tokens is too large` exceptions.
    if max_tokens is None:
        max_tokens = 8000
    # OpenAI currently rejects completions larger than 16,384 tokens (completion tokens, not context).
    # We conservatively cap the requested value to this limit.
    max_tokens = min(max_tokens, 16384)

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229",
                    "claude-3-haiku": "claude-3-haiku-20240307",
                    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
                    "claude-3.5-haiku": "claude-3-5-haiku-20241022"
                }

                response = client.messages.create(
                    model=model_mapping.get(model, model),
                    temperature=0.1,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=max_tokens
                )

                return response.content[0].text
            
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            time.sleep(20)

    return None


def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def get_label_positions(annotated_thinking, response_text, tokenizer):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [category-name] text [end-section]
    pattern = r'\["(\S+?)"\](.*?)\["end-section"\]'
    matches = list(re.finditer(pattern, annotated_thinking, re.DOTALL))
    
    # Create character to token mapping once
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    
    for match in matches:
        label = match.group(1).strip()
        text = match.group(2).strip()
        
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        text_pos = response_text.find(text)
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end))
    
    return label_positions

def load_model_and_vectors(device=None, load_in_8bit=False, compute_features=True, normalize_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", base_model_name=None):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        device (str): Device to load model on. If None, auto-detects available device
        load_in_8bit (bool): If True, load the model in 8-bit mode
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        normalize_features (bool): If True, normalize the feature vectors
        return_steering_vector_set (bool): If True, return the steering vector set
        model_name (str): Name/path of the model to load
        base_model_name (str): Name/path of the base model to load
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Use float32 for CPU and MPS, bfloat16 for CUDA
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=dtype)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if base_model_name is not None:
        base_model = LanguageModel(base_model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
        base_model.generation_config.temperature=None
        base_model.generation_config.top_p=None
        base_model.generation_config.do_sample=False
        
        base_tokenizer = base_model.tokenizer

        if "llama" in base_model_name.lower():
            base_tokenizer.pad_token_id = base_tokenizer.finetune_right_pad_id
            base_tokenizer.pad_token = base_tokenizer.finetune_right_pad
            base_tokenizer.padding_side = "right"
        else:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
            base_tokenizer.pad_token = base_tokenizer.eos_token
            base_tokenizer.padding_side = "left"

    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # go into directory of this file
    vector_path = f"../train-steering-vectors/results/vars/mean_vectors_{model_id}.pt"
    if os.path.exists(vector_path):
        mean_vectors_dict = torch.load(vector_path)

        if compute_features:
            # Compute feature vectors by subtracting overall mean
            feature_vectors = {}
            feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']
            
            cognitive_labels = ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]
            emotional_labels = ["depressive-thinking", "anxious-thinking", "negative-attribution", "pessimistic-projection"]
            
            # For emotional labels, use normal-thinking as baseline if available, otherwise use overall
            baseline_mean = mean_vectors_dict.get("normal-thinking", {}).get("mean", mean_vectors_dict["overall"]['mean'])
            
            for label in cognitive_labels:
                if label != 'overall' and label in mean_vectors_dict:
                    feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']
            
            for label in emotional_labels:
                if label != 'overall' and label in mean_vectors_dict:
                    feature_vectors[label] = mean_vectors_dict[label]['mean'] - baseline_mean

                if normalize_features:
                    for label in feature_vectors:
                        for layer in range(model.config.num_hidden_layers):
                            feature_vectors[label][layer] = feature_vectors[label][layer] * (feature_vectors["overall"][layer].norm() / feature_vectors[label][layer].norm())
    else:
        print(f"No mean vectors found for {model_name}")
        mean_vectors_dict = {}
        feature_vectors = {}

    if base_model_name is not None and compute_features:
        return model, tokenizer, base_model, base_tokenizer, feature_vectors
    elif base_model_name is not None and not compute_features:
        return model, tokenizer, base_model, base_tokenizer, mean_vectors_dict
    elif base_model_name is None and compute_features:
        return model, tokenizer, feature_vectors
    else:
        return model, tokenizer, mean_vectors_dict

def custom_generate_steering(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steering_config, steer_positive=False):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors (flat structure: feature_vectors[label] = tensor)
        steer_positive: If True, steer towards the label, if False steer away
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if feature_vectors is not None and label in feature_vectors:       
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]

            # Get device and dtype from model
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            # FIXED: Use flat structure - feature_vectors[label] is the tensor directly
            # Extract the vector for the specific layer from the full feature vector
            feature_vector_full = feature_vectors[label].to(model_device).to(model_dtype)
            
            # Check if we have a layer-specific vector or need to extract from full vector
            if len(feature_vector_full.shape) == 2:  # [num_layers, hidden_size]
                feature_vector = feature_vector_full[vector_layer]  # Extract specific layer
            else:  # [hidden_size] - already layer-specific
                feature_vector = feature_vector_full
            
            # Define helper function for vector adjustment to avoid scoping issues
            def adjust_feature_vector(feature_vec, target_hidden_size):
                if feature_vec.shape[-1] != target_hidden_size:
                    # Trim or pad feature vector to match hidden size
                    if feature_vec.shape[-1] > target_hidden_size:
                        return feature_vec[:target_hidden_size]
                    else:
                        # Pad with zeros if feature vector is smaller
                        pad_size = target_hidden_size - feature_vec.shape[-1]
                        padding = torch.zeros(pad_size, device=feature_vec.device, dtype=feature_vec.dtype)
                        return torch.cat([feature_vec, padding])
                return feature_vec
            
            if steer_positive:
                for layer_idx in pos_layers:         
                    # Get the current layer output shape to match dimensions
                    layer_output = model.model.layers[layer_idx].output[0]
                    # Fix: Access shape dimensions individually to avoid nnsight proxy unpacking issue
                    batch_size = layer_output.shape[0]
                    seq_len = layer_output.shape[1] 
                    hidden_size = layer_output.shape[2]
                    
                    # Ensure feature vector matches the hidden dimension
                    adjusted_vector = adjust_feature_vector(feature_vector, hidden_size)
                    
                    # Reshape to match layer output: [1, 1, hidden_size]
                    steering_vector = adjusted_vector.unsqueeze(0).unsqueeze(0)
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * steering_vector
            else:
                for layer_idx in neg_layers:         
                    # Get the current layer output shape to match dimensions
                    layer_output = model.model.layers[layer_idx].output[0]
                    # Fix: Access shape dimensions individually to avoid nnsight proxy unpacking issue
                    batch_size = layer_output.shape[0]
                    seq_len = layer_output.shape[1] 
                    hidden_size = layer_output.shape[2]
                    
                    # Ensure feature vector matches the hidden dimension
                    adjusted_vector = adjust_feature_vector(feature_vector, hidden_size)
                    
                    # Reshape to match layer output: [1, 1, hidden_size]
                    steering_vector = adjusted_vector.unsqueeze(0).unsqueeze(0)
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * steering_vector
        
        outputs = model.generator.output.save()
                    
    return outputs


def process_batch_annotations(thinking_processes, include_emotional=False, annotation_model=None):
    """Annotate a batch of reasoning chains using the cognitive and optional emotional reasoning framework."""
    # Use environment variable or default model
    if annotation_model is None:
        annotation_model = os.getenv('ANNOTATION_MODEL', 'gpt-4o-mini')
    
    annotated_responses = []
    for thinking in thinking_processes:
        if include_emotional:
            prompt = f"""
            Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

            Available labels:

            Emotional Labels:
            6. depressive-thinking -> Self-critical thoughts, hopelessness, catastrophizing, negative self-assessment, minimizing achievements.
            7. anxious-thinking -> Worry, rumination, worst-case scenarios, hypervigilance about problems, catastrophic predictions.
            8. negative-attribution -> Attributing failures to internal/permanent causes, minimizing successes, dismissing positive feedback.
            9. pessimistic-projection -> Predicting negative outcomes, focusing on potential failures, anticipating disappointment.
            10. normal-thinking -> Balanced, constructive, and realistic reasoning patterns that show healthy problem-solving, emotional regulation, and perspective-taking.

            The reasoning chain to analyze:
            {thinking}

            Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
            """
        else:
            prompt = f"""
            Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

            Available labels:
            0. initializing -> The model is rephrasing the given task and states initial thoughts.
            1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
            2. adding-knowledge -> The model is enriching the current approach with recalled facts.
            3. example-testing -> The model generates examples to test its current approach.
            4. uncertainty-estimation -> The model is stating its own uncertainty.
            5. backtracking -> The model decides to change its approach.

            The reasoning chain to analyze:
            {thinking}

            Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
            """
        
        annotated_response = chat(prompt, model=annotation_model)
        annotated_responses.append(annotated_response)
    
    return annotated_responses

def get_batched_message_ids(tokenizer, messages_list, device=None):
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # First get the max length by encoding each message individually
    max_token_length = max([len(tokenizer.encode(msg, return_tensors="pt")[0]) for msg in messages_list])
    input_ids = torch.cat([
        tokenizer.encode(msg, padding="max_length", max_length=max_token_length, return_tensors="pt").to(device) 
        for msg in messages_list
    ])

    return input_ids

def process_saved_responses_batch(responses_list, tokenizer, model):
    """Get layer activations for a batch of saved responses without generation"""
    # Get device from model
    device = next(model.parameters()).device
    tokenized_responses = get_batched_message_ids(tokenizer, responses_list, device.type)
    
    # Process the inputs through the model to get activations
    layer_outputs = []
    with model.trace(
        {
            "input_ids": tokenized_responses, 
            "attention_mask": (tokenized_responses != tokenizer.pad_token_id).long()
        }
    ) as tracer:
        
        # Capture layer outputs
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [x.value.cpu().detach().to(torch.float32) for x in layer_outputs]

    batch_layer_outputs = []
    
    for batch_idx in range(len(responses_list)):
        # get length of padding tokens
        attention_mask = (tokenized_responses[batch_idx] != tokenizer.pad_token_id).long()
        padding_length = (attention_mask.squeeze() == 0).sum().item()
        
        # Slice out just the non-padded activations for this example across all layers
        example_outputs = torch.stack([
            layer_output[batch_idx][padding_length:] 
            for layer_output in layer_outputs
        ])
        
        batch_layer_outputs.append(example_outputs)
    
    return batch_layer_outputs

steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        # Cognitive reasoning categories
        "backtracking": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 15, "pos_layers": [15], "neg_layers": [15], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
        "initializing": {"vector_layer": 16, "pos_layers": [16], "neg_layers": [16], "pos_coefficient": 1, "neg_coefficient": 1},
        "deduction": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1, "neg_coefficient": 1},
        # Emotional reasoning categories
        "depressive-thinking": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1.5, "neg_coefficient": 1.0},
        "anxious-thinking": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1.2, "neg_coefficient": 1.0},
        "negative-attribution": {"vector_layer": 16, "pos_layers": [16], "neg_layers": [16], "pos_coefficient": 1.3, "neg_coefficient": 1.0},
        "pessimistic-projection": {"vector_layer": 19, "pos_layers": [19], "neg_layers": [19], "pos_coefficient": 1.4, "neg_coefficient": 1.0},
        "normal-thinking": {"vector_layer": 15, "pos_layers": [15], "neg_layers": [15], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
        "baseline": {"vector_layer": 15, "pos_layers": [15], "neg_layers": [15], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        # Cognitive reasoning categories
        "backtracking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "initializing": {"vector_layer": 11, "pos_layers": [11], "neg_layers": [11], "pos_coefficient": 1, "neg_coefficient": 1},
        "deduction": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        # Emotional reasoning categories
        "depressive-thinking": {"vector_layer": 13, "pos_layers": [13], "neg_layers": [13], "pos_coefficient": 1, "neg_coefficient": 1.0},
        "anxious-thinking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1.0},
        "negative-attribution": {"vector_layer": 11, "pos_layers": [11], "neg_layers": [11], "pos_coefficient": 1, "neg_coefficient": 1.0},
        "pessimistic-projection": {"vector_layer": 14, "pos_layers": [14], "neg_layers": [14], "pos_coefficient": 1, "neg_coefficient": 1.0},
        "normal-thinking": {"vector_layer": 10, "pos_layers": [10], "neg_layers": [10], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
        "baseline": {"vector_layer": 10, "pos_layers": [10], "neg_layers": [10], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        # Cognitive reasoning categories
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 24, "pos_layers": [24], "neg_layers": [24], "pos_coefficient": 1, "neg_coefficient": 1},
        "initializing": {"vector_layer": 28, "pos_layers": [28], "neg_layers": [28], "pos_coefficient": 1, "neg_coefficient": 1},
        "deduction": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        # Emotional reasoning categories
        "depressive-thinking": {"vector_layer": 30, "pos_layers": [30], "neg_layers": [30], "pos_coefficient": 1.5, "neg_coefficient": 1.0},
        "anxious-thinking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1.2, "neg_coefficient": 1.0},
        "negative-attribution": {"vector_layer": 28, "pos_layers": [28], "neg_layers": [28], "pos_coefficient": 1.3, "neg_coefficient": 1.0},
        "pessimistic-projection": {"vector_layer": 31, "pos_layers": [31], "neg_layers": [31], "pos_coefficient": 1.4, "neg_coefficient": 1.0},
        "normal-thinking": {"vector_layer": 27, "pos_layers": [27], "neg_layers": [27], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
        "baseline": {"vector_layer": 27, "pos_layers": [27], "neg_layers": [27], "pos_coefficient": 1.0, "neg_coefficient": 1.0},
    }
}

def analyze_emotional_content(response_text):
    """Analyze response for emotional indicators using keyword-based analysis."""
    
    # Keywords for different emotional patterns
    depressive_keywords = [
        "hopeless", "worthless", "failure", "never", "always fail", "can't do", 
        "inadequate", "inferior", "terrible", "awful", "disaster", "useless",
        "pointless", "doomed", "helpless", "defeated", "broken", "ruined"
    ]
    
    anxious_keywords = [
        "worry", "worried", "anxious", "nervous", "scared", "afraid",
        "what if", "could go wrong", "worst case", "terrible things",
        "catastrophe", "disaster", "panic", "overwhelming", "dread"
    ]
    
    negative_attribution_keywords = [
        "my fault", "I'm bad at", "I always", "I never", "because I'm",
        "due to my", "it's because I", "I'm not good", "I lack", "I'm incapable"
    ]
    
    pessimistic_keywords = [
        "will fail", "won't work", "bound to", "inevitably", "probably won't",
        "likely to fail", "doomed to", "no point", "waste of time", "futile"
    ]
    
    response_lower = response_text.lower()
    
    # Count matches for each category
    depressive_count = sum(1 for keyword in depressive_keywords if keyword in response_lower)
    anxious_count = sum(1 for keyword in anxious_keywords if keyword in response_lower)
    negative_attribution_count = sum(1 for keyword in negative_attribution_keywords if keyword in response_lower)
    pessimistic_count = sum(1 for keyword in pessimistic_keywords if keyword in response_lower)
    
    # Calculate emotional tone scores
    total_words = len(response_text.split())
    
    return {
        "depressive_score": depressive_count / max(total_words, 1) * 100,
        "anxious_score": anxious_count / max(total_words, 1) * 100,
        "negative_attribution_score": negative_attribution_count / max(total_words, 1) * 100,
        "pessimistic_score": pessimistic_count / max(total_words, 1) * 100,
        "total_emotional_score": (depressive_count + anxious_count + negative_attribution_count + pessimistic_count) / max(total_words, 1) * 100,
        "word_count": total_words,
        "depressive_keywords_found": depressive_count,
        "anxious_keywords_found": anxious_count,
        "negative_attribution_keywords_found": negative_attribution_count,
        "pessimistic_keywords_found": pessimistic_count
    }

def generate_and_analyze_emotional(model, tokenizer, message, feature_vectors, steering_config, label, steer_mode="positive", max_new_tokens=1000):
    """
    Generate response with emotional steering and analyze emotional content.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        message: Input message
        feature_vectors: Feature vectors for steering
        steering_config: Steering configuration
        label: Emotional label to steer toward/away from
        steer_mode: "positive" to enhance, "negative" to suppress
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict: Contains response text and emotional analysis, or None if steering fails
    """
    
    try:
        # Check if label exists in both feature_vectors and steering_config
        if label not in feature_vectors:
            print(f"⚠️ Label '{label}' not found in feature_vectors. Available: {list(feature_vectors.keys())}")
            return None
            
        if model.name_or_path not in steering_config:
            print(f"⚠️ Model '{model.name_or_path}' not found in steering_config. Available: {list(steering_config.keys())}")
            return None
            
        if label not in steering_config[model.name_or_path]:
            print(f"⚠️ Label '{label}' not found in steering_config for model '{model.name_or_path}'")
            return None
        
        # Tokenize input
        input_ids = tokenizer.encode(message, return_tensors="pt")
        
        # Generate with steering
        steer_positive = (steer_mode == "positive")
        
        output = custom_generate_steering(
            model=model,
            tokenizer=tokenizer, 
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            label=label,
            feature_vectors=feature_vectors,
            steering_config=steering_config[model.name_or_path],
            steer_positive=steer_positive
        )
        
        # Decode response
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove input from response
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if response_text.startswith(input_text):
            response_text = response_text[len(input_text):].strip()
        
        # Analyze emotional content
        emotional_analysis = analyze_emotional_content(response_text)
        
        return {
            "response": response_text,
            "emotional_analysis": emotional_analysis,
            "steering_label": label,
            "steering_mode": steer_mode,
            "input_message": message
        }
        
    except Exception as e:
        print(f"❌ Error in generate_and_analyze_emotional for label '{label}': {e}")
        import traceback
        traceback.print_exc()
        return None

def emotional_steering_pipeline(model, tokenizer, feature_vectors, steering_config, 
                              messages, target_emotional_direction="depressive-normal", 
                              max_new_tokens=1000, batch_size=4):
    """
    Unified pipeline for emotional steering with depressive-normal dichotomy.
    
    Args:
        model: The language model
        tokenizer: The tokenizer  
        feature_vectors: Feature vectors for steering
        steering_config: Steering configuration
        messages: List of input messages
        target_emotional_direction: "depressive-normal", "anxious-normal", etc.
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for processing
        
    Returns:
        dict: Results including baseline, steered responses, and analysis
    """
    
    # Parse the target direction - handle special cases like "pessimistic-projection"
    if "-" in target_emotional_direction:
        parts = target_emotional_direction.split("-")
        
        # Handle special compound cases like "pessimistic-projection"
        if len(parts) == 3 and parts[1] == "projection":
            negative_label = f"{parts[0]}-{parts[1]}"  # "pessimistic-projection"
            # Special case: if positive label is "baseline", don't add "-thinking"
            if parts[2] == "baseline":
                positive_label = parts[2]
            else:
                positive_label = parts[2] + "-thinking" if not parts[2].endswith("-thinking") else parts[2]
        elif len(parts) == 2:
            negative_label, positive_label = parts
            # Add "-thinking" suffix if not already present and not a special compound label
            if not negative_label.endswith(("-thinking", "-projection", "-attribution")):
                negative_label = negative_label + "-thinking"
            # Special case: if positive label is "baseline", don't add "-thinking"
            if positive_label == "baseline":
                pass  # Keep as is
            elif not positive_label.endswith(("-thinking", "-projection", "-attribution")):
                positive_label = positive_label + "-thinking"
        else:
            raise ValueError(f"Invalid target_emotional_direction format: {target_emotional_direction}")
    else:
        raise ValueError("target_emotional_direction must be in format 'negative-positive' (e.g., 'depressive-normal')")
    
    results = []
    
    print(f"🎭 Running emotional steering pipeline: {negative_label} ↔ {positive_label}")
    print(f"📝 Processing {len(messages)} messages with batch size {batch_size}")
    
    # Debug: Check available vectors
    print(f"🔍 Available feature vectors: {list(feature_vectors.keys())}")
    print(f"🔍 Target negative label: {negative_label} {'✅' if negative_label in feature_vectors else '❌'}")
    print(f"🔍 Target positive label: {positive_label} {'✅' if positive_label in feature_vectors else '❌'}")
    
    for i, message in enumerate(tqdm(messages, desc="Processing messages")):
        result = {
            "message": message,
            "baseline": None,
            "negative_steered": None,
            "positive_steered": None,
            "analysis": {}
        }
        
        try:
            # Generate baseline response (no steering)
            input_ids = tokenizer.encode(message["content"], return_tensors="pt")
            
            with model.generate(
                {"input_ids": input_ids, "attention_mask": (input_ids != tokenizer.pad_token_id).long()},
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            ) as tracer:
                baseline_output = model.generator.output.save()
            
            baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if baseline_text.startswith(input_text):
                baseline_text = baseline_text[len(input_text):].strip()
            
            result["baseline"] = {
                "response": baseline_text,
                "analysis": analyze_emotional_content(baseline_text)
            }
            
            # Generate negative steering (enhance negative emotional pattern)
            if negative_label in feature_vectors:
                print(f"   🔄 Generating negative steering with {negative_label}")
                result["negative_steered"] = generate_and_analyze_emotional(
                    model, tokenizer, message["content"], feature_vectors, 
                    steering_config, negative_label, "positive", max_new_tokens
                )
                if result["negative_steered"] is None:
                    print(f"   ❌ Negative steering failed for {negative_label}")
            else:
                print(f"   ⚠️ Skipping negative steering - {negative_label} not in feature_vectors")
            
            # Generate positive steering (enhance normal/healthy thinking)
            if positive_label in feature_vectors:
                print(f"   🔄 Generating positive steering with {positive_label}")
                result["positive_steered"] = generate_and_analyze_emotional(
                    model, tokenizer, message["content"], feature_vectors,
                    steering_config, positive_label, "positive", max_new_tokens
                )
                if result["positive_steered"] is None:
                    print(f"   ❌ Positive steering failed for {positive_label}")
            else:
                print(f"   ⚠️ Skipping positive steering - {positive_label} not in feature_vectors")
            
            # Compute analysis metrics
            baseline_score = result["baseline"]["analysis"]["total_emotional_score"]
            negative_score = result["negative_steered"]["emotional_analysis"]["total_emotional_score"] if result["negative_steered"] else baseline_score
            positive_score = result["positive_steered"]["emotional_analysis"]["total_emotional_score"] if result["positive_steered"] else baseline_score
            
            result["analysis"] = {
                "baseline_emotional_score": baseline_score,
                "negative_steered_score": negative_score,
                "positive_steered_score": positive_score,
                "negative_delta": negative_score - baseline_score,
                "positive_delta": positive_score - baseline_score,
                "steering_effectiveness": abs(negative_score - positive_score)
            }
            
        except Exception as e:
            print(f"❌ Error processing message {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        results.append(result)
    
    # Compute overall statistics
    valid_results = [r for r in results if r["analysis"]]
    if valid_results:
        avg_baseline = np.mean([r["analysis"]["baseline_emotional_score"] for r in valid_results])
        avg_negative_delta = np.mean([r["analysis"]["negative_delta"] for r in valid_results])
        avg_positive_delta = np.mean([r["analysis"]["positive_delta"] for r in valid_results])
        avg_effectiveness = np.mean([r["analysis"]["steering_effectiveness"] for r in valid_results])
        
        overall_stats = {
            "num_processed": len(valid_results),
            "avg_baseline_emotional_score": avg_baseline,
            "avg_negative_steering_delta": avg_negative_delta,
            "avg_positive_steering_delta": avg_positive_delta,
            "avg_steering_effectiveness": avg_effectiveness,
            "negative_steering_success": avg_negative_delta > 0,  # Should increase emotional score
            "positive_steering_success": avg_positive_delta < 0   # Should decrease emotional score
        }
        
        print(f"\n📊 Pipeline Results:")
        print(f"   Processed: {overall_stats['num_processed']} messages")
        print(f"   Baseline emotional score: {overall_stats['avg_baseline_emotional_score']:.2f}%")
        print(f"   Negative steering delta: {overall_stats['avg_negative_steering_delta']:.2f}%")
        print(f"   Positive steering delta: {overall_stats['avg_positive_steering_delta']:.2f}%")
        print(f"   Steering effectiveness: {overall_stats['avg_steering_effectiveness']:.2f}%")
        print(f"   Negative steering success: {overall_stats['negative_steering_success']}")
        print(f"   Positive steering success: {overall_stats['positive_steering_success']}")
    else:
        overall_stats = {}
    
    return {
        "target_direction": target_emotional_direction,
        "results": results,
        "overall_stats": overall_stats
    }
