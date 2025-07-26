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
            
            for label in cognitive_labels + emotional_labels:
                if label != 'overall' and label in mean_vectors_dict:
                    feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

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
        feature_vectors: Dictionary of feature vectors containing steering_vector_set
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

        if feature_vectors is not None:       
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]
     

            # Get device and dtype from model
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            if steer_positive:
                feature_vector = feature_vectors[label][vector_layer].to(model_device).to(model_dtype)
                for layer_idx in pos_layers:         
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
            else:
                feature_vector = feature_vectors[label][vector_layer].to(model_device).to(model_dtype)
                for layer_idx in neg_layers:         
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
        
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
        "depressive-thinking": {"vector_layer": 13, "pos_layers": [13], "neg_layers": [13], "pos_coefficient": 1.5, "neg_coefficient": 1.0},
        "anxious-thinking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1.2, "neg_coefficient": 1.0},
        "negative-attribution": {"vector_layer": 11, "pos_layers": [11], "neg_layers": [11], "pos_coefficient": 1.3, "neg_coefficient": 1.0},
        "pessimistic-projection": {"vector_layer": 14, "pos_layers": [14], "neg_layers": [14], "pos_coefficient": 1.4, "neg_coefficient": 1.0},
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
        dict: Contains response text and emotional analysis
    """
    
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
