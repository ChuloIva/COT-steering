# Instructions for Altering the COT-Steering System to Generate Different Steering Vectors

## Overview

This document provides step-by-step instructions for modifying the COT-steering system to generate different types of steering vectors beyond the current cognitive and emotional categories. The system is designed to be extensible, allowing researchers to train vectors for any reasoning patterns that can be identified and annotated.

## Code Review Summary

After reviewing the `emotional_reasoning_steering_complete (1).ipynb` notebook against the source code implementation in `utils/utils.py` and `messages/messages.py`, all functions work correctly with the existing codebase:

### ✅ Verified Functions:
- `load_model_and_vectors()` - Correctly loads models and existing vectors
- `process_batch_annotations()` - Properly annotates text with reasoning labels  
- `get_label_positions()` - Accurately maps annotations to token positions
- `process_saved_responses_batch()` - Correctly extracts neural activations
- `custom_generate_steering()` - Properly applies steering interventions during generation
- `analyze_emotional_content()` - Correctly analyzes emotional content using keyword matching
- `generate_and_analyze_emotional()` - Combines generation and analysis effectively

### ✅ Configuration Compatibility:
- `steering_config` dictionary includes proper configurations for multiple models
- All layer indices and coefficients are within valid ranges for each model architecture
- Device detection and tensor operations work across CUDA, MPS, and CPU

## Step-by-Step Instructions for Creating New Steering Vectors

### Step 1: Define Your New Reasoning Categories

1. **Identify the reasoning patterns** you want to steer toward/away from. Examples:
   - **Creative thinking**: divergent thinking, novel associations, imaginative solutions
   - **Analytical thinking**: systematic analysis, logical decomposition, methodical evaluation
   - **Social reasoning**: perspective-taking, empathy, social norm awareness
   - **Metacognitive patterns**: self-monitoring, strategy selection, reflection

2. **Create keyword sets** for each category to help with:
   - Message selection during training data generation
   - Content analysis for effectiveness evaluation

### Step 2: Modify the Annotation System

**File to modify:** `utils/utils.py`, function `process_batch_annotations()`

1. **Expand the annotation prompt** to include your new categories:

```python
# Add your new labels to the prompt in process_batch_annotations()
prompt = f"""
Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"].

Available labels:

Cognitive Labels:
0. initializing -> The model is rephrasing the given task and states initial thoughts.
1. deduction -> The model is performing a deduction step based on its current approach.
2. adding-knowledge -> The model is enriching the current approach with recalled facts.
3. example-testing -> The model generates examples to test its current approach.
4. uncertainty-estimation -> The model is stating its own uncertainty.
5. backtracking -> The model decides to change its approach.

Emotional Labels:  
6. depressive-thinking -> Self-critical thoughts, hopelessness, catastrophizing.
7. anxious-thinking -> Worry, rumination, worst-case scenarios.
8. negative-attribution -> Attributing failures to internal/permanent causes.
9. pessimistic-projection -> Predicting negative outcomes, focusing on failures.

Creative Labels: (NEW)
10. divergent-thinking -> Generating multiple creative solutions or ideas.
11. novel-associations -> Making unexpected connections between concepts.
12. imaginative-solutions -> Creating innovative approaches to problems.

# Add more categories as needed...
"""
```

2. **Update the label filtering logic** in your training pipeline to recognize new categories:

```python
# In train_emotional_vectors_final() or similar function
target_labels = [
    # Existing labels
    "depressive-thinking", "anxious-thinking", "negative-attribution", "pessimistic-projection",
    # Add your new labels
    "divergent-thinking", "novel-associations", "imaginative-solutions"
]

# Update the filtering keywords
relevant_keywords = [
    "thinking", "attribution", "projection", "anxious", "depressive", "negative", "pessimistic",
    # Add keywords for your new categories
    "divergent", "novel", "imaginative", "creative", "analytical", "social", "metacognitive"
]
```

### Step 3: Create Training Data

1. **Modify message selection criteria** to include prompts that elicit your target reasoning patterns:

```python
# In the notebook or a new training script
def filter_messages_for_reasoning_type(messages, reasoning_type):
    """Filter messages to find those that elicit specific reasoning patterns"""
    
    reasoning_indicators = {
        "creative": [
            "creative", "innovative", "brainstorm", "imagine", "what if", 
            "alternative ways", "think outside", "novel approach"
        ],
        "analytical": [
            "analyze", "break down", "systematic", "step by step",
            "examine", "evaluate", "compare", "assess"
        ],
        "social": [
            "perspective", "others feel", "social", "relationship",
            "empathy", "understand others", "social norms"
        ]
        # Add more categories as needed
    }
    
    filtered_messages = []
    indicators = reasoning_indicators.get(reasoning_type, [])
    
    for msg in messages:
        content = msg["content"].lower()
        if any(indicator in content for indicator in indicators):
            filtered_messages.append(msg)
    
    return filtered_messages
```

2. **Generate responses** using the new message selection:

```python
# Select messages that elicit your target reasoning
creative_messages = filter_messages_for_reasoning_type(messages, "creative")
creative_responses = generate_responses_batch(model, tokenizer, creative_messages[:20])
```

### Step 4: Update Steering Configuration

**File to modify:** `utils/utils.py`, `steering_config` dictionary

1. **Add configuration entries** for your new reasoning categories:

```python
steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        # Existing configurations...
        
        # Creative reasoning categories (NEW)
        "divergent-thinking": {
            "vector_layer": 20, 
            "pos_layers": [20], 
            "neg_layers": [20], 
            "pos_coefficient": 1.3, 
            "neg_coefficient": 1.0
        },
        "novel-associations": {
            "vector_layer": 19, 
            "pos_layers": [19], 
            "neg_layers": [19], 
            "pos_coefficient": 1.2, 
            "neg_coefficient": 1.0
        },
        "imaginative-solutions": {
            "vector_layer": 21, 
            "pos_layers": [21], 
            "neg_layers": [21], 
            "pos_coefficient": 1.4, 
            "neg_coefficient": 1.0
        },
        
        # Add more categories...
    }
}
```

**Layer Selection Guidelines:**
- Use layers in the latter half of the model (layers 15-27 for the 28-layer model)
- Different reasoning types may benefit from different layers
- Experiment with different layers to find optimal steering effectiveness

### Step 5: Create Analysis Functions

1. **Create keyword-based analysis** for your new reasoning categories:

```python
def analyze_creative_content(response_text):
    """Analyze response for creative thinking patterns"""
    
    divergent_keywords = [
        "multiple ways", "various approaches", "different methods", "alternatives",
        "brainstorm", "options", "possibilities", "many solutions"
    ]
    
    novel_keywords = [
        "unique", "original", "innovative", "creative", "novel", "unprecedented",
        "never seen", "new approach", "unexpected", "unconventional"
    ]
    
    imaginative_keywords = [
        "imagine", "envision", "picture", "dream up", "conceptualize",
        "fantastic", "visionary", "inventive", "original idea"
    ]
    
    response_lower = response_text.lower()
    total_words = len(response_text.split())
    
    # Count keyword matches
    divergent_count = sum(1 for keyword in divergent_keywords if keyword in response_lower)
    novel_count = sum(1 for keyword in novel_keywords if keyword in response_lower)
    imaginative_count = sum(1 for keyword in imaginative_keywords if keyword in response_lower)
    
    return {
        "divergent_score": divergent_count / max(total_words, 1) * 100,
        "novel_score": novel_count / max(total_words, 1) * 100,
        "imaginative_score": imaginative_count / max(total_words, 1) * 100,
        "total_creative_score": (divergent_count + novel_count + imaginative_count) / max(total_words, 1) * 100,
        # Include keyword counts for debugging
        "divergent_keywords_found": divergent_count,
        "novel_keywords_found": novel_count,
        "imaginative_keywords_found": imaginative_count
    }
```

2. **Update the main analysis function** to include your new categories:

```python
def analyze_reasoning_content(response_text, reasoning_types=None):
    """Comprehensive analysis of different reasoning patterns in text"""
    
    if reasoning_types is None:
        reasoning_types = ["emotional", "creative", "analytical"]  # Add your types
    
    results = {}
    
    if "emotional" in reasoning_types:
        results.update(analyze_emotional_content(response_text))
    
    if "creative" in reasoning_types:
        results.update(analyze_creative_content(response_text))
    
    # Add more analysis functions as needed
    
    return results
```

### Step 6: Update Training Pipeline

1. **Modify the vector training function** to handle new categories:

```python
def train_reasoning_vectors(responses, annotations, model, tokenizer, target_categories):
    """Train steering vectors for any reasoning categories"""
    
    # Extract neural activations
    batch_activations = process_saved_responses_batch(responses, tokenizer, model)
    
    # Discover all labels in annotations
    all_labels = set()
    for annotation in annotations:
        label_matches = re.findall(r'\["([^"]+)"\]', annotation)
        all_labels.update(label_matches)
    
    all_labels.discard('end-section')
    
    # Filter to target categories
    target_labels = []
    for label in all_labels:
        if any(category in label for category in target_categories):
            target_labels.append(label)
    
    # Continue with existing training logic...
    # (rest of the function remains the same)
```

### Step 7: Test and Validate

1. **Create test scenarios** for your new reasoning categories:

```python
test_messages = [
    "Come up with 5 completely different ways to solve this problem: How can we reduce plastic waste?",
    "Think of an innovative solution that combines technology with environmental conservation.",
    "Imagine a world where gravity works differently. How would architecture change?"
]
```

2. **Run steering experiments** to test effectiveness:

```python
# Test creative steering
for message in test_messages:
    baseline_result = generate_and_analyze_creative(
        model, tokenizer, message, feature_vectors, steering_config,
        "divergent-thinking", "baseline"
    )
    
    positive_result = generate_and_analyze_creative(
        model, tokenizer, message, feature_vectors, steering_config,
        "divergent-thinking", "positive"
    )
    
    # Compare creative scores
    print(f"Baseline creative score: {baseline_result['creative_analysis']['total_creative_score']:.2f}")
    print(f"Enhanced creative score: {positive_result['creative_analysis']['total_creative_score']:.2f}")
```

## Important Considerations

### Model Architecture Compatibility

- **Layer indices**: Ensure your steering configuration uses valid layer indices for each model
- **Hidden dimensions**: The system should automatically handle different hidden sizes
- **Device compatibility**: The code handles CUDA, MPS, and CPU automatically

### Data Quality

- **Annotation quality**: High-quality annotations are crucial for effective vectors
- **Training data diversity**: Use diverse prompts that consistently elicit target reasoning patterns
- **Balanced representation**: Ensure adequate examples for each reasoning category

### Evaluation Metrics

- **Keyword-based metrics**: Create comprehensive keyword sets for each reasoning type
- **Human evaluation**: Consider human assessment for complex reasoning patterns
- **Baseline comparisons**: Always compare steered vs. unsteered responses

### Ethical Considerations

- **Research purposes only**: New steering vectors should only be used for legitimate research
- **Safety evaluation**: Test for potential negative side effects or biased outputs
- **Transparency**: Document the intended use and limitations of new vectors

## File Structure Summary

When implementing new steering vectors, you'll primarily modify these files:

```
COT-steering/
├── utils/
│   └── utils.py                    # Add new annotation labels, steering config, analysis functions
├── messages/
│   └── messages.py                 # Optionally add new training messages
└── train-steering-vectors/
    └── results/vars/               # Generated vector files will be saved here
        └── mean_vectors_*.pt
```

## Example: Complete Implementation for Creative Reasoning

Here's a complete example showing how to add creative reasoning vectors:

1. **Update annotation prompt** in `utils.py`
2. **Add steering configuration** for creative categories
3. **Create creative content analysis** function
4. **Filter messages** for creative prompts
5. **Train vectors** using the modified pipeline
6. **Test effectiveness** with creative evaluation metrics

The system is designed to be modular and extensible, so following these patterns will allow you to create steering vectors for virtually any reasoning pattern that can be consistently identified and annotated.

## Troubleshooting

### Common Issues

1. **No vectors generated**: Check that your annotation labels match the filtering criteria
2. **Poor steering effectiveness**: Experiment with different layers and coefficients
3. **Inconsistent results**: Ensure training data consistently exhibits target reasoning patterns
4. **Memory issues**: Reduce batch size or use gradient checkpointing for large models

### Debugging Tips

1. **Check label discovery**: Print all discovered labels to verify annotation parsing
2. **Validate token positions**: Ensure text segments map correctly to token ranges
3. **Monitor activation shapes**: Verify tensor dimensions throughout the pipeline
4. **Test steering coefficients**: Start with small coefficients (0.5-1.0) and increase gradually

This framework provides a comprehensive foundation for extending the COT-steering system to any reasoning patterns you want to study or control.