What the App Does

The steering-thinking-llms project is a research application that studies and controls reasoning behavior in "thinking" language models (like DeepSeek-R1 variants) using steering vectors. It's designed to understand and manipulate different types of reasoning patterns that occur during the model's internal "thinking" process.

Core Functionality

The app provides tools to:

1. Extract and analyze reasoning patterns from model thinking processes
2. Train steering vectors that can enhance or suppress specific reasoning behaviors  
3. Evaluate the effects of steering on model reasoning quality
4. Compare reasoning patterns across different models and conditions

How It Works - Application Flow

1. Response Generation Phase (train-steering-vectors/generate_responses.py)
•  Takes reasoning prompts (math problems, spatial puzzles, logic questions, pattern recognition)
•  Generates baseline responses from thinking models without any steering
•  Extracts the <think>...</think> sections containing the model's reasoning process
•  Saves responses in JSON format for further processing

2. Annotation & Vector Training Phase (train-steering-vectors/train_vectors.py)
•  Processes the thinking sections using GPT-4 to annotate them into 6 reasoning categories:
•  initializing: Rephrasing tasks and initial thoughts
•  deduction: Logical deduction steps
•  adding-knowledge: Incorporating recalled facts
•  example-testing: Testing approaches with examples
•  uncertainty-estimation: Expressing uncertainty
•  backtracking: Changing approach/correcting mistakes
•  Extracts neural activations from specific model layers during these reasoning types
•  Computes mean activation vectors for each reasoning category
•  Creates "steering vectors" by subtracting the overall mean from category-specific means

3. Steering Evaluation Phase (steering/evaluate_steering.py)
•  Uses the trained steering vectors to modify model behavior during inference
•  Can steer positively (enhance a reasoning type) or negatively (suppress it)
•  Applies steering by adding/subtracting the vectors to specific transformer layers during generation
•  Measures how steering affects the frequency of different reasoning patterns

4. Analysis & Visualization
•  Generates plots showing how steering affects reasoning behavior
•  Compares steered vs unsteered model performance
•  Analyzes layer-wise effects of steering across the model

Key Innovation

The core insight is that different types of reasoning create distinct activation patterns in transformer layers. By identifying these patterns and creating "steering vectors," the system can:

•  Enhance desired reasoning (e.g., make the model do more systematic deduction)
•  Suppress unwanted patterns (e.g., reduce uncertainty or backtracking)
•  Understand what makes thinking models reason the way they do

Technical Implementation

•  Uses nnsight library for model introspection and activation extraction
•  Applies steering by modifying intermediate layer outputs during generation
•  Different models use different optimal layers for steering (configured in steering_config)
•  Supports DeepSeek-R1 model variants (1.5B, 8B, 14B parameters)

## How Vectors Are Captured Based on Annotated Reasoning Steps

### 1. **Annotation Process**
The system takes raw thinking processes from the model and sends them to GPT-4 for annotation using a specific prompt:

```
Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. 

Available labels:
0. initializing -> The model is rephrasing the given task and states initial thoughts.
1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
2. adding-knowledge -> The model is enriching the current approach with recalled facts.
3. example-testing -> The model generates examples to test its current approach.
4. uncertainty-estimation -> The model is stating its own uncertainty.
5. backtracking -> The model decides to change its approach.
```

### 2. **Token Position Mapping**
Once annotated, the system maps the labeled text segments back to specific token positions in the original response:

- **`get_char_to_token_map()`**: Creates a mapping from character positions to token indices
- **`get_label_positions()`**: Uses regex to find annotated segments like `["deduction"]text["end-section"]` and maps them to token positions
- Each labeled segment gets converted to token ranges (start_token, end_token)

### 3. **Neural Activation Extraction**
The system then extracts neural activations for these specific token positions:

```python
# Extract layer outputs for the full response
with model.trace({"input_ids": tokenized_responses, "attention_mask": attention_mask}) as tracer:
    # Capture layer outputs from all transformer layers
    for layer_idx in range(model.config.num_hidden_layers):
        layer_outputs.append(model.model.layers[layer_idx].output[0].save())
```

### 4. **Vector Computation**
For each reasoning type, the system:

- **Extracts activations** from the token ranges corresponding to that reasoning type
- **Averages activations** across the token sequence for each layer
- **Updates running means** using incremental averaging:

```python
def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    # For each label (e.g., "deduction", "uncertainty-estimation")
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            # Extract activations for this token range
            vectors = layer_outputs[:, start-1:min(end-1, start+10)].mean(dim=1)
            
            # Update running mean
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1
```

### 5. **Steering Vector Creation**
Finally, steering vectors are created by:

- Computing an **overall mean** across all reasoning types
- Creating **differential vectors** by subtracting the overall mean from each specific reasoning type mean:

```python
feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']
```

### Example from the Data
Looking at a sample annotation:
```
["initializing"]Okay, so I have this analogy here: "Camera is to photo as pen is to _____." I need to figure out what word goes in the blank.["end-section"]

["deduction"]First, I know that a camera is used to take photos. So, the first part is clear.["end-section"]

["adding-knowledge"]I remember that a pen is used to write, so maybe the blank is about writing.["end-section"]
```

The system:
1. Maps `"initializing"` to specific token positions in the original response
2. Extracts neural activations from those exact tokens across all model layers
3. Accumulates these activations to build a mean "initializing" vector
4. Repeats for all other reasoning types
5. Creates steering vectors that can enhance/suppress each type of reasoning

This process creates vectors that capture the neural representations of different reasoning patterns, allowing the system to steer the model toward or away from specific types of thinking during generation.

This research helps understand how thinking language models work internally and provides a method to control their reasoning behavior, which could be valuable for improving AI reasoning capabilities or ensuring more reliable problem-solving approaches.
