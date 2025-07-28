# Emotional Reasoning Steering Complete - Technical Analysis

## Overview

The `emotional_reasoning_steering_complete (1).ipynb` notebook implements a comprehensive system for training and applying **emotional reasoning steering vectors** to language models. This extends the existing COT-steering framework to include emotional thinking patterns like depression, anxiety, negative attribution, and pessimistic projection. The notebook demonstrates how to systematically influence a model's emotional reasoning patterns during text generation.

## Workflow Summary

The notebook follows an 11-step process to implement emotional reasoning steering:

### 1. Setup and Model Loading
- **Purpose**: Load the target language model and any existing cognitive steering vectors
- **Key Function**: `load_model_and_vectors()` from `utils/utils.py`
- **Process**: 
  - Loads the DeepSeek-R1-Distill-Qwen-1.5B model using nnsight for neural intervention
  - Auto-detects optimal device (CUDA/MPS/CPU) and sets appropriate data types
  - Loads existing cognitive reasoning vectors (backtracking, uncertainty-estimation, etc.)
  - Returns model, tokenizer, and feature vectors for downstream use

### 2. Message Preparation
- **Purpose**: Curate training messages that elicit emotional reasoning patterns
- **Process**: Filters the message dataset to identify emotionally-charged prompts using heuristics
- **Key Indicators**: Messages containing phrases like "you've been", "everyone around you", "after receiving criticism"
- **Result**: 26 emotional messages identified from 540 total messages

### 3. Response Generation
- **Purpose**: Generate baseline responses to emotional prompts for training data
- **Process**: 
  - Uses the model's standard generation without any steering
  - Processes 20 selected emotional messages
  - Captures raw model responses that will be used for training steering vectors

### 4. Emotional Annotation
- **Purpose**: Label the generated responses with emotional reasoning categories
- **Key Function**: `process_batch_annotations()` from `utils/utils.py`
- **Process**:
  - Uses GPT-4 to annotate responses with emotional labels
  - Labels include: `depressive-thinking`, `anxious-thinking`, `negative-attribution`, `pessimistic-projection`
  - Creates structured annotations in format: `["label"] text segment ["end-section"]`
  - Enables precise mapping of emotional patterns to specific text segments

### 5. Neural Activation Extraction

This is the most technically complex step, involving precise extraction of neural activations from specific text segments that have been labeled with emotional reasoning patterns.

#### 5.1 Overview of the Process
- **Purpose**: Extract neural activations corresponding to labeled emotional segments
- **Goal**: Create training data that links specific emotional reasoning patterns to their neural representations in the model's hidden states
- **Challenge**: Map text-level annotations to precise token-level neural activations across all model layers

#### 5.2 Key Functions and Their Roles

##### `process_saved_responses_batch()` - Batch Neural Processing
```python
def process_saved_responses_batch(responses_list, tokenizer, model):
    """Get layer activations for a batch of saved responses without generation"""
```

**Detailed Process:**
1. **Tokenization with Padding**: 
   - Uses `get_batched_message_ids()` to tokenize all responses with consistent padding
   - Determines maximum token length across all responses
   - Pads shorter sequences to match the longest sequence
   - Creates attention masks to track real vs. padded tokens

2. **Neural Activation Capture**:
   - Uses `model.trace()` context manager from nnsight library
   - Processes all responses simultaneously through the model
   - Captures hidden states from every transformer layer (28 layers for DeepSeek-R1-Distill-Qwen-1.5B)
   - Each layer output has shape: `(batch_size, sequence_length, hidden_size)`

3. **Activation Post-processing**:
   - Converts tensors to CPU and float32 for memory efficiency
   - Removes padding tokens from each example using attention masks
   - Returns list of activations per response: `(num_layers, actual_seq_len, hidden_size)`

##### `get_label_positions()` - Annotation-to-Token Mapping
```python
def get_label_positions(annotated_thinking, response_text, tokenizer):
    """Parse SAE annotations and find token positions for each label"""
```

**Detailed Process:**
1. **Annotation Parsing**:
   - Uses regex pattern: `r'\["(\S+?)"\](.*?)\["end-section"\]'`
   - Extracts label-text pairs from structured annotations
   - Example: `["anxious-thinking"] I'm worried about... ["end-section"]`

2. **Character-to-Token Mapping**:
   - Calls `get_char_to_token_map()` to create precise mapping
   - Uses tokenizer's `encode_plus()` with `return_offsets_mapping=True`
   - Creates dictionary mapping each character position to its token index
   - Handles subword tokenization correctly (e.g., "worried" → ["wor", "ried"])

3. **Text Segment Localization**:
   - Finds each labeled text segment within the original response
   - Maps character positions to token positions
   - Handles edge cases: empty segments, missing text, token boundaries
   - Returns dictionary: `{label: [(start_token, end_token), ...]}`

##### `get_char_to_token_map()` - Precise Position Mapping
```python
def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
```

**Technical Details:**
- Uses tokenizer's offset mapping to track character ranges for each token
- Creates comprehensive mapping: `{char_position: token_index}`
- Essential for handling subword tokenization where single words split across multiple tokens
- Enables precise extraction of neural activations for specific text spans

#### 5.3 The Complete Extraction Pipeline

The notebook implements this through `train_emotional_vectors_final()`:

##### Phase 1: Batch Activation Extraction
```python
print("🧠 Extracting neural activations...")
batch_activations = process_saved_responses_batch(responses, tokenizer, model)
```

**What happens:**
- All 20 emotional responses are processed simultaneously
- Model forward pass captures activations from all 28 layers
- Result: List of 20 activation tensors, each with shape `(28, seq_len, 1536)`
- Memory efficient: processes entire batch without individual forward passes

##### Phase 2: Label Discovery and Filtering
```python
# First pass: discover all labels in the annotations
all_labels = set()
for annotation in annotations:
    label_matches = re.findall(r'\["([^"]+)"\]', annotation)
    all_labels.update(label_matches)

# Filter to emotional labels
emotional_labels = []
for label in all_labels:
    if label in target_emotional_labels or any(keyword in label for keyword in emotional_keywords):
        emotional_labels.append(label)
```

**Label Discovery Process:**
- Scans all annotations to discover every label used
- Filters for emotional reasoning labels vs. structural labels ("end-section")
- Target emotional labels: `depressive-thinking`, `anxious-thinking`, `negative-attribution`, `pessimistic-projection`
- Also captures any labels containing emotional keywords

##### Phase 3: Segment-Level Activation Extraction
```python
for i, (response, annotation) in enumerate(zip(responses, annotations)):
    # Get label positions in the response
    label_positions = get_label_positions(annotation, response, tokenizer)
    
    # Get activations for this response
    activations = batch_activations[i]  # Shape: (layers, seq_len, hidden_size)
    
    # Extract activations for each emotional label
    for label, positions in label_positions.items():
        if label in label_activations:  # Only collect emotional labels
            for start_pos, end_pos in positions:
                if start_pos < activations.shape[1] and end_pos <= activations.shape[1]:
                    # Extract segment activations across all layers
                    segment_activation = activations[:, start_pos:end_pos, :].mean(dim=1)
                    label_activations[label].append(segment_activation)
```

**Segment Extraction Details:**
- For each response, maps annotations to precise token positions
- Extracts activations for each labeled segment across all 28 layers
- Averages activations across tokens within each segment (mean pooling)
- Stores segment activations: shape `(28, 1536)` per segment
- Accumulates multiple segments per emotional category for robust training data

##### Phase 4: Mean Vector Computation
```python
# Compute mean vectors for each label
for label, activations_list in label_activations.items():
    if activations_list:
        activations_tensor = torch.stack(activations_list)  # (num_segments, layers, hidden_size)
        mean_vector = activations_tensor.mean(dim=0)  # (layers, hidden_size)
        
        mean_vectors[label] = {
            'mean': mean_vector,
            'count': len(activations_list)
        }
```

**Statistical Aggregation:**
- Stacks all segment activations for each emotional category
- Computes mean across all segments to get representative vector
- Results in mean activation pattern for each emotional reasoning type
- Tracks count of segments contributing to each mean (for quality assessment)

#### 5.4 Technical Challenges and Solutions

##### Challenge 1: Annotation-Text Alignment
**Problem**: GPT-4 annotations may not perfectly match original text due to minor formatting differences.

**Solution**: 
- Robust text matching using `response_text.find(text)`
- Character-level mapping to token positions
- Graceful handling of missing or misaligned segments

##### Challenge 2: Variable Sequence Lengths
**Problem**: Different responses have different lengths, complicating batch processing.

**Solution**:
- Dynamic padding to maximum sequence length
- Attention mask tracking for real vs. padded tokens
- Per-example padding removal after activation extraction

##### Challenge 3: Subword Tokenization
**Problem**: Emotional reasoning spans may cross token boundaries due to subword tokenization.

**Solution**:
- Character-to-token mapping using tokenizer offset information
- Inclusive token range selection for partial overlaps
- Mean pooling across tokens within segments

##### Challenge 4: Memory Efficiency
**Problem**: Storing activations for all layers and all tokens requires significant memory.

**Solution**:
- CPU offloading after extraction: `.cpu().detach().to(torch.float32)`
- Batch processing to amortize model forward pass costs
- Segment-level mean pooling to reduce storage requirements

#### 5.5 Data Structures and Shapes

**Input Data:**
- `responses`: List of 20 text strings (model outputs)
- `annotations`: List of 20 annotated strings with emotional labels

**Intermediate Representations:**
- `batch_activations`: List of 20 tensors, each `(28, seq_len, 1536)`
- `label_positions`: Dict mapping labels to `[(start_token, end_token), ...]`
- `label_activations`: Dict accumulating segment activations per label

**Output:**
- `mean_vectors`: Dict with structure:
  ```python
  {
    'depressive-thinking': {
      'mean': torch.Tensor(28, 1536),  # Mean activation across all segments
      'count': 15  # Number of segments contributing to this mean
    },
    'anxious-thinking': {...},
    # ... other emotional categories
  }
  ```

#### 5.6 Quality Assurance and Validation

The notebook includes extensive debugging and validation:

1. **Text Matching Verification**: Confirms that annotation segments appear in original responses
2. **Token Position Validation**: Ensures token ranges are within sequence bounds
3. **Activation Shape Checking**: Verifies tensor dimensions at each processing step
4. **Segment Count Reporting**: Tracks how many segments contribute to each emotional category
5. **Statistical Summaries**: Reports mean vector shapes and counts for quality assessment

This comprehensive neural activation extraction process creates the foundation for training effective emotional steering vectors by precisely linking emotional reasoning patterns to their neural representations in the language model.

### 6. Steering Vector Training
- **Purpose**: Compute differential feature vectors for emotional steering
- **Process**:
  - Calculates mean activations for each emotional category
  - Computes differential vectors by subtracting overall mean from category means
  - Creates steering vectors that represent the neural signature of each emotional pattern
  - Combines with existing cognitive vectors for comprehensive steering capabilities

### 7. Steering Configuration
- **Purpose**: Define layer-specific steering parameters for each emotional category
- **Key Data Structure**: `steering_config` dictionary in `utils/utils.py`
- **Configuration Elements**:
  - `vector_layer`: Which layer's activations to use for the steering vector
  - `pos_layers`/`neg_layers`: Which layers to apply positive/negative steering
  - `pos_coefficient`/`neg_coefficient`: Strength multipliers for steering interventions
- **Emotional Categories**:
  - `depressive-thinking`: Layer 18, coefficient 1.5
  - `anxious-thinking`: Layer 17, coefficient 1.2  
  - `negative-attribution`: Layer 16, coefficient 1.3
  - `pessimistic-projection`: Layer 19, coefficient 1.4

### 8. Steering Implementation and Testing
- **Purpose**: Apply emotional steering during text generation and evaluate effectiveness
- **Key Function**: `custom_generate_steering()` from `utils/utils.py`
- **Process**:
  - Modifies model activations during generation using nnsight interventions
  - Supports both positive steering (enhance pattern) and negative steering (suppress pattern)
  - Tests on evaluation messages with different steering configurations
  - Compares baseline vs. steered responses

### 9. Emotional Content Analysis
- **Purpose**: Quantify emotional content in generated responses
- **Key Function**: `analyze_emotional_content()` from `utils/utils.py`
- **Analysis Method**:
  - Keyword-based scoring for each emotional category
  - Calculates percentage scores based on emotional keyword density
  - Tracks specific emotional indicators found in responses
  - Provides quantitative metrics for steering effectiveness

### 10. Results Visualization
- **Purpose**: Create visual analysis of steering effectiveness
- **Visualizations**:
  - Bar charts comparing baseline vs. steered emotional scores
  - Heatmaps showing steering effectiveness across categories and directions
  - Effectiveness metrics for positive vs. negative steering

### 11. Safety and Ethics Documentation
- **Purpose**: Document responsible use guidelines and safety considerations
- **Key Elements**:
  - Research-only usage warnings
  - Ethical guidelines and institutional oversight requirements
  - Technical safety measures and access controls
  - Potential risks and mitigation strategies

## Key Functions from Main Codebase

### Core Utility Functions (`utils/utils.py`)

#### `load_model_and_vectors()`
- **Purpose**: Loads language model and existing steering vectors
- **Parameters**: device, load_in_8bit, compute_features, model_name
- **Returns**: model, tokenizer, feature_vectors
- **Key Features**:
  - Auto-detects optimal device and data types
  - Loads pre-trained cognitive steering vectors
  - Computes differential feature vectors by subtracting overall mean
  - Supports both cognitive and emotional vector categories

#### `process_batch_annotations()`
- **Purpose**: Annotates reasoning chains with cognitive/emotional labels
- **Parameters**: thinking_processes, include_emotional, annotation_model
- **Process**:
  - Uses GPT-4 to label text segments with reasoning categories
  - Supports both cognitive labels (deduction, backtracking) and emotional labels
  - Returns structured annotations in `["label"]...["end-section"]` format
  - Enables precise mapping of reasoning patterns to text segments

#### `custom_generate_steering()`
- **Purpose**: Generates text with neural steering interventions
- **Parameters**: model, tokenizer, input_ids, label, feature_vectors, steer_positive
- **Process**:
  - Uses nnsight to modify model activations during generation
  - Applies steering vectors to specified layers with configurable coefficients
  - Supports both positive steering (enhance pattern) and negative steering (suppress pattern)
  - Maintains generation quality while systematically biasing reasoning patterns

#### `get_label_positions()`
- **Purpose**: Maps annotation labels to token positions in text
- **Parameters**: annotated_thinking, response_text, tokenizer
- **Process**:
  - Parses structured annotations using regex patterns
  - Creates character-to-token mapping for precise positioning
  - Returns dictionary mapping labels to token ranges
  - Enables extraction of neural activations for specific labeled segments

#### `process_saved_responses_batch()`
- **Purpose**: Extracts neural activations from saved responses
- **Parameters**: responses_list, tokenizer, model
- **Process**:
  - Processes responses through model to capture hidden states
  - Handles padding and attention masks correctly
  - Returns activations for all layers across all responses
  - Provides training data for steering vector computation

#### `analyze_emotional_content()`
- **Purpose**: Quantifies emotional content in text using keyword analysis
- **Parameters**: response_text
- **Analysis Categories**:
  - **Depressive**: hopeless, worthless, failure, inadequate, useless
  - **Anxious**: worry, scared, what if, worst case, overwhelming
  - **Negative Attribution**: my fault, I'm bad at, due to my, I lack
  - **Pessimistic**: will fail, won't work, doomed to, no point, futile
- **Returns**: Percentage scores, word counts, and keyword matches for each category

#### `generate_and_analyze_emotional()`
- **Purpose**: Combines steering generation with emotional analysis
- **Parameters**: model, tokenizer, message, feature_vectors, steering_config, label, steer_mode
- **Process**:
  - Generates response with specified emotional steering
  - Analyzes emotional content of the generated response
  - Returns both the response text and emotional analysis metrics
  - Provides end-to-end emotional steering evaluation

### Message Data (`messages/messages.py`)

#### `messages`
- **Purpose**: Training dataset of 540 reasoning prompts
- **Categories**: Mathematical logic, spatial reasoning, verbal logic, pattern recognition, lateral thinking, causal reasoning, probabilistic thinking, systems thinking, creative problem solving
- **Usage**: Source material for generating training responses and testing steering effectiveness

#### `eval_messages`
- **Purpose**: Evaluation dataset for testing steering capabilities
- **Structure**: Similar categories to training messages but distinct prompts
- **Usage**: Independent test set for measuring steering effectiveness without training data contamination

## Technical Architecture

### Neural Intervention Mechanism
The system uses **nnsight** to perform surgical interventions on language model activations:

1. **Activation Capture**: During generation, hidden states are captured at each layer
2. **Vector Application**: Steering vectors are added/subtracted from activations at specified layers
3. **Coefficient Scaling**: Intervention strength is controlled via layer-specific coefficients
4. **Selective Targeting**: Different emotional patterns target different layers for optimal effectiveness

### Training Data Pipeline
The emotional steering training follows this pipeline:

1. **Message Selection**: Identify emotionally-charged prompts from the dataset
2. **Response Generation**: Generate baseline responses without steering
3. **Annotation**: Use GPT-4 to label emotional reasoning patterns in responses
4. **Activation Extraction**: Process annotated responses to capture neural activations
5. **Vector Computation**: Calculate differential vectors representing emotional patterns
6. **Configuration**: Define layer-specific steering parameters for each emotional category

### Evaluation Framework
Steering effectiveness is measured through:

1. **Quantitative Analysis**: Keyword-based emotional content scoring
2. **Comparative Analysis**: Baseline vs. steered response comparison
3. **Directional Validation**: Positive steering increases scores, negative steering decreases scores
4. **Statistical Reporting**: Average score changes and effectiveness metrics
5. **Visual Analysis**: Charts and heatmaps showing steering patterns

## Safety and Ethical Considerations

### Research-Only Usage
- This implementation is designed exclusively for research purposes
- Requires institutional review board (IRB) approval for human subjects research
- Should not be deployed in production systems without extensive safety testing

### Potential Risks
- **Psychological Harm**: Steering toward negative emotional states could be harmful
- **Misuse**: Could be used to manipulate users or create harmful content  
- **Bias Amplification**: May amplify existing biases in training data
- **Unintended Effects**: Steering may have unpredictable side effects

### Required Safeguards
- **Informed Consent**: Users must know when emotional steering is active
- **Monitoring**: Continuous monitoring for harmful outputs
- **Reversibility**: Always provide countermeasures (negative steering)
- **Access Controls**: Restrict access to authorized researchers only
- **Documentation**: Maintain detailed logs of all experiments

## Applications and Future Work

### Research Applications
- **Mental Health Research**: Understanding AI representation of emotional states
- **Bias Detection**: Identifying problematic thinking patterns in AI outputs
- **Therapeutic AI**: Training models to recognize and counter negative thought patterns
- **Content Moderation**: Detecting and filtering emotionally harmful content
- **AI Safety**: Understanding and controlling emotional biases in language models

### Technical Extensions
- **Expanded Training Data**: More diverse emotional prompts and responses
- **Fine-tuned Coefficients**: Optimization of steering parameters for each model
- **Real-time Monitoring**: Automated detection of harmful emotional content
- **Multi-model Support**: Extension to other language model architectures
- **Clinical Applications**: Integration with therapeutic frameworks under proper oversight

## Conclusion

The emotional reasoning steering notebook demonstrates a sophisticated approach to understanding and controlling emotional biases in language models. By extending the COT-steering framework with emotional reasoning categories, it provides researchers with powerful tools for studying how AI systems represent and generate emotional content. The implementation emphasizes safety, ethical considerations, and responsible research practices while providing comprehensive technical capabilities for emotional AI research. 