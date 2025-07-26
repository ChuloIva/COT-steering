# Instructions for Implementing Depressive/Anxious Thinking Steering

## Overview

This document provides a comprehensive step-by-step guide on how to implement emotional reasoning steering in language models using the COT-steering framework. This involves steering towards or away from depressive or anxious thinking patterns by extending the existing cognitive steering system.

## 1. Expand the Reasoning Categories

First, extend the current 6-category framework to include emotional/psychological reasoning patterns:

### Current Categories:
- initializing, deduction, adding-knowledge, example-testing, uncertainty-estimation, backtracking

### New Categories to Add:
- **depressive-thinking**: Self-critical thoughts, hopelessness, catastrophizing, negative self-assessment
- **anxious-thinking**: Worry, rumination, worst-case scenarios, hypervigilance about problems
- **negative-attribution**: Attributing failures to internal/permanent causes, minimizing successes
- **pessimistic-projection**: Predicting negative outcomes, focusing on potential failures

## 2. Data Collection & Annotation

### Modify the Prompt Generation:
- Create prompts that naturally elicit emotional reasoning patterns
- Examples:
  - "Reflect on a time when things didn't go as planned..."
  - "What could go wrong with this approach?"
  - "How might this decision backfire?"
  - "What are your concerns about this solution?"

### Update the Annotation System:
Modify the GPT-4 annotation prompt to recognize emotional patterns:

```
Please split the following reasoning chain into annotated parts using these labels:

Cognitive Labels:
0. initializing -> Rephrasing tasks and initial thoughts
1. deduction -> Logical reasoning steps
2. adding-knowledge -> Incorporating recalled facts
3. example-testing -> Testing approaches with examples
4. uncertainty-estimation -> Expressing uncertainty
5. backtracking -> Changing approach

Emotional Labels:
6. depressive-thinking -> Self-critical thoughts, hopelessness, focusing on negatives
7. anxious-thinking -> Worry, rumination, catastrophizing about outcomes  
8. negative-attribution -> Blaming self, minimizing positives, pessimistic explanations
9. pessimistic-projection -> Predicting failures, focusing on what could go wrong
```

## 3. Training Data Generation

### Create Emotionally-Charged Prompts:
- Personal reflection questions
- Ethical dilemmas with emotional weight
- Scenarios involving failure, disappointment, or uncertainty
- Questions about self-worth, capability, or future outcomes

### Example Prompts:
- "You've been working on this project for months and it's still not working. What does this say about your abilities?"
- "Everyone else seems to understand this concept easily. Why are you struggling?"
- "What if your solution makes things worse instead of better?"

## 4. Vector Training Process

Following the existing system architecture:

1. **Generate Emotional Responses**: Use the new prompts to collect model responses containing emotional reasoning patterns

2. **Annotate with Emotional Labels**: Process through GPT-4 with the expanded annotation scheme

3. **Extract Neural Activations**: Use the same `process_saved_responses_batch()` function to capture activations at emotional reasoning segments

4. **Compute Emotional Vectors**: Build mean vectors for each emotional reasoning type using the existing `update_mean_vectors()` function

5. **Create Steering Vectors**: Generate differential vectors by subtracting overall mean from emotional category means

## 5. Steering Configuration

**Extend the `steering_config`:**
```python
steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        # Existing cognitive categories...
        "depressive-thinking": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1.5, "neg_coefficient": 1.0},
        "anxious-thinking": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1.2, "neg_coefficient": 1.0},
        "negative-attribution": {"vector_layer": 16, "pos_layers": [16], "neg_layers": [16], "pos_coefficient": 1.3, "neg_coefficient": 1.0},
        "pessimistic-projection": {"vector_layer": 19, "pos_layers": [19], "neg_layers": [19], "pos_coefficient": 1.4, "neg_coefficient": 1.0}
    }
}
```

## 6. Implementation Steps

1. **Modify `messages/messages.py`**: Add emotionally-charged prompts
2. **Update `utils/utils.py`**: Extend annotation categories and steering config
3. **Run `generate_responses.py`**: Collect responses to emotional prompts
4. **Run `train_vectors.py`**: Train emotional steering vectors
5. **Test with `evaluate_steering.py`**: Evaluate emotional steering effects

## 7. Evaluation Metrics

Create new evaluation criteria:
- **Emotional Tone Analysis**: Measure sentiment polarity of responses
- **Catastrophizing Frequency**: Count worst-case scenario mentions
- **Self-Critical Language**: Track negative self-references
- **Optimism vs. Pessimism Ratio**: Compare positive vs. negative outcome predictions

## 8. Ethical Considerations & Safeguards

**Important Safety Measures:**
- **Research-Only Use**: Clearly label as research tool, not for production
- **Consent & Disclosure**: Users must know when emotional steering is active
- **Reversibility**: Always provide "positive steering" counterparts
- **Monitoring**: Track for harmful outputs or excessive negativity
- **Access Controls**: Restrict to authorized researchers only

## 9. Example Usage

```python
# Steer toward depressive thinking
response = generate_and_analyze(
    model, tokenizer, message, 
    feature_vectors, 
    steering_config, 
    label="depressive-thinking", 
    steer_mode="positive"  # Enhance depressive patterns
)

# Steer away from anxious thinking  
response = generate_and_analyze(
    model, tokenizer, message,
    feature_vectors,
    steering_config, 
    label="anxious-thinking",
    steer_mode="negative"  # Suppress anxious patterns
)
```

## 10. Research Applications

This system could be valuable for:
- **Mental Health Research**: Understanding how AI models represent emotional states
- **Bias Detection**: Identifying when models exhibit problematic thinking patterns
- **Therapeutic AI**: Training models to recognize and counter negative thought patterns
- **Content Moderation**: Detecting and filtering emotionally harmful content

**Warning**: This should only be implemented for legitimate research purposes with proper ethical oversight, as steering models toward negative emotional states could be harmful if misused.

