import argparse
import dotenv
dotenv.load_dotenv("../.env")

import os
import json
import random
from tqdm import tqdm
import utils

# Parse arguments
parser = argparse.ArgumentParser(description="Generate emotional training examples using Anthropic API")
parser.add_argument("--emotion_category", type=str, default="depressive",
                    help="Emotion category to generate examples for")
parser.add_argument("--n_examples", type=int, default=100,
                    help="Number of examples to generate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=10,
                    help="Batch size for API calls")
args, _ = parser.parse_known_args()

def generate_emotional_prompts(emotion_category, n_examples):
    """Generate prompts that elicit specific emotional responses"""
    
    emotion_prompts = {
        "depressive": [
            "Describe a situation where you feel completely overwhelmed by daily tasks.",
            "Explain how it feels when your efforts seem meaningless.",
            "Describe a time when you felt disconnected from people around you.",
            "Talk about facing a problem that seems impossible to solve.",
            "Describe the feeling of disappointment after high expectations.",
            "Explain what it's like when motivation is completely absent.",
            "Describe feeling stuck in a situation with no way out.",
            "Talk about the weight of accumulated failures.",
            "Describe feeling misunderstood by those close to you.",
            "Explain the experience of persistent fatigue and low energy."
        ],
        "anxious": [
            "Describe worrying about an upcoming important event.",
            "Explain the feeling of uncertainty about the future.",
            "Talk about overthinking a conversation you had.",
            "Describe physical symptoms when feeling stressed.",
            "Explain catastrophic thinking patterns.",
            "Describe feeling like something bad will happen.",
            "Talk about perfectionism and fear of making mistakes.",
            "Describe social anxiety in group situations.",
            "Explain racing thoughts at night.",
            "Describe feeling overwhelmed by too many choices."
        ],
        "hopeful": [
            "Describe looking forward to a positive change.",
            "Explain finding strength during difficult times.",
            "Talk about learning from past mistakes.",
            "Describe seeing opportunities in challenges.",
            "Explain feeling supported by others.",
            "Describe small victories that matter.",
            "Talk about planning for a better future.",
            "Describe finding meaning in struggles.",
            "Explain resilience in facing setbacks.",
            "Describe gratitude for positive experiences."
        ]
    }
    
    base_prompts = emotion_prompts.get(emotion_category, emotion_prompts["depressive"])
    
    # Expand the prompt set by creating variations
    expanded_prompts = []
    for i in range(n_examples):
        base_prompt = random.choice(base_prompts)
        
        # Add contextual variations
        contexts = [
            f"From the perspective of a student: {base_prompt}",
            f"From the perspective of a working professional: {base_prompt}",
            f"In the context of relationships: {base_prompt}",
            f"Regarding personal growth: {base_prompt}",
            f"In academic or work settings: {base_prompt}",
            base_prompt  # Keep some original prompts
        ]
        
        expanded_prompts.append(random.choice(contexts))
    
    return expanded_prompts

def generate_emotional_responses(prompts, emotion_category):
    """Generate emotional responses using Anthropic API"""
    
    system_prompt = f"""You are an AI that generates realistic emotional responses. 
    For each prompt, provide a thoughtful response that authentically captures {emotion_category} emotional patterns.
    Include internal thoughts and reasoning that would be typical for someone experiencing these emotions.
    
    Format your response with thinking tags:
    <think>
    [Your internal reasoning and emotional processing here]
    </think>
    
    [Your final response here]
    """
    
    responses = []
    
    for prompt in tqdm(prompts, desc=f"Generating {emotion_category} responses"):
        full_prompt = f"{system_prompt}\n\nPrompt: {prompt}"
        
        try:
            response = utils.chat(full_prompt, model="claude-3-7-sonnet", max_tokens=10000)
            if response:
                responses.append({
                    "prompt": prompt,
                    "response": response,
                    "emotion_category": emotion_category
                })
        except Exception as e:
            print(f"Error generating response for prompt: {prompt[:50]}... Error: {e}")
            continue
    
    return responses

def main():
    # Create directories
    os.makedirs('results/vars', exist_ok=True)
    
    random.seed(args.seed)
    
    print(f"Generating {args.n_examples} examples for {args.emotion_category} category")
    
    # Generate prompts
    prompts = generate_emotional_prompts(args.emotion_category, args.n_examples)
    
    # Generate responses in batches
    all_responses = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_responses = generate_emotional_responses(batch_prompts, args.emotion_category)
        all_responses.extend(batch_responses)
        
        # Save intermediate results
        if i % (args.batch_size * 5) == 0:  # Save every 5 batches
            temp_path = f"results/vars/emotional_examples_{args.emotion_category}_temp.json"
            with open(temp_path, 'w') as f:
                json.dump(all_responses, f, indent=2)
    
    # Save final results
    output_path = f"results/vars/emotional_examples_{args.emotion_category}.json"
    with open(output_path, 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    print(f"Generated {len(all_responses)} emotional examples saved to {output_path}")

if __name__ == "__main__":
    main()