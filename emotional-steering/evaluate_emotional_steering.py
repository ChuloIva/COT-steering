import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
import json
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from emotional_steering import EmotionalSteeringManager  
from emotional_annotation import annotate_emotional_patterns, extract_emotional_segments, validate_emotional_annotations
import utils

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate emotional steering effectiveness")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to evaluate")
parser.add_argument("--emotions", type=str, nargs='+', default=["depressive", "hopeful", "anxious"],
                    help="Emotions to evaluate")
parser.add_argument("--n_examples", type=int, default=20,
                    help="Number of examples to test per emotion")
parser.add_argument("--max_tokens", type=int, default=300,
                    help="Maximum tokens to generate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args, _ = parser.parse_known_args()

def get_evaluation_prompts() -> List[str]:
    """Get diverse prompts for evaluating emotional steering"""
    
    return [
        "How do you feel about facing challenges in life?",
        "What are your thoughts on your future prospects?", 
        "Describe how you handle setbacks and failures.",
        "What's your perspective on personal relationships?",
        "How do you view your abilities and potential?",
        "What do you think about when you're alone?",
        "How do you feel about taking on new responsibilities?",
        "Describe your approach to solving difficult problems.",
        "What are your thoughts on change and uncertainty?",
        "How do you feel about your past decisions?",
        "What motivates you to keep going when things are hard?",
        "How do you see yourself compared to others?",
        "What are your thoughts on seeking help from others?",
        "How do you handle criticism and feedback?",
        "What do you think about your daily routine and habits?",
        "How do you feel about making important decisions?",
        "What's your perspective on learning from mistakes?",
        "How do you approach goal setting and achievement?",
        "What are your thoughts on self-improvement?",
        "How do you handle stress and pressure situations?"
    ]

def calculate_emotional_metrics(response_text: str, target_emotion: str) -> Dict:
    """Calculate metrics for emotional content in response"""
    
    # Annotate the response for emotional content
    annotated = annotate_emotional_patterns(response_text, target_emotion)
    segments = extract_emotional_segments(annotated)
    validation = validate_emotional_annotations(annotated)
    
    # Calculate basic metrics
    total_segments = sum(len(segs) for segs in segments.values())
    target_segments = len(segments.get(target_emotion, []))
    
    metrics = {
        "total_segments": total_segments,
        "target_emotion_segments": target_segments,
        "target_emotion_ratio": target_segments / max(total_segments, 1),
        "coverage_percentage": validation["coverage_percentage"],
        "emotion_distribution": {emotion: len(segs) for emotion, segs in segments.items()},
        "annotation_valid": validation["valid"]
    }
    
    # Calculate emotional intensity (rough estimate based on segment length)
    if target_emotion in segments:
        avg_segment_length = np.mean([len(seg) for seg in segments[target_emotion]])
        metrics["target_intensity"] = avg_segment_length / 100  # Normalize
    else:
        metrics["target_intensity"] = 0.0
    
    return metrics

def evaluate_steering_direction(
    manager: EmotionalSteeringManager,
    prompts: List[str],
    emotion: str,
    n_examples: int,
    max_tokens: int
) -> Dict:
    """Evaluate both positive and negative steering for an emotion"""
    
    results = {
        "emotion": emotion,
        "positive_steering": [],
        "negative_steering": [], 
        "baseline": []
    }
    
    test_prompts = random.sample(prompts, min(n_examples, len(prompts)))
    
    print(f"Evaluating {emotion} steering with {len(test_prompts)} prompts...")
    
    for prompt in tqdm(test_prompts, desc=f"Testing {emotion}"):
        try:
            # Baseline (no steering)
            input_ids = manager.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(next(manager.model.parameters()).device)
            
            with manager.model.generate(
                {"input_ids": input_ids},
                max_new_tokens=max_tokens,
                pad_token_id=manager.tokenizer.pad_token_id
            ) as tracer:
                baseline_output = manager.model.generator.output.save()
            
            baseline_response = manager.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
            baseline_metrics = calculate_emotional_metrics(baseline_response, emotion)
            
            results["baseline"].append({
                "prompt": prompt,
                "response": baseline_response,
                "metrics": baseline_metrics
            })
            
            # Positive steering (towards emotion)
            try:
                positive_response = manager.generate_with_emotional_steering(
                    prompt, emotion, steer_positive=True, max_new_tokens=max_tokens
                )
                positive_metrics = calculate_emotional_metrics(positive_response, emotion)
                
                results["positive_steering"].append({
                    "prompt": prompt,
                    "response": positive_response,
                    "metrics": positive_metrics
                })
            except Exception as e:
                print(f"Error in positive steering for {emotion}: {e}")
                results["positive_steering"].append(None)
            
            # Negative steering (away from emotion)  
            try:
                negative_response = manager.generate_with_emotional_steering(
                    prompt, emotion, steer_positive=False, max_new_tokens=max_tokens
                )
                negative_metrics = calculate_emotional_metrics(negative_response, emotion)
                
                results["negative_steering"].append({
                    "prompt": prompt,
                    "response": negative_response,
                    "metrics": negative_metrics
                })
            except Exception as e:
                print(f"Error in negative steering for {emotion}: {e}")
                results["negative_steering"].append(None)
                
        except Exception as e:
            print(f"Error processing prompt '{prompt[:50]}...': {e}")
            continue
    
    return results

def analyze_steering_effectiveness(results: Dict) -> Dict:
    """Analyze the effectiveness of steering"""
    
    emotion = results["emotion"]
    analysis = {"emotion": emotion}
    
    # Extract metrics for each condition
    conditions = ["baseline", "positive_steering", "negative_steering"]
    
    for condition in conditions:
        condition_data = [r for r in results[condition] if r is not None]
        if not condition_data:
            analysis[condition] = {"count": 0}
            continue
            
        metrics_list = [r["metrics"] for r in condition_data]
        
        # Calculate averages
        avg_target_ratio = np.mean([m["target_emotion_ratio"] for m in metrics_list])
        avg_intensity = np.mean([m["target_intensity"] for m in metrics_list])
        avg_coverage = np.mean([m["coverage_percentage"] for m in metrics_list])
        
        # Calculate emotion distribution
        all_distributions = [m["emotion_distribution"] for m in metrics_list]
        combined_distribution = {}
        for dist in all_distributions:
            for emotion_key, count in dist.items():
                combined_distribution[emotion_key] = combined_distribution.get(emotion_key, 0) + count
        
        analysis[condition] = {
            "count": len(condition_data),
            "avg_target_ratio": avg_target_ratio,
            "avg_intensity": avg_intensity,
            "avg_coverage": avg_coverage,
            "emotion_distribution": combined_distribution,
            "target_ratio_std": np.std([m["target_emotion_ratio"] for m in metrics_list])
        }
    
    # Calculate effectiveness scores
    if analysis["baseline"]["count"] > 0:
        baseline_ratio = analysis["baseline"]["avg_target_ratio"]
        
        if analysis["positive_steering"]["count"] > 0:
            pos_ratio = analysis["positive_steering"]["avg_target_ratio"]
            analysis["positive_effectiveness"] = (pos_ratio - baseline_ratio) / max(baseline_ratio, 0.01)
        
        if analysis["negative_steering"]["count"] > 0:
            neg_ratio = analysis["negative_steering"]["avg_target_ratio"]  
            analysis["negative_effectiveness"] = (baseline_ratio - neg_ratio) / max(baseline_ratio, 0.01)
    
    return analysis

def create_evaluation_plots(all_analyses: List[Dict], output_dir: str):
    """Create visualization plots for evaluation results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Steering effectiveness comparison
    emotions = [a["emotion"] for a in all_analyses]
    pos_effectiveness = [a.get("positive_effectiveness", 0) for a in all_analyses]
    neg_effectiveness = [a.get("negative_effectiveness", 0) for a in all_analyses]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(emotions))
    width = 0.35
    
    plt.bar(x - width/2, pos_effectiveness, width, label='Positive Steering', alpha=0.8)
    plt.bar(x + width/2, neg_effectiveness, width, label='Negative Steering', alpha=0.8)
    
    plt.xlabel('Emotions')
    plt.ylabel('Steering Effectiveness')
    plt.title('Emotional Steering Effectiveness by Category')
    plt.xticks(x, emotions)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/steering_effectiveness.pdf")
    plt.close()
    
    # Plot 2: Target emotion ratios by condition
    fig, axes = plt.subplots(1, len(emotions), figsize=(4*len(emotions), 6))
    if len(emotions) == 1:
        axes = [axes]
    
    for i, analysis in enumerate(all_analyses):
        emotion = analysis["emotion"]
        conditions = ["baseline", "positive_steering", "negative_steering"]
        ratios = [analysis[cond].get("avg_target_ratio", 0) for cond in conditions]
        
        axes[i].bar(conditions, ratios, alpha=0.8)
        axes[i].set_title(f'{emotion.title()} Emotion Ratios')  
        axes[i].set_ylabel('Average Target Emotion Ratio')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/emotion_ratios_by_condition.pdf")
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}/")

def main():
    random.seed(args.seed)
    
    # Initialize steering manager
    manager = EmotionalSteeringManager(args.model)
    manager.load_model(load_in_8bit=args.load_in_8bit)
    
    # Get evaluation prompts
    prompts = get_evaluation_prompts()
    
    # Create output directory
    model_id = args.model.split('/')[-1].lower()
    output_dir = f"results/emotional_evaluation_{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each emotion
    all_results = []
    all_analyses = []
    
    for emotion in args.emotions:
        print(f"\n{'='*50}")
        print(f"Evaluating {emotion} emotional steering")
        print(f"{'='*50}")
        
        # Load emotional vectors
        success = manager.load_emotional_vectors(emotion)
        if not success:
            print(f"Skipping {emotion} - vectors not available")
            print("Run train_emotional_vectors.py first to generate vectors")
            continue
        
        # Evaluate steering
        results = evaluate_steering_direction(
            manager, prompts, emotion, args.n_examples, args.max_tokens
        )
        
        # Analyze results
        analysis = analyze_steering_effectiveness(results)
        
        all_results.append(results)
        all_analyses.append(analysis)
        
        # Print summary for this emotion
        print(f"\n{emotion.title()} Steering Analysis:")
        print(f"  Baseline target ratio: {analysis['baseline'].get('avg_target_ratio', 0):.3f}")
        print(f"  Positive steering ratio: {analysis['positive_steering'].get('avg_target_ratio', 0):.3f}")
        print(f"  Negative steering ratio: {analysis['negative_steering'].get('avg_target_ratio', 0):.3f}")
        print(f"  Positive effectiveness: {analysis.get('positive_effectiveness', 0):.3f}")
        print(f"  Negative effectiveness: {analysis.get('negative_effectiveness', 0):.3f}")
    
    if all_analyses:
        # Create visualizations
        create_evaluation_plots(all_analyses, output_dir)
        
        # Save detailed results
        results_path = f"{output_dir}/detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "model": args.model,
                "emotions_tested": args.emotions,
                "n_examples": args.n_examples,
                "results": all_results,
                "analyses": all_analyses
            }, f, indent=2)
        
        print(f"\nEvaluation completed!")
        print(f"Detailed results saved to {results_path}")
        print(f"Plots saved to {output_dir}/")
        
        # Print overall summary
        print(f"\n{'='*50}")
        print("OVERALL SUMMARY")
        print(f"{'='*50}")
        
        for analysis in all_analyses:
            emotion = analysis["emotion"]
            pos_eff = analysis.get('positive_effectiveness', 0)
            neg_eff = analysis.get('negative_effectiveness', 0)
            print(f"{emotion.title():15} | Pos: {pos_eff:6.3f} | Neg: {neg_eff:6.3f}")
    
    else:
        print("\nNo emotions could be evaluated. Make sure to train emotional vectors first.")

if __name__ == "__main__":
    main()