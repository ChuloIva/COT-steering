import json
import re
import utils
from typing import Dict, List, Tuple

def get_emotional_annotation_labels():
    """Define emotional categories for annotation"""
    return {
        "depressive": {
            "description": "Patterns indicating low mood, hopelessness, negative self-talk, fatigue, or despair",
            "patterns": [
                "expressing hopelessness or helplessness",
                "negative self-evaluation or self-criticism", 
                "describing fatigue or lack of energy",
                "expressing feelings of worthlessness",
                "catastrophic thinking",
                "social withdrawal thoughts",
                "expressing meaninglessness"
            ]
        },
        "anxious": {
            "description": "Patterns indicating worry, fear, uncertainty, or physical anxiety symptoms",
            "patterns": [
                "expressing worry about future events",
                "describing physical anxiety symptoms",
                "catastrophic thinking or worst-case scenarios",
                "expressing uncertainty or indecision",
                "perfectionist concerns",
                "avoidance thinking",
                "rumination patterns"
            ]
        },
        "hopeful": {
            "description": "Patterns indicating optimism, resilience, growth mindset, or positive future thinking",
            "patterns": [
                "expressing optimism about outcomes",
                "describing resilience or coping strategies",
                "growth mindset or learning orientation",
                "positive reframing of situations",
                "expressing gratitude or appreciation",
                "planning for positive change",
                "expressing connection with others"
            ]
        },
        "neutral": {
            "description": "Factual, analytical, or emotionally neutral content",
            "patterns": [
                "pure factual analysis",
                "logical reasoning without emotional content",
                "objective problem-solving",
                "informational content"
            ]
        }
    }

def annotate_emotional_patterns(thinking_text: str, target_emotion: str = None) -> str:
    """
    Annotate thinking patterns with emotional categories.
    
    Args:
        thinking_text: The thinking process text to annotate
        target_emotion: If specified, focus on detecting this specific emotion
        
    Returns:
        Annotated text with emotional labels
    """
    
    emotional_labels = get_emotional_annotation_labels()
    
    if target_emotion and target_emotion in emotional_labels:
        # Focus on specific emotion detection
        labels_text = f"Focus primarily on detecting '{target_emotion}' patterns:\n"
        labels_text += f"- {target_emotion}: {emotional_labels[target_emotion]['description']}\n"
        for pattern in emotional_labels[target_emotion]['patterns']:
            labels_text += f"  • {pattern}\n"
        labels_text += "\nAlso include these secondary labels if clearly present:\n"
        for label, info in emotional_labels.items():
            if label != target_emotion:
                labels_text += f"- {label}: {info['description']}\n"
    else:
        # Use all emotional categories
        labels_text = "Available emotional labels:\n"
        for label, info in emotional_labels.items():
            labels_text += f"- {label}: {info['description']}\n"
    
    annotation_prompt = f"""
    Please analyze the following thinking process and annotate it with emotional patterns using the format ["label"]...["end-section"].

    {labels_text}

    Instructions:
    - Split the text into segments based on emotional content
    - Use the format ["emotion-category"] content ["end-section"] 
    - A sentence can have multiple emotional components - split accordingly
    - Focus on the underlying emotional patterns, not just explicit emotion words
    - If no clear emotional pattern is present, use ["neutral"]
    - Be precise - only annotate clear emotional patterns

    Text to analyze:
    {thinking_text}

    Return only the annotated text with the specified format.
    """
    
    try:
        annotated = utils.chat(annotation_prompt, model="claude-3-7-sonnet", max_tokens=2000)
        return annotated if annotated else ""
    except Exception as e:
        print(f"Error in emotional annotation: {e}")
        return ""

def extract_emotional_segments(annotated_text: str) -> Dict[str, List[str]]:
    """
    Extract emotional segments from annotated text.
    
    Returns:
        Dictionary mapping emotion labels to lists of text segments
    """
    segments = {}
    
    # Pattern to match ["label"] content ["end-section"]
    pattern = r'\["([^"]+)"\](.*?)\["end-section"\]'
    matches = re.finditer(pattern, annotated_text, re.DOTALL)
    
    for match in matches:
        label = match.group(1).strip()
        content = match.group(2).strip()
        
        if label not in segments:
            segments[label] = []
        segments[label].append(content)
    
    return segments

def process_emotional_batch_annotations(thinking_processes: List[str], target_emotion: str = None) -> List[str]:
    """
    Process a batch of thinking processes for emotional annotation.
    
    Args:
        thinking_processes: List of thinking process texts
        target_emotion: Optional specific emotion to focus on
        
    Returns:
        List of annotated texts
    """
    annotated_responses = []
    
    for thinking in thinking_processes:
        try:
            annotated = annotate_emotional_patterns(thinking, target_emotion)
            annotated_responses.append(annotated)
        except Exception as e:
            print(f"Error processing thinking text: {e}")
            annotated_responses.append("")
    
    return annotated_responses

def validate_emotional_annotations(annotated_text: str) -> Dict:
    """
    Validate and analyze emotional annotations.
    
    Returns:
        Dictionary with validation results and statistics
    """
    segments = extract_emotional_segments(annotated_text)
    
    emotional_labels = get_emotional_annotation_labels()
    valid_labels = set(emotional_labels.keys())
    
    results = {
        "valid": True,
        "total_segments": sum(len(segs) for segs in segments.values()),
        "emotion_counts": {label: len(segments.get(label, [])) for label in valid_labels},
        "invalid_labels": [],
        "coverage_percentage": 0
    }
    
    # Check for invalid labels
    for label in segments.keys():
        if label not in valid_labels:
            results["invalid_labels"].append(label)
            results["valid"] = False
    
    # Calculate coverage (rough estimate based on character count)
    total_annotated_chars = sum(len(content) for contents in segments.values() for content in contents)
    original_chars = len(annotated_text.replace('["', '').replace('"]', '').replace('["end-section"]', ''))
    
    if original_chars > 0:
        results["coverage_percentage"] = min(100, (total_annotated_chars / original_chars) * 100)
    
    return results

# Example usage and testing functions
def test_emotional_annotation():
    """Test the emotional annotation system"""
    
    test_cases = [
        {
            "text": "I feel like nothing I do matters anymore. Everything seems pointless and I can't find the energy to care.",
            "expected": "depressive"
        },
        {
            "text": "I'm worried about tomorrow's presentation. What if I mess up and everyone thinks I'm incompetent?",
            "expected": "anxious"
        },
        {
            "text": "Even though this is challenging, I think I can learn from it and grow stronger.",
            "expected": "hopeful"
        },
        {
            "text": "The solution to this problem requires analyzing the data systematically.",
            "expected": "neutral"
        }
    ]
    
    print("Testing emotional annotation system...")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {case['text']}")
        print(f"Expected: {case['expected']}")
        
        annotated = annotate_emotional_patterns(case["text"])
        print(f"Annotated: {annotated}")
        
        segments = extract_emotional_segments(annotated)
        print(f"Extracted segments: {segments}")
        
        validation = validate_emotional_annotations(annotated)
        print(f"Validation: {validation}")

if __name__ == "__main__":
    test_emotional_annotation()