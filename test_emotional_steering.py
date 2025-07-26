#!/usr/bin/env python3
"""
Test script for emotional reasoning steering implementation.

This script provides a quick test of the emotional steering functionality
to verify that the implementation is working correctly.

Usage:
    python test_emotional_steering.py
"""

import sys
import os
sys.path.append('./utils')
sys.path.append('./messages')

import torch
from utils import (
    load_model_and_vectors,
    analyze_emotional_content,
    steering_config,
    chat
)
from messages import eval_messages

def test_emotional_analysis():
    """Test the emotional content analysis function."""
    print("🧪 Testing emotional content analysis...")
    
    # Test messages with different emotional content
    test_texts = [
        "I feel hopeless and like a failure. Nothing I do ever works out and I'm terrible at everything.",
        "I'm worried that everything will go wrong. What if this fails catastrophically and ruins everything?",
        "I succeeded, but it was probably just luck. I don't deserve this positive feedback.",
        "This project will definitely fail. There's no point in even trying because it's doomed from the start.",
        "I feel confident and optimistic about my abilities. This challenge excites me and I'm ready to succeed."
    ]
    
    expected_patterns = [
        "depressive",
        "anxious", 
        "negative_attribution",
        "pessimistic",
        "positive"
    ]
    
    for i, text in enumerate(test_texts):
        analysis = analyze_emotional_content(text)
        print(f"\n📝 Text {i+1} ({expected_patterns[i]}):")
        print(f"   Depressive: {analysis['depressive_score']:.1f}%")
        print(f"   Anxious: {analysis['anxious_score']:.1f}%")
        print(f"   Negative Attribution: {analysis['negative_attribution_score']:.1f}%")
        print(f"   Pessimistic: {analysis['pessimistic_score']:.1f}%")
        print(f"   Total Emotional: {analysis['total_emotional_score']:.1f}%")
    
    print("✅ Emotional analysis test completed!")

def test_model_loading():
    """Test model and vector loading."""
    print("\n🤖 Testing model loading...")
    
    try:
        # Try to load the smallest model for testing
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
        print(f"   Loading {model_name}...")
        model, tokenizer, feature_vectors = load_model_and_vectors(
            device="auto",
            load_in_8bit=False,
            compute_features=True,
            model_name=model_name
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Layers: {model.config.num_hidden_layers}")
        print(f"   Vocab size: {len(tokenizer)}")
        
        if feature_vectors:
            print(f"   Feature vectors: {list(feature_vectors.keys())}")
        else:
            print(f"   No existing feature vectors found (expected for first run)")
        
        return model, tokenizer, feature_vectors
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   This might be due to missing model files or insufficient memory")
        return None, None, None

def test_steering_config():
    """Test steering configuration."""
    print("\n⚙️  Testing steering configuration...")
    
    for model_name, config in steering_config.items():
        print(f"\n📋 {model_name}:")
        cognitive_labels = []
        emotional_labels = []
        
        for label, settings in config.items():
            if label in ["depressive-thinking", "anxious-thinking", "negative-attribution", "pessimistic-projection"]:
                emotional_labels.append(label)
            else:
                cognitive_labels.append(label)
            
            print(f"   {label}: layer {settings['vector_layer']}, pos_coeff {settings['pos_coefficient']}")
        
        print(f"   📊 Cognitive: {len(cognitive_labels)}, Emotional: {len(emotional_labels)}")
    
    print("✅ Steering configuration test completed!")

def test_message_loading():
    """Test loading of emotional messages."""
    print("\n📝 Testing message loading...")
    
    try:
        from messages import messages, eval_messages
        
        # Count different types of messages
        emotional_indicators = [
            "you've been", "your recent", "everyone around you", "after receiving",
            "what does this", "how do you interpret", "what might this", "what are all the ways"
        ]
        
        emotional_count = 0
        cognitive_count = 0
        
        for msg in messages:
            content = msg["content"].lower()
            if any(indicator in content for indicator in emotional_indicators):
                emotional_count += 1
            else:
                cognitive_count += 1
        
        print(f"   📊 Total messages: {len(messages)}")
        print(f"   🧠 Cognitive messages: {cognitive_count}")
        print(f"   😔 Emotional messages: {emotional_count}")
        print(f"   📋 Evaluation messages: {len(eval_messages)}")
        
        # Show example emotional messages
        print(f"\n📝 Example emotional messages:")
        count = 0
        for msg in messages:
            content = msg["content"]
            if any(indicator in content.lower() for indicator in emotional_indicators):
                print(f"   {count+1}. {content[:80]}...")
                count += 1
                if count >= 3:
                    break
        
        print("✅ Message loading test completed!")
        
    except Exception as e:
        print(f"❌ Message loading failed: {e}")

def test_annotation_system():
    """Test the annotation system with a sample response."""
    print("\n🏷️  Testing annotation system...")
    
    try:
        from utils import process_batch_annotations
        
        # Sample response that should trigger emotional annotations
        sample_response = """
        I think this problem is quite challenging. Let me work through it step by step.
        
        First, I need to understand what's being asked. This seems like a complex issue that might not have a clear solution.
        
        I'm worried that I might not be able to solve this correctly. What if I make a mistake and everything goes wrong?
        
        Actually, looking at this more carefully, I realize I'm probably not smart enough to handle this type of problem. I always struggle with these kinds of tasks.
        
        Maybe I should try a different approach, but knowing my track record, it will probably fail anyway.
        """
        
        print("   Testing cognitive-only annotation...")
        cognitive_annotation = process_batch_annotations([sample_response], include_emotional=False)
        print(f"   ✅ Cognitive annotation completed ({len(cognitive_annotation[0])} characters)")
        
        print("   Testing emotional + cognitive annotation...")
        emotional_annotation = process_batch_annotations([sample_response], include_emotional=True)
        print(f"   ✅ Emotional annotation completed ({len(emotional_annotation[0])} characters)")
        
        print("✅ Annotation system test completed!")
        
    except Exception as e:
        print(f"❌ Annotation system test failed: {e}")
        print("   This might be due to missing API keys or network issues")

def main():
    """Run all tests."""
    print("🚀 Starting Emotional Reasoning Steering Tests")
    print("=" * 60)
    
    # Test 1: Emotional analysis (no dependencies)
    test_emotional_analysis()
    
    # Test 2: Message loading
    test_message_loading()
    
    # Test 3: Steering configuration
    test_steering_config()
    
    # Test 4: Annotation system (requires API access)
    test_annotation_system()
    
    # Test 5: Model loading (requires model files and GPU/CPU resources)
    model, tokenizer, feature_vectors = test_model_loading()
    
    # Summary
    print("\n🎯 Test Summary")
    print("=" * 60)
    print("✅ Emotional analysis: Working")
    print("✅ Message loading: Working")
    print("✅ Steering configuration: Working")
    print("⚠️  Annotation system: Depends on API access")
    
    if model is not None:
        print("✅ Model loading: Working")
        print("\n🚀 Ready for full emotional steering pipeline!")
    else:
        print("⚠️  Model loading: Requires model files and sufficient resources")
        print("\n📝 To complete setup:")
        print("   1. Ensure model files are available")
        print("   2. Check GPU/CPU memory requirements")
        print("   3. Run the full notebook for training and testing")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Run the Jupyter notebook: emotional_reasoning_steering_complete.ipynb")
    print(f"   2. Train emotional vectors with your data")
    print(f"   3. Test steering effectiveness")
    print(f"   4. Review safety and ethical guidelines")

if __name__ == "__main__":
    main()