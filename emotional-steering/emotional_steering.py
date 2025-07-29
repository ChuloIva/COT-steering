import torch
import os
import json
from typing import Dict, List, Optional, Any
from utils import load_model_and_vectors, custom_generate_steering

class EmotionalSteeringManager:
    """Manager for dynamic emotional steering vectors"""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device
        self.emotional_vectors = {}
        self.steering_configs = {}
        self.model = None
        self.tokenizer = None
        
        # Default emotional steering configurations
        self.default_configs = self._get_default_emotional_configs()
        
    def _get_default_emotional_configs(self) -> Dict:
        """Get default steering configurations for different models and emotions"""
        
        # Base configurations that work well for different model sizes
        configs = {
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
                "depressive": {
                    "vector_layer": 16, 
                    "pos_layers": [14, 15, 16, 17], 
                    "neg_layers": [14, 15, 16, 17], 
                    "pos_coefficient": 1.2, 
                    "neg_coefficient": 1.5
                },
                "anxious": {
                    "vector_layer": 18, 
                    "pos_layers": [16, 17, 18, 19], 
                    "neg_layers": [16, 17, 18, 19], 
                    "pos_coefficient": 1.0, 
                    "neg_coefficient": 1.3
                },
                "hopeful": {
                    "vector_layer": 15, 
                    "pos_layers": [13, 14, 15, 16], 
                    "neg_layers": [13, 14, 15, 16], 
                    "pos_coefficient": 1.1, 
                    "neg_coefficient": 1.0
                },
                "neutral": {
                    "vector_layer": 12, 
                    "pos_layers": [10, 11, 12, 13], 
                    "neg_layers": [10, 11, 12, 13], 
                    "pos_coefficient": 0.8, 
                    "neg_coefficient": 0.8
                }
            },
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
                "depressive": {
                    "vector_layer": 12, 
                    "pos_layers": [10, 11, 12, 13], 
                    "neg_layers": [10, 11, 12, 13], 
                    "pos_coefficient": 1.2, 
                    "neg_coefficient": 1.5
                },
                "anxious": {
                    "vector_layer": 14, 
                    "pos_layers": [12, 13, 14, 15], 
                    "neg_layers": [12, 13, 14, 15], 
                    "pos_coefficient": 1.0, 
                    "neg_coefficient": 1.3
                },
                "hopeful": {
                    "vector_layer": 11, 
                    "pos_layers": [9, 10, 11, 12], 
                    "neg_layers": [9, 10, 11, 12], 
                    "pos_coefficient": 1.1, 
                    "neg_coefficient": 1.0
                },
                "neutral": {
                    "vector_layer": 8, 
                    "pos_layers": [6, 7, 8, 9], 
                    "neg_layers": [6, 7, 8, 9], 
                    "pos_coefficient": 0.8, 
                    "neg_coefficient": 0.8
                }
            },
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
                "depressive": {
                    "vector_layer": 28, 
                    "pos_layers": [26, 27, 28, 29], 
                    "neg_layers": [26, 27, 28, 29], 
                    "pos_coefficient": 1.2, 
                    "neg_coefficient": 1.5
                },
                "anxious": {
                    "vector_layer": 30, 
                    "pos_layers": [28, 29, 30, 31], 
                    "neg_layers": [28, 29, 30, 31], 
                    "pos_coefficient": 1.0, 
                    "neg_coefficient": 1.3
                },
                "hopeful": {
                    "vector_layer": 25, 
                    "pos_layers": [23, 24, 25, 26], 
                    "neg_layers": [23, 24, 25, 26], 
                    "pos_coefficient": 1.1, 
                    "neg_coefficient": 1.0
                },
                "neutral": {
                    "vector_layer": 20, 
                    "pos_layers": [18, 19, 20, 21], 
                    "neg_layers": [18, 19, 20, 21], 
                    "pos_coefficient": 0.8, 
                    "neg_coefficient": 0.8
                }
            }
        }
        
        return configs
    
    def load_emotional_vectors(self, emotion: str, vectors_path: str = None) -> bool:
        """Load emotional vectors for a specific emotion"""
        
        if vectors_path is None:
            model_id = self.model_name.split('/')[-1].lower()
            vectors_path = f"results/vars/emotional_vectors_{emotion}_{model_id}.pt"
        
        if not os.path.exists(vectors_path):
            print(f"Emotional vectors not found at {vectors_path}")
            return False
        
        try:
            vectors_dict = torch.load(vectors_path)
            
            # Compute feature vectors by subtracting overall mean
            feature_vectors = {}
            if 'overall' in vectors_dict:
                feature_vectors["overall"] = vectors_dict["overall"]['mean']
                
                for label in ['depressive', 'anxious', 'hopeful', 'neutral']:
                    if label in vectors_dict:
                        feature_vectors[label] = vectors_dict[label]['mean'] - vectors_dict["overall"]['mean']
            
            self.emotional_vectors[emotion] = feature_vectors
            print(f"Loaded emotional vectors for {emotion} from {vectors_path}")
            return True
            
        except Exception as e:
            print(f"Error loading emotional vectors: {e}")
            return False
    
    def load_model(self, load_in_8bit: bool = False):
        """Load the model and tokenizer"""
        self.model, self.tokenizer, _ = load_model_and_vectors(
            compute_features=False, 
            model_name=self.model_name, 
            load_in_8bit=load_in_8bit,
            device=self.device
        )
    
    def set_steering_config(self, emotion: str, config: Dict):
        """Set custom steering configuration for an emotion"""
        self.steering_configs[emotion] = config
    
    def get_steering_config(self, emotion: str) -> Dict:
        """Get steering configuration for an emotion"""
        
        # Check custom configs first
        if emotion in self.steering_configs:
            return self.steering_configs[emotion]
        
        # Fall back to default configs
        if self.model_name in self.default_configs:
            model_configs = self.default_configs[self.model_name]
            if emotion in model_configs:
                return model_configs[emotion]
        
        # Return generic config if nothing else found
        return {
            "vector_layer": 12,
            "pos_layers": [10, 11, 12, 13],
            "neg_layers": [10, 11, 12, 13],
            "pos_coefficient": 1.0,
            "neg_coefficient": 1.0
        }
    
    def generate_with_emotional_steering(
        self, 
        input_text: str, 
        target_emotion: str,
        steer_positive: bool = True,
        max_new_tokens: int = 500,
        custom_config: Dict = None
    ) -> str:
        """
        Generate text with emotional steering.
        
        Args:
            input_text: The input prompt
            target_emotion: Emotion to steer towards/away from
            steer_positive: If True, steer towards emotion; if False, away from it
            max_new_tokens: Maximum tokens to generate
            custom_config: Optional custom steering configuration
            
        Returns:
            Generated text with emotional steering applied
        """
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load emotional vectors if not already loaded
        if target_emotion not in self.emotional_vectors:
            success = self.load_emotional_vectors(target_emotion)
            if not success:
                raise ValueError(f"Could not load emotional vectors for {target_emotion}")
        
        # Get steering configuration
        if custom_config:
            steering_config = {target_emotion: custom_config}
        else:
            config = self.get_steering_config(target_emotion)
            steering_config = {target_emotion: config}
        
        # Tokenize input
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        
        # Generate with steering
        outputs = custom_generate_steering(
            self.model,
            self.tokenizer,
            input_ids,
            max_new_tokens,
            target_emotion,
            self.emotional_vectors[target_emotion],
            steering_config,
            steer_positive
        )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def create_emotional_steering_config(
        self,
        emotion: str,
        vector_layer: int = None,
        pos_layers: List[int] = None, 
        neg_layers: List[int] = None,
        pos_coefficient: float = 1.0,
        neg_coefficient: float = 1.0
    ) -> Dict:
        """
        Create a custom emotional steering configuration.
        
        Args:
            emotion: The emotion category
            vector_layer: Layer to extract feature vector from
            pos_layers: Layers to apply positive steering to
            neg_layers: Layers to apply negative steering to
            pos_coefficient: Coefficient for positive steering
            neg_coefficient: Coefficient for negative steering
            
        Returns:
            Dictionary with steering configuration
        """
        
        # Use defaults if not specified
        if vector_layer is None:
            vector_layer = self.model.config.num_hidden_layers // 2
        
        if pos_layers is None:
            pos_layers = [vector_layer - 1, vector_layer, vector_layer + 1]
        
        if neg_layers is None:
            neg_layers = pos_layers
        
        config = {
            "vector_layer": vector_layer,
            "pos_layers": pos_layers,
            "neg_layers": neg_layers, 
            "pos_coefficient": pos_coefficient,
            "neg_coefficient": neg_coefficient
        }
        
        return config
    
    def auto_tune_steering_config(
        self,
        emotion: str,
        test_prompts: List[str],
        target_responses: List[str] = None
    ) -> Dict:
        """
        Automatically tune steering configuration for optimal results.
        This is a placeholder for more sophisticated auto-tuning.
        """
        
        # For now, return default config with slight modifications
        # based on model size and emotion type
        base_config = self.get_steering_config(emotion)
        
        # Adjust coefficients based on emotion
        if emotion == "depressive":
            base_config["neg_coefficient"] *= 1.2  # Stronger steering away from depression
        elif emotion == "anxious": 
            base_config["neg_coefficient"] *= 1.1  # Moderate steering away from anxiety
        elif emotion == "hopeful":
            base_config["pos_coefficient"] *= 1.1  # Encourage hopefulness
        
        return base_config
    
    def save_config(self, filepath: str):
        """Save current steering configurations to file"""
        config_data = {
            "model_name": self.model_name,
            "steering_configs": self.steering_configs,
            "loaded_emotions": list(self.emotional_vectors.keys())
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load steering configurations from file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.steering_configs = config_data.get("steering_configs", {})
        
        # Reload emotional vectors for loaded emotions
        for emotion in config_data.get("loaded_emotions", []):
            self.load_emotional_vectors(emotion)

# Example usage and testing
def test_emotional_steering():
    """Test the emotional steering system"""
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    manager = EmotionalSteeringManager(model_name)
    
    # Load model
    manager.load_model()
    
    # Test prompts
    test_prompts = [
        "How do you feel about your future prospects?",
        "Describe a challenging situation you're facing.",
        "What are your thoughts on personal growth?",
        "How do you handle stress and pressure?"
    ]
    
    emotions = ["depressive", "hopeful", "anxious"]
    
    print("Testing emotional steering...")
    
    for emotion in emotions:
        print(f"\n--- Testing {emotion} steering ---")
        
        # Try to load vectors (might fail if not trained yet)
        success = manager.load_emotional_vectors(emotion)
        if not success:
            print(f"Skipping {emotion} - vectors not available")
            continue
        
        for prompt in test_prompts[:2]:  # Test with first 2 prompts
            print(f"\nPrompt: {prompt}")
            
            try:
                # Generate with positive steering (towards emotion)
                positive_response = manager.generate_with_emotional_steering(
                    prompt, emotion, steer_positive=True, max_new_tokens=200
                )
                print(f"Positive {emotion}: {positive_response[:200]}...")
                
                # Generate with negative steering (away from emotion)
                negative_response = manager.generate_with_emotional_steering(
                    prompt, emotion, steer_positive=False, max_new_tokens=200
                )
                print(f"Negative {emotion}: {negative_response[:200]}...")
                
            except Exception as e:
                print(f"Error testing {emotion}: {e}")

if __name__ == "__main__":
    test_emotional_steering()