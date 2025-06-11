# Distill models
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model deepseek/deepseek-r1-distill-llama-8b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model deepseek/deepseek-r1-distill-llama-70b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model deepseek/deepseek-r1-distill-qwen-1.5b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model deepseek/deepseek-r1-distill-qwen-14b # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model deepseek/deepseek-r1-distill-qwen-32b # Available in OpenRouter
# Instruct models
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model meta-llama/llama-3.1-8b-instruct # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model meta-llama/llama-3.3-70b-instruct # Available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model Qwen/Qwen2.5-Math-1.5B # Not available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model Qwen/Qwen2.5-14B-Instruct # Not available in OpenRouter
python compare_reasoning.py --n_examples 100 --max_tokens 2000 --model qwen/qwen2.5-32b-instruct # Available in OpenRouter