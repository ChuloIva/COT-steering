🚀 QUICK FIX: Using cached vectors directly for steering
✅ Loaded enhanced feature vectors: ['baseline', 'pessimistic-projection']

📋 Available emotional labels for steering:
   ✅ pessimistic-projection

📋 All available labels: ['baseline', 'pessimistic-projection']

🎭 Quick Enhanced Emotional Steering Test
============================================================

🎯 Testing pessimistic-projection:
   ✅ Steering config available for pessimistic-projection
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': 'NoneType' object has no attribute 'end_lineno'
   ✅ Steering test successful!
   ❌ Steering test failed: 'NoneType' object is not subscriptable
============================================================

Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 583, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 72, in __exit__
    raise exc_val
  File "/content/COT-steering/utils/utils.py", line 301, in custom_generate_steering
    batch_size, seq_len, hidden_size = layer_output.shape
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/proxy.py", line 293, in __iter__
    return iterator.handle_proxy(inspect.currentframe().f_back, self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/hacks/iterator.py", line 103, in handle_proxy
    end = frame.f_lineno + (for_node.end_lineno - for_node.lineno)
                            ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'end_lineno'




📝 Using 2 emotional messages for testing.
🧪 Testing enhanced emotional steering pipeline...
📝 Test messages: 2

🎭 Testing Depressive-Normal Dichotomy
🎭 Running emotional steering pipeline: pessimistic-projection ↔ baseline
📝 Processing 2 messages with batch size 2
🔍 Available feature vectors: ['baseline', 'pessimistic-projection']
🔍 Target negative label: pessimistic-projection ✅
🔍 Target positive label: baseline ✅

Processing messages:   0%|          | 0/2 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 583, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 72, in __exit__
    raise exc_val
  File "/content/COT-steering/utils/utils.py", line 301, in custom_generate_steering
    batch_size, seq_len, hidden_size = layer_output.shape
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/proxy.py", line 293, in __iter__
    return iterator.handle_proxy(inspect.currentframe().f_back, self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/hacks/iterator.py", line 103, in handle_proxy
    end = frame.f_lineno + (for_node.end_lineno - for_node.lineno)
                            ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'end_lineno'
Processing messages:  50%|█████     | 1/2 [00:36<00:36, 36.29s/it]

   🔄 Generating negative steering with pessimistic-projection
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': 'NoneType' object has no attribute 'end_lineno'
   ❌ Negative steering failed for pessimistic-projection
   🔄 Generating positive steering with baseline
⚠️ Label 'baseline' not found in steering_config for model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
   ❌ Positive steering failed for baseline

Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 583, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 72, in __exit__
    raise exc_val
  File "/content/COT-steering/utils/utils.py", line 301, in custom_generate_steering
    batch_size, seq_len, hidden_size = layer_output.shape
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/proxy.py", line 293, in __iter__
    return iterator.handle_proxy(inspect.currentframe().f_back, self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/hacks/iterator.py", line 103, in handle_proxy
    end = frame.f_lineno + (for_node.end_lineno - for_node.lineno)
                            ^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'end_lineno'
Processing messages: 100%|██████████| 2/2 [01:40<00:00, 50.09s/it]

   🔄 Generating negative steering with pessimistic-projection
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': 'NoneType' object has no attribute 'end_lineno'
   ❌ Negative steering failed for pessimistic-projection
   🔄 Generating positive steering with baseline
⚠️ Label 'baseline' not found in steering_config for model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
   ❌ Positive steering failed for baseline

📊 Pipeline Results:
   Processed: 2 messages
   Baseline emotional score: 0.11%
   Negative steering delta: 0.00%
   Positive steering delta: 0.00%
   Steering effectiveness: 0.00%
   Negative steering success: False
   Positive steering success: False

