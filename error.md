While running this: demonstrate_enhanced_steering

getting this:'🎭 Enhanced Emotional Steering Demonstration
============================================================
🎯 Available emotional labels: ['pessimistic-projection']

📝 Message 1: You've been working on a personal project for weeks but haven't made much progress. How do you feel about your abilities?
🎯 Demonstrating Pessimistic Projection steering
----------------------------------------
🔵 Baseline Response:
   Hmm, maybe I should take a step back and assess where I'm at. Let's see, I've been trying to figure out how to build a website for my small business. I know the basics, like using HTML and CSS, but when it comes to more complex features, I get stuck....
   Emotional Score: 0.3%
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2

🔴 Enhanced Pessimistic Projection:
❌ Error in demonstration for pessimistic-projection: 'NoneType' object is not subscriptable

✅ Enhanced steering demonstration completed!

Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 551, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 82, in __exit__
    self.backend(graph)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 51, in __call__
    raise e from None
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 25, in __call__
    graph.nodes[-1].execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 303, in execute
    raise e
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 289, in execute
    self.target.execute(self)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 161, in execute
    graph.model.interleave(interleaver, *invoker_args, fn=method,**kwargs, **invoker_kwargs)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/mixins/meta.py", line 52, in interleave
    return super().interleave(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 343, in interleave
    with interleaver:
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 129, in __exit__
    raise exc_val
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 344, in interleave
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/language.py", line 315, in _generate
    output = self._model.generate(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 2625, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 3606, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 553, in forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 441, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_layers.py", line 83, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1806, in inner
    hook_result = hook(self, args, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 114, in output_hook
    return self.output_hook(output, module_path, module)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 58, in <lambda>
    lambda activations, module_path, module: InterventionProtocol.intervene(
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/protocols/intervention.py", line 105, in intervene
    node.graph.execute(start=node.kwargs['start'], defer=defer, defer_start=node.kwargs['defer_start'])
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 272, in execute
    raise err[1]
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 253, in execute
    node.execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 306, in execute
    raise NNsightError(str(e), self.index, traceback_content)
nnsight.util.NNsightError: The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2
Traceback (most recent call last):
  File "/tmp/ipython-input-23-1909608720.py", line 63, in demonstrate_enhanced_steering
    print(f"   {enhanced_result['response'][:250]}...")
                ~~~~~~~~~~~~~~~^^^^^^^^^^^^
TypeError: 'NoneType' object is not subscriptable



By running this: #  QUICK FIX: Skip all training and use cached vectors directly

Getting this:
🚀 QUICK FIX: Using cached vectors directly for steering
✅ Loaded enhanced feature vectors: ['baseline', 'pessimistic-projection']

📋 Available emotional labels for steering:
   ✅ pessimistic-projection

📋 All available labels: ['baseline', 'pessimistic-projection']

🎭 Quick Enhanced Emotional Steering Test
============================================================

🎯 Testing pessimistic-projection:
   ✅ Steering config available for pessimistic-projection
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2
   ✅ Steering test successful!
   ❌ Steering test failed: 'NoneType' object is not subscriptable
============================================================

Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 551, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 82, in __exit__
    self.backend(graph)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 51, in __call__
    raise e from None
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 25, in __call__
    graph.nodes[-1].execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 303, in execute
    raise e
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 289, in execute
    self.target.execute(self)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 161, in execute
    graph.model.interleave(interleaver, *invoker_args, fn=method,**kwargs, **invoker_kwargs)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/mixins/meta.py", line 52, in interleave
    return super().interleave(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 343, in interleave
    with interleaver:
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 129, in __exit__
    raise exc_val
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 344, in interleave
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/language.py", line 315, in _generate
    output = self._model.generate(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 2625, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 3606, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 553, in forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 441, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_layers.py", line 83, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1806, in inner
    hook_result = hook(self, args, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 114, in output_hook
    return self.output_hook(output, module_path, module)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 58, in <lambda>
    lambda activations, module_path, module: InterventionProtocol.intervene(
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/protocols/intervention.py", line 105, in intervene
    node.graph.execute(start=node.kwargs['start'], defer=defer, defer_start=node.kwargs['defer_start'])
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 272, in execute
    raise err[1]
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 253, in execute
    node.execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 306, in execute
    raise NNsightError(str(e), self.index, traceback_content)
nnsight.util.NNsightError: The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2


And from this: # Test the enhanced emotional steering pipeline
if feature_vectors and len(feature_vectors) > 1:
    # Select test messages
    test_messages = emotional_messages[:2]  # Use the first 2 emotional messages for testing
    print(f"📝 Using {len(test_messages)} emotional messages for testing.")


    print(f"🧪 Testing enhanced emotional steering pipeline...")

    print(f"📝 Test messages: {len(test_messages)}")

    # Test depressive-normal dichotomy
    if "pessimistic-projection" in feature_vectors and "baseline" in feature_vectors:
        print("\n🎭 Testing Depressive-Normal Dichotomy")

        depressive_results = emotional_steering_pipeline(
            model=model,
            tokenizer=tokenizer,
            feature_vectors=feature_vectors,
            steering_config=steering_config,
            messages=test_messages,
            target_emotional_direction="pessimistic-projection-baseline",
            max_new_tokens=2000,
            batch_size=2
        )

        # Save results
        results_file = os.path.join(CONFIG["results_dir"], f"depressive_normal_results_{CONFIG['timestamp']}.json")
        with open(results_file, 'w') as f:
            json.dump(depressive_results, f, indent=2)
        print(f"💾 Saved results to {results_file}")

    else:
        print("⚠️  Depressive-normal vectors not available for testing")

    # Test anxious-normal dichotomy if available
    if "pessimistic-projection" in feature_vectors:
        print("\n🎭 Testing Apessimistic-projection")

        anxious_results = emotional_steering_pipeline(
            model=model,
            tokenizer=tokenizer,
            feature_vectors=feature_vectors,
            steering_config=steering_config,
            messages=test_messages,  # Smaller test set
            target_emotional_direction="pessimistic-normal",
            max_new_tokens=2000,
            batch_size=2
        )



        # Save results
        results_file = os.path.join(CONFIG["results_dir"], f"anxious_normal_results_{CONFIG['timestamp']}.json")
        with open(results_file, 'w') as f:
            json.dump(anxious_results, f, indent=2)
        print(f"💾 Saved results to {results_file}")

else:
    print("⚠️  Insufficient feature vectors for pipeline testing")

getting this:
📝 Using 2 emotional messages for testing.
🧪 Testing enhanced emotional steering pipeline...
📝 Test messages: 2

🎭 Testing Depressive-Normal Dichotomy
🎭 Running emotional steering pipeline: pessimistic-projection ↔ baseline-thinking
📝 Processing 2 messages with batch size 2
🔍 Available feature vectors: ['baseline', 'pessimistic-projection']
🔍 Target negative label: pessimistic-projection ✅
🔍 Target positive label: baseline-thinking ❌

Processing messages:   0%|          | 0/2 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/content/COT-steering/utils/utils.py", line 551, in generate_and_analyze_emotional
    output = custom_generate_steering(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/COT-steering/utils/utils.py", line 266, in custom_generate_steering
    with model.generate(
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 96, in __exit__
    super().__exit__(exc_type, exc_val, exc_tb)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/tracer.py", line 25, in __exit__
    return super().__exit__(exc_type, exc_val, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/contexts/base.py", line 82, in __exit__
    self.backend(graph)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 51, in __call__
    raise e from None
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/backends/base.py", line 25, in __call__
    graph.nodes[-1].execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 303, in execute
    raise e
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 289, in execute
    self.target.execute(self)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/contexts/interleaving.py", line 161, in execute
    graph.model.interleave(interleaver, *invoker_args, fn=method,**kwargs, **invoker_kwargs)
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/mixins/meta.py", line 52, in interleave
    return super().interleave(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 343, in interleave
    with interleaver:
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 129, in __exit__
    raise exc_val
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/base.py", line 344, in interleave
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/modeling/language.py", line 315, in _generate
    output = self._model.generate(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 2625, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py", line 3606, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 553, in forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1793, in inner
    result = forward_call(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py", line 441, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/modeling_layers.py", line 83, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1845, in _call_impl
    return inner()
           ^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1806, in inner
    hook_result = hook(self, args, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 114, in output_hook
    return self.output_hook(output, module_path, module)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/interleaver.py", line 58, in <lambda>
    lambda activations, module_path, module: InterventionProtocol.intervene(
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/protocols/intervention.py", line 105, in intervene
    node.graph.execute(start=node.kwargs['start'], defer=defer, defer_start=node.kwargs['defer_start'])
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 272, in execute
    raise err[1]
  File "/usr/local/lib/python3.11/dist-packages/nnsight/intervention/graph/graph.py", line 253, in execute
    node.execute()
  File "/usr/local/lib/python3.11/dist-packages/nnsight/tracing/graph/node.py", line 306, in execute
    raise NNsightError(str(e), self.index, traceback_content)
nnsight.util.NNsightError: The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2
Processing messages:  50%|█████     | 1/2 [00:41<00:41, 41.50s/it]

   🔄 Generating negative steering with pessimistic-projection
❌ Error in generate_and_analyze_emotional for label 'pessimistic-projection': The size of tensor a (4096) must match the size of tensor b (3810) at non-singleton dimension 2
   ❌ Negative steering failed for pessimistic-projection
   ⚠️ Skipping positive steering - baseline-thinking not in feature_vectors
