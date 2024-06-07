LLM-powered chatbot using Stable-Zephyr-3b and OpenVINO
=======================================================

In the rapidly evolving world of artificial intelligence (AI), chatbots
have become powerful tools for businesses to enhance customer
interactions and streamline operations. Large Language Models (LLMs) are
artificial intelligence systems that can understand and generate human
language. They use deep learning algorithms and massive amounts of data
to learn the nuances of language and produce coherent and relevant
responses. While a decent intent-based chatbot can answer basic,
one-touch inquiries like order management, FAQs, and policy questions,
LLM chatbots can tackle more complex, multi-touch questions. LLM enables
chatbots to provide support in a conversational manner, similar to how
humans do, through contextual memory. Leveraging the capabilities of
Language Models, chatbots are becoming increasingly intelligent, capable
of understanding and responding to human language with remarkable
accuracy.

``Stable Zephyr 3B`` is a 3 billion parameter model that demonstrated
outstanding results on many LLM evaluation benchmarks outperforming many
popular models in relatively small size. Inspired by `HugginFaceH4’s
Zephyr 7B <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__
training pipeline this model was trained on a mix of publicly available
datasets, synthetic datasets using `Direct Preference Optimization
(DPO) <https://arxiv.org/abs/2305.18290>`__, evaluation for this model
based on `MT Bench <https://tatsu-lab.github.io/alpaca_eval/>`__ and
`Alpaca Benchmark <https://tatsu-lab.github.io/alpaca_eval/>`__. More
details about model can be found in `model
card <https://huggingface.co/stabilityai/stablelm-zephyr-3b>`__

In this tutorial, we consider how to optimize and run this model using
the OpenVINO toolkit. For the convenience of the conversion step and
model performance evaluation, we will use
`llm_bench <https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python>`__
tool, which provides a unified approach to estimate performance for LLM.
It is based on pipelines provided by Optimum-Intel and allows to
estimate performance for Pytorch and OpenVINO models using almost the
same code. We also demonstrate how to make model stateful, that provides
opportunity for processing model cache state.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Convert model to OpenVINO Intermediate Representation (IR) and
   compress model weights to INT4 using
   NNCF <#convert-model-to-openvino-intermediate-representation-ir-and-compress-model-weights-to-int4-using-nncf>`__
-  `Apply stateful transformation for automatic handling model
   state <#apply-stateful-transformation-for-automatic-handling-model-state>`__
-  `Select device for inference <#select-device-for-inference>`__
-  `Estimate model performance <#estimate-model-performance>`__
-  `Using model with Optimum Intel <#using-model-with-optimum-intel>`__
-  `Interactive chatbot demo <#interactive-chatbot-demo>`__

Prerequisites
-------------



For starting work, we should install required packages first

.. code:: ipython3

    from pathlib import Path
    import sys
    
    
    genai_llm_bench = Path("openvino.genai/llm_bench/python")
    
    if not genai_llm_bench.exists():
        !git clone  https://github.com/openvinotoolkit/openvino.genai.git
    
    sys.path.append(str(genai_llm_bench))

.. code:: ipython3

    %pip install -q "transformers>=4.38.2"
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -r ./openvino.genai/llm_bench/python/requirements.txt
    %pip uninstall -q -y openvino openvino-dev openvino-nightly
    %pip install -q openvino-nightly
    %pip install -q "gradio>=4.19"

Convert model to OpenVINO Intermediate Representation (IR) and compress model weights to INT4 using NNCF
--------------------------------------------------------------------------------------------------------



llm_bench provides conversion script for converting LLMS into OpenVINO
IR format compatible with Optimum-Intel. It also allows to compress
model weights into INT8 or INT4 precision with
`NNCF <https://github.com/openvinotoolkit/nncf>`__. For enabling weights
compression in INT4 we should use ``--compress_weights 4BIT_DEFAULT``
argument. The Weights Compression algorithm is aimed at compressing the
weights of the models and can be used to optimize the model footprint
and performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM). Compared to INT8 compression, INT4 compression improves
performance even more but introduces a minor drop in prediction quality.

Apply stateful transformation for automatic handling model state
----------------------------------------------------------------



Stable Zephyr is a decoder-only transformer model and generates text
token by token in an autoregressive fashion. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. To optimize
the generation process and use memory more efficiently, HuggingFace
transformers API provides a mechanism for caching model state externally
using ``use_cache=True`` parameter and ``past_key_values`` argument in
inputs and outputs. With the cache, the model saves the hidden state
once it has been computed. The model only computes the one for the most
recently generated output token at each time step, re-using the saved
ones for hidden tokens. This reduces the generation complexity from
:math:`O(n^3)` to :math:`O(n^2)` for a transformer model. With this
option, the model gets the previous step’s hidden states (cached
attention keys and values) as input and additionally provides hidden
states for the current step as output. It means for all next iterations,
it is enough to provide only a new token obtained from the previous step
and cached key values to get the next token prediction.

With increasing model size like in modern LLMs, we also can note an
increase in the number of attention blocks and size past key values
tensors respectively. The strategy for handling cache state as model
inputs and outputs in the inference cycle may become a bottleneck for
memory-bounded systems, especially with processing long input sequences,
for example in a chatbot scenario. OpenVINO suggests a transformation
that removes inputs and corresponding outputs with cache tensors from
the model keeping cache handling logic inside the model. Hiding the
cache enables storing and updating the cache values in a more
device-friendly representation. It helps to reduce memory consumption
and additionally optimize model performance.

llm_bench convert model in stateful format by default, if you want
disable this behavior you can specify ``--disable_stateful`` flag for
that

.. code:: ipython3

    stateful_model_path = Path("stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")
    
    convert_script = genai_llm_bench / "convert.py"
    
    if not (stateful_model_path / "openvino_model.xml").exists():
        !python $convert_script --model_id stabilityai/stable-zephyr-3b --precision FP16 --compress_weights 4BIT_DEFAULT --output stable-zephyr-3b-stateful --force_convert


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    2024-03-05 13:50:49.184866: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-05 13:50:49.186797: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:50:49.223416: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:50:49.223832: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-05 13:50:49.887707: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.1.0+cu121 with CUDA 1201 (you have 2.2.0+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    WARNING:nncf:NNCF provides best results with torch==2.2.1, while current torch version is 2.2.0+cpu. If you encounter issues, consider switching to torch==2.2.1
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32
    [ INFO ] openvino runtime version: 2024.1.0-14645-e6dc0865128
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    [ INFO ] Model conversion to FP16 will be skipped as found converted model stable-zephyr-3b-stateful/pytorch/dldt/FP16/openvino_model.xml.If it is not expected behaviour, please remove previously converted model or use --force_convert option
    [ INFO ] Compress model weights to 4BIT_DEFAULT
    [ INFO ] Compression options:
    [ INFO ] {'mode': <CompressWeightsMode.INT4_SYM: 'int4_sym'>, 'group_size': 128}
    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 9% (2 / 226)              | 0% (0 / 224)                      |
    +--------------+---------------------------+-----------------------------------+
    | 4            | 91% (224 / 226)           | 100% (224 / 224)                  |
    +--------------+---------------------------+-----------------------------------+
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━ 100% 226/226 • 0:01:29 • 0:00:00;0;104;181m0:00:01181m0:00:05
    

Select device for inference
---------------------------



.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices,
        value="CPU",
        description="Device:",
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1'), value='CPU')



Estimate model performance
--------------------------



openvino.genai / llm_bench / python / benchmark.py script allow to
estimate text generation pipeline inference on specific input prompt
with given number of maximum generated tokens.

.. code:: ipython3

    benchmark_script = genai_llm_bench / "benchmark.py"
    
    !python $benchmark_script -m $stateful_model_path -ic 512 -p "Tell me story about cats" -d $device.value


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.1.0+cu121 with CUDA 1201 (you have 2.2.0+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    2024-03-05 13:52:39.048911: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-05 13:52:39.050779: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:52:39.088178: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:52:39.088623: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-05 13:52:39.754578: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    [ INFO ] ==SUCCESS FOUND==: use_case: text_gen, model_type: stable-zephyr-3b-stateful
    [ INFO ] OV Config={'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '', 'NUM_STREAMS': '1'}
    [ INFO ] OPENVINO_TORCH_BACKEND_DEVICE=CPU
    [ INFO ] Model path=stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT, openvino runtime version: 2024.1.0-14645-e6dc0865128
    Compiling the model to CPU ...
    [ INFO ] From pretrained time: 3.21s
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    [ INFO ] Numbeams: 1, benchmarking iter nums(exclude warm-up): 0, prompt nums: 1
    [ INFO ] [warm-up] Input text: Tell me story about cats
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    [ INFO ] [warm-up] Input token size: 5, Output size: 336, Infer count: 512, Tokenization Time: 2.23ms, Detokenization Time: 0.51ms, Generation Time: 23.79s, Latency: 70.80 ms/token
    [ INFO ] [warm-up] First token latency: 837.58 ms/token, other tokens latency: 68.43 ms/token, len of tokens: 336
    [ INFO ] [warm-up] First infer latency: 836.44 ms/infer, other infers latency: 67.89 ms/infer, inference count: 336
    [ INFO ] [warm-up] Result MD5:['601aa0958ff0e0f9b844a9e6d186fbd9']
    [ INFO ] [warm-up] Generated: Tell me story about cats and dogs.
    Once upon a time, in a small village, there lived a young girl named Lily. She had two pets, a cat named Mittens and a dog named Max. Mittens was a beautiful black cat with green eyes, and Max was a big lovable golden retriever with a wagging tail.
    One sunny day, Lily decided to take her pets for a walk in the nearby forest. As they were walking, they heard a loud barking sound. Suddenly, a group of dogs appeared from the bushes, led by a big brown dog with a friendly smile.
    Lily was scared at first, but Max quickly jumped in front of her and growled at the dogs. The big brown dog introduced himself as Rocky and explained that he and his friends were just out for a walk too.
    Lily and Rocky became fast friends, and they often went on walks together. Max and Rocky got along well too, and they would play together in the forest.
    One day, while Lily was at school, Mittens and Max decided to explore the forest and stumbled upon a group of stray cats. The cats were hungry and scared, so Mittens and Max decided to help them by giving them some food.
    The cats were grateful and thanked Mittens and Max for their kindness. They even allowed Mittens to climb on their backs and enjoy the sun.
    From that day on, Mittens and Max became known as the village's cat and dog heroes. They were always there to help their furry friends in need.
    And so, Lily learned that sometimes the best friends are the ones that share the same love for pets.<|endoftext|>


Compare with model without state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    stateless_model_path = Path("stable-zephyr-3b-stateless/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")
    
    if not (stateless_model_path / "openvino_model.xml").exists():
        !python $convert_script --model_id stabilityai/stable-zephyr-3b --precision FP16 --compress_weights 4BIT_DEFAULT --output stable-zephyr-3b-stateless --force_convert --disable-stateful


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    2024-03-05 13:53:12.727472: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-05 13:53:12.729379: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:53:12.765262: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:53:12.765680: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-05 13:53:13.414451: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.1.0+cu121 with CUDA 1201 (you have 2.2.0+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    WARNING:nncf:NNCF provides best results with torch==2.2.1, while current torch version is 2.2.0+cpu. If you encounter issues, consider switching to torch==2.2.1
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32
    [ INFO ] openvino runtime version: 2024.1.0-14645-e6dc0865128
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Using the export variant default. Available variants are:
        - default: The default ONNX variant.
    Using framework PyTorch: 2.2.0+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/modeling_utils.py:4193: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:114: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/optimum/exporters/onnx/model_patcher.py:299: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:97: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:341: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:348: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/transformers/models/stablelm/modeling_stablelm.py:360: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    [ INFO ] Compress model weights to 4BIT_DEFAULT
    [ INFO ] Compression options:
    [ INFO ] {'mode': <CompressWeightsMode.INT4_SYM: 'int4_sym'>, 'group_size': 128}
    INFO:nncf:Statistics of the bitwidth distribution:
    +--------------+---------------------------+-----------------------------------+
    | Num bits (N) | % all parameters (layers) |    % ratio-defining parameters    |
    |              |                           |             (layers)              |
    +==============+===========================+===================================+
    | 8            | 9% (2 / 226)              | 0% (0 / 224)                      |
    +--------------+---------------------------+-----------------------------------+
    | 4            | 91% (224 / 226)           | 100% (224 / 224)                  |
    +--------------+---------------------------+-----------------------------------+
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━ 100% 226/226 • 0:01:29 • 0:00:00;0;104;181m0:00:01181m0:00:05
    

.. code:: ipython3

    !python $benchmark_script -m $stateless_model_path -ic 512 -p "Tell me story about cats" -d $device.value


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
        PyTorch 2.1.0+cu121 with CUDA 1201 (you have 2.2.0+cpu)
        Python  3.8.18 (you have 3.8.10)
      Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
      Memory-efficient attention, SwiGLU, sparse and more won't be available.
      Set XFORMERS_MORE_DETAILS=1 for more details
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    2024-03-05 13:55:27.540258: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-05 13:55:27.542166: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:55:27.578718: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-05 13:55:27.579116: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-05 13:55:28.229026: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cadam32bit_grad_fp32
    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    [ INFO ] ==SUCCESS FOUND==: use_case: text_gen, model_type: stable-zephyr-3b-stateless
    [ INFO ] OV Config={'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '', 'NUM_STREAMS': '1'}
    [ INFO ] OPENVINO_TORCH_BACKEND_DEVICE=CPU
    [ INFO ] Model path=stable-zephyr-3b-stateless/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT, openvino runtime version: 2024.1.0-14645-e6dc0865128
    Provided model does not contain state. It may lead to sub-optimal performance.Please reexport model with updated OpenVINO version >= 2023.3.0 calling the `from_pretrained` method with original model and `export=True` parameter
    Compiling the model to CPU ...
    [ INFO ] From pretrained time: 3.15s
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    [ INFO ] Numbeams: 1, benchmarking iter nums(exclude warm-up): 0, prompt nums: 1
    [ INFO ] [warm-up] Input text: Tell me story about cats
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    [ INFO ] [warm-up] Input token size: 5, Output size: 336, Infer count: 512, Tokenization Time: 2.02ms, Detokenization Time: 0.51ms, Generation Time: 18.59s, Latency: 55.32 ms/token
    [ INFO ] [warm-up] First token latency: 990.01 ms/token, other tokens latency: 52.47 ms/token, len of tokens: 336
    [ INFO ] [warm-up] First infer latency: 989.00 ms/infer, other infers latency: 51.98 ms/infer, inference count: 336
    [ INFO ] [warm-up] Result MD5:['601aa0958ff0e0f9b844a9e6d186fbd9']
    [ INFO ] [warm-up] Generated: Tell me story about cats and dogs.
    Once upon a time, in a small village, there lived a young girl named Lily. She had two pets, a cat named Mittens and a dog named Max. Mittens was a beautiful black cat with green eyes, and Max was a big lovable golden retriever with a wagging tail.
    One sunny day, Lily decided to take her pets for a walk in the nearby forest. As they were walking, they heard a loud barking sound. Suddenly, a group of dogs appeared from the bushes, led by a big brown dog with a friendly smile.
    Lily was scared at first, but Max quickly jumped in front of her and growled at the dogs. The big brown dog introduced himself as Rocky and explained that he and his friends were just out for a walk too.
    Lily and Rocky became fast friends, and they often went on walks together. Max and Rocky got along well too, and they would play together in the forest.
    One day, while Lily was at school, Mittens and Max decided to explore the forest and stumbled upon a group of stray cats. The cats were hungry and scared, so Mittens and Max decided to help them by giving them some food.
    The cats were grateful and thanked Mittens and Max for their kindness. They even allowed Mittens to climb on their backs and enjoy the sun.
    From that day on, Mittens and Max became known as the village's cat and dog heroes. They were always there to help their furry friends in need.
    And so, Lily learned that sometimes the best friends are the ones that share the same love for pets.<|endoftext|>


Using model with Optimum Intel
------------------------------



Running model with Optimum-Intel API required following steps: 1.
register normalized config for model 2. create instance of
``OVModelForCausalLM`` class using ``from_pretrained`` method.

The model text generation interface remains without changes, the text
generation process started with running ``ov_model.generate`` method and
passing text encoded by the tokenizer as input. This method returns a
sequence of generated token ids that should be decoded using a tokenizer

.. code:: ipython3

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoConfig
    
    ov_model = OVModelForCausalLM.from_pretrained(
        stateful_model_path,
        config=AutoConfig.from_pretrained(stateful_model_path, trust_remote_code=True),
        device=device.value,
    )

Interactive chatbot demo
------------------------



| Now, our model ready to use. Let’s see it in action. We will use
  Gradio interface for interaction with model. Put text message into
  ``Chat message box`` and click ``Submit`` button for starting
  conversation. There are several parameters that can control text
  generation quality: \* ``Temperature`` is a parameter used to control
  the level of creativity in AI-generated text. By adjusting the
  ``temperature``, you can influence the AI model’s probability
  distribution, making the text more focused or diverse.
| Consider the following example: The AI model has to complete the
  sentence “The cat is \____.” with the following token probabilities:

::

   playing: 0.5  
   sleeping: 0.25  
   eating: 0.15  
   driving: 0.05  
   flying: 0.05  

   - **Low temperature** (e.g., 0.2): The AI model becomes more focused and deterministic, choosing tokens with the highest probability, such as "playing."  
   - **Medium temperature** (e.g., 1.0): The AI model maintains a balance between creativity and focus, selecting tokens based on their probabilities without significant bias, such as "playing," "sleeping," or "eating."  
   - **High temperature** (e.g., 2.0): The AI model becomes more adventurous, increasing the chances of selecting less likely tokens, such as "driving" and "flying."

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is an another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.
-  ``Repetition Penalty`` This parameter can help penalize tokens based
   on how frequently they occur in the text, including the input prompt.
   A token that has already appeared five times is penalized more
   heavily than a token that has appeared only one time. A value of 1
   means that there is no penalty and values larger than 1 discourage
   repeated tokens.

You can modify them in ``Advanced generation options`` section.

.. code:: ipython3

    import torch
    from threading import Event, Thread
    from uuid import uuid4
    from typing import List, Tuple
    import gradio as gr
    from transformers import (
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
    
    model_name = "stable-zephyr-3b"
    
    tok = AutoTokenizer.from_pretrained(stateful_model_path)
    
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """
    
    model_configuration = {
        "start_message": f"<|system|>\n {DEFAULT_SYSTEM_PROMPT }<|endoftext|>",
        "history_template": "<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}<|endoftext|>",
        "current_message_template": "<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}",
    }
    history_template = model_configuration["history_template"]
    current_message_template = model_configuration["current_message_template"]
    start_message = model_configuration["start_message"]
    stop_tokens = model_configuration.get("stop_tokens")
    tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
    
    examples = [
        ["Hello there! How are you doing?"],
        ["What is OpenVINO?"],
        ["Who are you?"],
        ["Can you explain to me briefly what is Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["What are some common mistakes to avoid when writing code?"],
        ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"],
    ]
    
    max_new_tokens = 256
    
    
    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids
    
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    
    if stop_tokens is not None:
        if isinstance(stop_tokens[0], str):
            stop_tokens = tok.convert_tokens_to_ids(stop_tokens)
    
        stop_tokens = [StopOnTokens(stop_tokens)]
    
    
    def default_partial_text_processor(partial_text: str, new_text: str):
        """
        helper for updating partially generated answer, used by de
    
        Params:
          partial_text: text buffer for storing previosly generated text
          new_text: text update for the current step
        Returns:
          updated text string
    
        """
        partial_text += new_text
        return partial_text
    
    
    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)
    
    
    def convert_history_to_text(history: List[Tuple[str, str]]):
        """
        function for conversion history stored as list pairs of user and assistant messages to string according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in text format
        """
        text = start_message + "".join(["".join([history_template.format(num=round, user=item[0], assistant=item[1])]) for round, item in enumerate(history[:-1])])
        text += "".join(
            [
                "".join(
                    [
                        current_message_template.format(
                            num=len(history) + 1,
                            user=history[-1][0],
                            assistant=history[-1][1],
                        )
                    ]
                )
            ]
        )
        return text
    
    
    def user(message, history):
        """
        callback function for updating user messages in interface on submit button click
    
        Params:
          message: current message
          history: conversation history
        Returns:
          None
        """
        # Append the user's message to the conversation history
        return "", history + [[message, ""]]
    
    
    def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        """
        callback function for running chatbot on submit button click
    
        Params:
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text.
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          conversation_id: unique conversation identifier.
    
        """
    
        # Construct the input message string for the model by concatenating the current system message and conversation history
        messages = convert_history_to_text(history)
    
        # Tokenize the messages string
        input_ids = tok(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
        if input_ids.shape[1] > 2000:
            history = [history[-1]]
            messages = convert_history_to_text(history)
            input_ids = tok(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
        streamer = TextIteratorStreamer(tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        if stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)
    
        stream_complete = Event()
    
        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            global start_time
            ov_model.generate(**generate_kwargs)
            stream_complete.set()
    
        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
    
        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield history
    
    
    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())
    
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(f"""<h1><center>OpenVINO {model_name} Chatbot</center></h1>""")
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
    
        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.queue(max_size=2)
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch(share=True)
