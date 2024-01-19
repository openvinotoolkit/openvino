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

    %pip uninstall -q -y optimum-intel optimum
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu -r ./openvino.genai/llm_bench/python/requirements.txt
    %pip uninstall -q -y openvino openvino-dev openvino-nightly
    %pip install -q openvino-nightly
    %pip install -q gradio

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

.. code:: ipython3

    model_path = Path("stable-zephyr-3b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")

    convert_script = genai_llm_bench / "convert.py"

    !python $convert_script --model_id stabilityai/stable-zephyr-3b --precision FP16 --compress_weights 4BIT_DEFAULT --output stable-zephyr-3b --force_convert


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    [ INFO ] openvino runtime version: 2024.0.0-13826-b51c5c0a997
    model.safetensors: 100%|███████████████████| 5.59G/5.59G [04:19<00:00, 21.6MB/s]
    generation_config.json: 100%|██████████████████| 111/111 [00:00<00:00, 13.6kB/s]
    tokenizer_config.json: 100%|████████████████| 5.21k/5.21k [00:00<00:00, 839kB/s]
    tokenizer.json: 100%|██████████████████████| 2.11M/2.11M [00:01<00:00, 2.10MB/s]
    special_tokens_map.json: 100%|██████████████████| 587/587 [00:00<00:00, 429kB/s]
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Using the export variant default. Available variants are:
        - default: The default ONNX variant.
    Using framework PyTorch: 2.1.2+cu121
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/.cache/huggingface/modules/transformers_modules/stabilityai/stable-zephyr-3b/9974c58a0ec4be4cd6f55e814a2a93b9cf163823/modeling_stablelm_epoch.py:106: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    /home/ea/.cache/huggingface/modules/transformers_modules/stabilityai/stable-zephyr-3b/9974c58a0ec4be4cd6f55e814a2a93b9cf163823/modeling_stablelm_epoch.py:236: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    /home/ea/.cache/huggingface/modules/transformers_modules/stabilityai/stable-zephyr-3b/9974c58a0ec4be4cd6f55e814a2a93b9cf163823/modeling_stablelm_epoch.py:243: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    /home/ea/.cache/huggingface/modules/transformers_modules/stabilityai/stable-zephyr-3b/9974c58a0ec4be4cd6f55e814a2a93b9cf163823/modeling_stablelm_epoch.py:253: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
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
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━ 100% 226/226 • 0:02:36 • 0:00:00;0;104;181m0:00:01181m0:00:08


Estimate model performance
--------------------------

openvino.genai / llm_bench / python / benchmark.py script allow to
estimate text generation pipeline inference on specific input prompt
with given number of maximum generated tokens.

.. code:: ipython3

    benchmark_script = genai_llm_bench / "benchmark.py"

    !python $benchmark_script -m $model_path -ic 512 -p "Tell me story about cats"


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    [ INFO ] ==SUCCESS FOUND==: use_case: text_gen, model_type: stable-zephyr-3b
    [ INFO ] ov_config={'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '', 'NUM_STREAMS': '1'}
    OPENVINO_TORCH_BACKEND_DEVICE=CPU
    [ INFO ] model_path=stable-zephyr-3b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT, openvino runtime version: 2024.0.0-13826-b51c5c0a997
    Compiling the model to CPU ...
    [ INFO ] From pretrained time: 5.89s
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    [ INFO ] num_iters=0, num_text_list=1
    [ INFO ] input_text=Tell me story about cats
    [ INFO ] Input token size:5, max_output_token_size:512
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    [ INFO ] [warm-up] Input token size: 5
    [ INFO ] [warm-up] Output size: 290
    [ INFO ] [warm-up] Infer count: 512
    [ INFO ] [warm-up] Tokenization Time: 2.29ms
    [ INFO ] [warm-up] Detokenization Time: 0.50ms
    [ INFO ] [warm-up] Generation Time: 19.75s
    [ INFO ] [warm-up] Latency: 68.09 ms/token
    [ INFO ] [warm-up] Generated:
    Tell me story about cats and dogs.
    Once upon a time, in a small village, there lived a young girl named Lily. She had two pets, a cat named Mittens and a dog named Max. Mittens was very playful and loved to chase Max around the house. Max, on the other hand, was a bit timid and would often hide when Mittens came around.
    One day, Mittens and Max were playing together in the backyard when a loud thunderstorm came suddenly. Mittens, being afraid of the thunder, ran inside the house, leaving Max behind. The rain was coming down hard, and Max was struggling to find his way back inside.
    Lily, who was watching the storm from her bedroom, heard Max's cries and knew she had to help him. She ran down the stairs and found Max standing in the rain, looking lost. Lily knew just what to do. She picked up Max and carried him back to the house, where Mittens was waiting.
    Mittens was relieved to see Max safe and sound, and the two of them snuggled up together on the couch for the rest of the storm. From that day on, Max was no longer afraid of Mittens, and the three of them became closer than ever before.
    And that, my dear friends, is the story of Mittens, Max, and Lily, and how they overcame their fears and became a true family.<|endoftext|>
    [ INFO ] [warm-up] Result MD5:['f5575487f181d7de8e4c095b39fa4180']
    [ INFO ] [warm-up] First token latency: 1030.65 ms/token, other tokens latency: 64.68 ms/token, len of tokens: 290
    [ INFO ] [warm-up] First token infer latency: 1021.36 ms/token, other tokens infer latency: 64.10 ms/token, len of tokens: 290


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

You can estimate the model performance by adding stateful transformation
using ``--stateful`` flag on conversion step

.. code:: ipython3

    stateful_model_path = Path("stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT")

    !python $convert_script --model_id stabilityai/stable-zephyr-3b --precision FP16 --compress_weights 4BIT_DEFAULT --output stable-zephyr-3b-stateful --force_convert --stateful


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    [ INFO ] openvino runtime version: 2024.0.0-13826-b51c5c0a997
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Using the export variant default. Available variants are:
        - default: The default ONNX variant.
    Using framework PyTorch: 2.1.2+cu121
    The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.
    Overriding 1 configuration item(s)
    	- use_cache -> True
    /home/ea/work/openvino_notebooks/notebooks/273-stable-zephyr-3b-chatbot/openvino.genai/llm_bench/python/utils/conversion_utils/better_transformer_patch.py:289: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size(0) > 1:
    /home/ea/work/openvino_notebooks/notebooks/273-stable-zephyr-3b-chatbot/openvino.genai/llm_bench/python/utils/conversion_utils/better_transformer_patch.py:290: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1:
    /home/ea/.cache/huggingface/modules/transformers_modules/stabilityai/stable-zephyr-3b/9974c58a0ec4be4cd6f55e814a2a93b9cf163823/modeling_stablelm_epoch.py:106: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    /home/ea/work/openvino_notebooks/notebooks/273-stable-zephyr-3b-chatbot/openvino.genai/llm_bench/python/utils/conversion_utils/better_transformer_patch.py:380: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
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
    [2KApplying Weight Compression ━━━━━━━━━━━━━━━━━━━ 100% 226/226 • 0:02:35 • 0:00:00;0;104;181m0:00:01181m0:00:08


.. code:: ipython3

    !python $benchmark_script -m $stateful_model_path -ic 512 -p "Tell me story about cats"


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    [ INFO ] ==SUCCESS FOUND==: use_case: text_gen, model_type: stable-zephyr-3b-stateful
    [ INFO ] ov_config={'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '', 'NUM_STREAMS': '1'}
    OPENVINO_TORCH_BACKEND_DEVICE=CPU
    [ INFO ] model_path=stable-zephyr-3b-stateful/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT, openvino runtime version: 2024.0.0-13826-b51c5c0a997
    Compiling the model to CPU ...
    [ INFO ] From pretrained time: 5.70s
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    [ INFO ] num_iters=0, num_text_list=1
    [ INFO ] input_text=Tell me story about cats
    [ INFO ] Input token size:5, max_output_token_size:512
    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    [ INFO ] [warm-up] Input token size: 5
    [ INFO ] [warm-up] Output size: 290
    [ INFO ] [warm-up] Infer count: 512
    [ INFO ] [warm-up] Tokenization Time: 1.99ms
    [ INFO ] [warm-up] Detokenization Time: 0.46ms
    [ INFO ] [warm-up] Generation Time: 16.35s
    [ INFO ] [warm-up] Latency: 56.37 ms/token
    [ INFO ] [warm-up] Generated:
    Tell me story about cats and dogs.
    Once upon a time, in a small village, there lived a young girl named Lily. She had two pets, a cat named Mittens and a dog named Max. Mittens was very playful and loved to chase Max around the house. Max, on the other hand, was a bit timid and would often hide when Mittens came around.
    One day, Mittens and Max were playing together in the backyard when a loud thunderstorm came suddenly. Mittens, being afraid of the thunder, ran inside the house, leaving Max behind. The rain was coming down hard, and Max was struggling to find his way back inside.
    Lily, who was watching the storm from her bedroom, heard Max's cries and knew she had to help him. She ran down the stairs and found Max standing in the rain, looking lost. Lily knew just what to do. She picked up Max and carried him back to the house, where Mittens was waiting.
    Mittens was relieved to see Max safe and sound, and the two of them snuggled up together on the couch for the rest of the storm. From that day on, Max was no longer afraid of Mittens, and the three of them became closer than ever before.
    And that, my dear friends, is the story of Mittens, Max, and Lily, and how they overcame their fears and became a true family.<|endoftext|>
    [ INFO ] [warm-up] Result MD5:['f5575487f181d7de8e4c095b39fa4180']
    [ INFO ] [warm-up] First token latency: 1074.80 ms/token, other tokens latency: 52.77 ms/token, len of tokens: 290
    [ INFO ] [warm-up] First token infer latency: 1073.78 ms/token, other tokens infer latency: 52.15 ms/token, len of tokens: 290


Using model with Optimum Intel
------------------------------

Running model with Optimum-Intel API required following steps: 1.
register normalized config for model 2. create instance of
``OVModelForCausalLM`` class using ``from_pretrained`` method and
providing path to the model and ``stateful`` flag

The model text generation interface remains without changes, the text
generation process started with running ``ov_model.generate`` method and
passing text encoded by the tokenizer as input. This method returns a
sequence of generated token ids that should be decoded using a tokenizer

.. code:: ipython3

    from utils.ov_model_classes import register_normalized_configs
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoConfig

    # Load model into Optimum Interface
    register_normalized_configs()

    ov_model = OVModelForCausalLM.from_pretrained(model_path, compile=False, config=AutoConfig.from_pretrained(stateful_model_path, trust_remote_code=True), stateful=True)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


.. parsed-literal::

    /home/ea/work/genai_env/lib/python3.8/site-packages/torch/cuda/__init__.py:138: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0
    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Interactive chatbot demo
------------------------

Now, our model ready to use. Let’s see it in action. We will use
Gradio interface for interaction with model. Put text message into
``Chat message box`` and click ``Submit`` button for starting
conversation. There are several parameters that can control text
generation quality: \* ``Temperature`` is a parameter used to control
the level of creativity in AI-generated text. By adjusting the
``temperature``, you can influence the AI model’s probability
distribution, making the text more focused or diverse.
Consider the following example: The AI model has to complete the
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

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """

    model_configuration = {
        "start_message": f"<|system|>\n {DEFAULT_SYSTEM_PROMPT }<|endoftext|>",
        "history_template": "<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}<|endoftext|>",
        "current_message_template": '<|user|>\n{user}<|endoftext|><|assistant|>\n{assistant}',
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
        [
            "Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"
        ],
    ]

    max_new_tokens = 256


    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
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


    text_processor = model_configuration.get(
        "partial_text_processor", default_partial_text_processor
    )

    def convert_history_to_text(history: List[Tuple[str, str]]):
        """
        function for conversion history stored as list pairs of user and assistant messages to string according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in text format
        """
        text = start_message + "".join(
            [
                "".join(
                    [history_template.format(num=round, user=item[0], assistant=item[1])]
                )
                for round, item in enumerate(history[:-1])
            ]
        )
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
        streamer = TextIteratorStreamer(
            tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True
        )
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
        gr.Examples(
            examples, inputs=msg, label="Click on any example and press the 'Submit' button"
        )

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
