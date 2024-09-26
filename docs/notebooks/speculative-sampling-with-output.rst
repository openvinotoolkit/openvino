Text Generation via Speculative Sampling, KV Caching, and OpenVINO™
===================================================================

As model sizes grow, Generative AI implementations require significant
inference resources. This not only increases the cost per generation
from a prompt, but also increases the power consumption used to serve
such requests.

Inference optimizations for text generation are essential for reducing
costs and power consumption. When optimizing the inference process, the
amount of time and energy required to generate text can be significantly
reduced. This can lead to cost savings in terms of hardware and
software, as well as reduced power consumption. Additionally, inference
optimizations can help improve the accuracy of text generation as well
as the speed at which it can be generated. This can lead to an improved
user experience and increased efficiency in text-generation tasks. In
summary, inference optimizations for text generation are essential to
reduce costs and power consumption, while also improving the accuracy
and speed of text generation.

Another necessary condition is that the optimizations are compatible
with each other. That is, implementing a certain optimization should not
preclude other optimizations. There are several levels of optimizations
that can provide significant speedup without “bumping into each other”
in a way that will compromise overall efficiency.

For details on this method, please refer to the paper by Chen et al,
http://arxiv.org/abs/2302.01318. Additionally, there’s an interesting
proof of correctness of speculative sampling (showing that the original
distribution is preserved) by Leviathan et al,
http://arxiv.org/abs/2211.17192

Our blog article describing this implementation with OpenVino is
available at openvino.ai


**Table of contents:**


-  `Prerequisites <#prerequisites>`__

   -  `Select inference device <#select-inference-device>`__

-  `Create autoregressive and speculative forms of sampling with KV
   Cache
   support <#create-autoregressive-and-speculative-forms-of-sampling-with-kv-cache-support>`__

   -  `Setup imports <#setup-imports>`__
   -  `Prepare autoregressive
      sampling <#prepare-autoregressive-sampling>`__
   -  `Prepare speculative sampling <#prepare-speculative-sampling>`__

-  `Main generation function <#main-generation-function>`__

   -  `Download and Convert Model <#download-and-convert-model>`__ ###
      Installation Instructions

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



First, we should install the `Hugging Face
Optimum <https://huggingface.co/docs/optimum/installation>`__ library
accelerated by OpenVINO integration. The Hugging Face Optimum Intel API
is a high-level API that enables us to convert and quantize models from
the Hugging Face Transformers library to the OpenVINO™ IR format. For
more details, refer to the `Hugging Face Optimum Intel
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

We will also need to install transformers (HuggingFace) and some other
useful modules.

.. code:: ipython3

    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install --pre -Uq "openvino>=2024.2.0" openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q --upgrade transformers "torch>=2.1" "torchvision" "gradio>=4.19" accelerate "onnx<1.16.2" ipywidgets --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select the device from dropdown list for running inference using
OpenVINO.

.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU.0', 'GPU.1', 'AUTO'), value='CPU')



Create autoregressive and speculative forms of sampling with KV Cache support
-----------------------------------------------------------------------------



Text generation is often done in an autoregressive fashion. We will all
support a KV cache (aka Past Value Cache) in the code. Note that we are
using greedy sampling. We do not adjust other text generation parameters
(e.g. temperature) so keep this illustration of speculative sampling as
simple and understandable as possible.

Setup imports
~~~~~~~~~~~~~



.. code:: ipython3

    import time
    import numpy as np
    import openvino as ov

Prepare autoregressive sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def autoregressive_sampling_with_pkv(input, model, N=30):
        input_ids, attention_mask = input.input_ids, input.attention_mask
        seq_len = input_ids.shape[-1]
        position_ids = np.arange(0, seq_len, dtype=np.int64).reshape([-1, seq_len])
    
        # in all subsequent inferences we feed tokens one by one,
        # but for the first one we feed the whole encoded prompt
        request = model.create_infer_request()
        request.infer((input_ids, attention_mask, position_ids, np.array([0])))
        next_token = np.argmax(request.results["logits"][:, -1]).reshape([1])
    
        all_tokens = []
        all_tokens.extend(input_ids[0])
        all_tokens.append(next_token[0])
    
        while seq_len < N:
            input_ids = next_token.reshape([1, 1])
            attention_mask = np.concatenate((attention_mask, np.array([1]).reshape([1, 1])), axis=1)
            position_ids = np.array([attention_mask.shape[1]]).reshape([1, 1])
    
            request.infer((input_ids, attention_mask, position_ids, np.array([0])))
            next_token = np.argmax(request.results["logits"][:, -1])
            all_tokens.append(next_token)
            seq_len += 1
    
        return all_tokens

Prepare speculative sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



-  Step 1: With speculative sampling, we first generate K samples from
   the draft model (in an autoregressive manner).
-  Step 2: These are now candidates to examine using the main model
   (step 2) using a batch size of K.
-  Step 3: We go through each K predicted tokens, and if tokens differ,
   we stop and keep the last token predicted by the main model.
-  Step 4: We update KV-cache dropping keys & values for differing
   tokens and repeat Step 1.

.. code:: ipython3

    def update_state(request, seq_len):
        for state in request.query_state():
            old_seq_len = state.state.shape[2]
            if seq_len >= old_seq_len:
                continue
            # After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
            # Increment the sequence length by the number of matched tokens, and
            # trim the KV cache to match the new sequence length.
            state.state = ov.Tensor(state.state.data[:, :, :seq_len])
    
    
    def speculative_sampling_with_pkv(input, draft_model, main_model, K, N=30, **kwargs):
        input_ids, attention_mask = input.input_ids, input.attention_mask
        # seq_len number of key/values or number of already processed input tokens
        seq_len = input_ids.shape[-1]
        position_ids = np.arange(0, seq_len, dtype=np.int64).reshape([-1, seq_len])
    
        draft_request = draft_model.create_infer_request()
        draft_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
    
        main_request = main_model.create_infer_request()
        main_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
        first_token = np.argmax(main_request.results["logits"][:, -1]).reshape([1])
    
        all_tokens = []
        all_tokens.extend(input_ids[0])
        all_tokens.append(first_token[0])
    
        accum_draft_tokens = []
        while seq_len < N:
            next_token = first_token
            for i in range(K):
                input_ids = next_token.reshape([1, 1])
                attention_mask = np.concatenate((attention_mask, np.array([1]).reshape([1, 1])), axis=1)
                position_ids = np.array([attention_mask.shape[1]]).reshape([1, 1])
    
                draft_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
                next_token = np.argmax(draft_request.results["logits"][:, -1])
                accum_draft_tokens.append(next_token)
    
            # main model will give also K out tokens
            # feed the same first token to the main model and do not give the last token generated by the draft
            input_ids = np.concatenate((first_token.reshape([1]), accum_draft_tokens[:-1])).reshape([1, -1])
            attention_mask = np.ones((1, seq_len + K))
            position_ids = np.arange(seq_len, seq_len + K, dtype=np.int64).reshape([1, -1])
    
            main_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
            next_tokens = np.argmax(main_request.results["logits"], axis=-1)[0]
    
            # if disagrees from the very beggining then context will be expanded only for one element
            # all elements match then context will be expanded to K elements
            for disagree_idx, (t1, t2) in enumerate(zip(accum_draft_tokens, next_tokens)):
                if t1 != t2:
                    break
    
            first_token = next_tokens[disagree_idx]
            all_tokens.extend(next_tokens[: disagree_idx + 1])
            seq_len += disagree_idx + 1
    
            # cut key/values depending on the position where disagreement starts
            update_state(draft_request, seq_len)
            update_state(main_request, seq_len)
    
            attention_mask = np.ones((1, seq_len))
            accum_draft_tokens = []
        all_tokens.extend(accum_draft_tokens)
        return all_tokens

Main generation function
------------------------



Download and Convert Model
~~~~~~~~~~~~~~~~~~~~~~~~~~



Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. For speculative decoding we need to manually update states,
therefore we will use directly openvino inference api, and optimum only
for model conversion. >To download Llama-2-7b-chat-hf, you will need to
accept license agreement. You must be a registered user in Hugging
Face Hub. Please visit HuggingFace model
`card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
carefully read terms of usage and click accept button. You will need to
use an access token for the code below to run. For more information on
access tokens, refer to this section of the documentation.

.. code:: ipython3

    from pathlib import Path
    
    main_model_id = "meta-llama/Llama-2-7b-chat-hf"
    main_model_path = Path("Llama-2-7b-chat-hf")
    draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    draft_model_path = Path("TinyLlama-1.1B-Chat-v1.0")
    
    from transformers import AutoTokenizer
    
    main_tokenizer = AutoTokenizer.from_pretrained(main_model_id)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)

.. code:: ipython3

    # In order for speculative sampling to work, both main and draft tokenizers should be the same.
    token_test_txt = "text to ensure tokenizers work the same, as of 2024"
    tokens_1 = draft_tokenizer(token_test_txt, return_tensors="pt").input_ids
    tokens_2 = main_tokenizer(token_test_txt, return_tensors="pt").input_ids
    
    assert all((tokens_1 - tokens_2)[0] == 0)

.. code:: ipython3

    if not main_model_path.exists():
        !optimum-cli export openvino --model $main_model_id --weight-format fp16 {main_model_path}
    if not draft_model_path.exists():
        !optimum-cli export openvino --model $draft_model_id --weight-format fp16 {draft_model_path}

Infer directly using OpenVINO Inference Pipeline

.. code:: ipython3

    core = ov.Core()
    draft_ov_model = core.read_model(draft_model_path / "openvino_model.xml")
    draft_model = core.compile_model(draft_ov_model, device_name=device.value)
    
    main_ov_model = core.read_model(main_model_path / "openvino_model.xml")
    main_model = core.compile_model(main_ov_model, device_name=device.value)

.. code:: ipython3

    def main(
        prompt: str,
        n_tokens_to_generate: int = 75,
        K: int = 5,
        seed: int = 5555,
    ):
        # seed numpy rng
        np.random.seed(seed)
        tokenized = main_tokenizer(prompt, return_tensors="pt")
    
        def run_autoregressive_sampling_fn(decode_fn, tokenized, **kwargs):
            start = time.perf_counter()
            output_ids = decode_fn(tokenized, **kwargs)
            text = main_tokenizer.decode(output_ids, skip_special_tokens=True)
            elapsed_time = time.perf_counter() - start
            return text, elapsed_time
    
        def run_speculative_sampling_fn(decode_fn, input_ids, **kwargs):
            start = time.perf_counter()
            output_ids = decode_fn(input_ids, **kwargs)
            text = main_tokenizer.decode(output_ids, skip_special_tokens=True)
            elapsed_time = time.perf_counter() - start
            return text, elapsed_time
    
        autoregressive_text, autoregressive_time = run_autoregressive_sampling_fn(
            autoregressive_sampling_with_pkv,
            tokenized,
            model=main_model,
            N=n_tokens_to_generate,
        )
    
        speculative_text, speculative_time = run_speculative_sampling_fn(
            speculative_sampling_with_pkv,
            tokenized,
            main_model=main_model,
            draft_model=draft_model,
            N=n_tokens_to_generate,
            K=K,
        )
    
        # Format results for output in gradio
        out = "\n" + "Autoregressive Decode" + "\n" + "---------------------" + "\n"
        out = out + f"Time = {autoregressive_time:.2f}s" + "\n" + f"Text = {autoregressive_text}" + "\n"
        out = out + "\n" + "Speculative Decode" + "\n" + "------------------" + "\n"
        out = out + f"Time = {speculative_time:.2f}s" + "\n" + f"Text = {speculative_text}"
        return out

.. code:: ipython3

    res = main("Alan Turing was a", n_tokens_to_generate=100)
    print(res)


.. parsed-literal::

    2024-04-17 10:21:41.642283: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-17 10:21:41.644834: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-17 10:21:41.677055: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-04-17 10:21:41.677093: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-04-17 10:21:41.677119: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-04-17 10:21:41.683198: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-04-17 10:21:41.683977: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-04-17 10:21:42.477656: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    
    Autoregressive Decode
    ---------------------
    Time = 44.39s
    Text = Alan Turing was a British mathematician, computer scientist, and codebreaker who played a pivotal role in cracking the German Enigma code during World War II. He was also a pioneer in the field of artificial intelligence and made significant contributions to the development of computer science.
    
    Turing was born on June 23, 1912, in London, England. He was educated at Cambridge University, where he earned a degree in mathematics in 
    
    Speculative Decode
    ------------------
    Time = 22.96s
    Text = Alan Turing was a British mathematician, computer scientist, and codebreaker who played a pivotal role in cracking the German Enigma code during World War II. He was also a pioneer in the field of artificial intelligence and made significant contributions to the development of computer science.
    
    Turing was born on June 23, 1912, in London, England. He was educated at Cambridge University, where he earned a degree in mathematics in 1


.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/speculative-sampling/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=main)
    
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
