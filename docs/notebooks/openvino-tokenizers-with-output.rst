OpenVINO Tokenizers: Incorporate Text Processing Into OpenVINO Pipelines
========================================================================

.. raw:: html

   <center>

.. raw:: html

   </center>

OpenVINO Tokenizers is an OpenVINO extension and a Python library
designed to streamline tokenizer conversion for seamless integration
into your projects. It supports Python and C++ environments and is
compatible with all major platforms: Linux, Windows, and MacOS.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Tokenization Basics <#Tokenization-Basics>`__
-  `Acquiring OpenVINO Tokenizers <#Acquiring-OpenVINO-Tokenizers>`__

   -  `Convert Tokenizer from HuggingFace Hub with CLI
      Tool <#Convert-Tokenizer-from_HuggingFace-Hub-with-CLI-Tool>`__
   -  `Convert Tokenizer from HuggingFace Hub with Python
      API <#Convert-Tokenizer-from-HuggingFace-Hub-with-Python-API>`__

-  `Text Generation Pipeline with OpenVINO
   Tokenizers <#Text-Generation-Pipeline-with-OpenVINO-Tokenizers>`__
-  `Merge Tokenizer into a Model <#Merge-Tokenizer-into-a-Model>`__
-  `Conclusion <#Conclusion>`__
-  `Links <#Links>`__

Tokenization Basics
-------------------

`back to top ⬆️ <#Table-of-contents:>`__

One does not simply put text into a neural network, only numbers. The
process of transforming text into a sequence of numbers is called
**tokenization**. It usually contains several steps that transform the
original string, splitting it into parts - tokens - with an associated
number in a dictionary. You can check the `interactive GPT-4
tokenizer <https://platform.openai.com/tokenizer>`__ to gain an
intuitive understanding of the principles of tokenizer work.

.. raw:: html

   <center>

.. raw:: html

   </center>

There are two important points in the tokenizer-model relation: 1. Every
neural network with text input is paired with a tokenizer and *cannot be
used without it*. 2. To reproduce the model’s accuracy on a specific
task, it is essential to *utilize the same tokenizer employed during the
model training*.

That is why almost all model repositories on `HuggingFace
Hub <https://HuggingFace.co/models>`__ also contain tokenizer files
(``tokenizer.json``, ``vocab.txt``, ``merges.txt``, etc.).

The process of transforming a sequence of numbers into a string is
called **detokenization**. Detokenizer can share the token dictionary
with a tokenizer, like any LLM chat model, or operate with an entirely
distinct dictionary. For instance, translation models dealing with
different source and target languages often necessitate separate
dictionaries.

.. raw:: html

   <center>

.. raw:: html

   </center>

Some tasks only need a tokenizer, like text classification, named entity
recognition, question answering, and feature extraction. On the other
hand, for tasks such as text generation, chat, translation, and
abstractive summarization, both a tokenizer and a detokenizer are
required.

Acquiring OpenVINO Tokenizers
-----------------------------

`back to top ⬆️ <#Table-of-contents:>`__

OpenVINO Tokenizers Python library allows you to convert HuggingFace
tokenizers into OpenVINO models. To install all required dependencies
use ``pip install openvino-tokenizers[transformers]``.

.. code:: ipython3

    %pip install -Uq pip
    %pip install --pre -Uq openvino-tokenizers[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    openvino-dev 2024.2.0 requires openvino==2024.2.0, but you have openvino 2024.3.0.dev20240627 which is incompatible.
    openvino-genai 2024.2.0.0 requires openvino_tokenizers~=2024.2.0.0, but you have openvino-tokenizers 2024.3.0.0.dev20240627 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu
    Requirement already satisfied: torch>=2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2.3.1+cpu)
    Requirement already satisfied: filelock in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (3.15.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (4.12.2)
    Requirement already satisfied: sympy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (1.12.1)
    Requirement already satisfied: networkx in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (3.1)
    Requirement already satisfied: jinja2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (3.1.4)
    Requirement already satisfied: fsspec in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch>=2.1) (2024.5.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jinja2->torch>=2.1) (2.1.5)
    Requirement already satisfied: mpmath<1.4.0,>=1.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from sympy->torch>=2.1) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    
    
    tokenizer_dir = Path("tokenizer/")
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

Convert Tokenizer from HuggingFace Hub with CLI Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The first way is to use the CLI utility, bundled with OpenVINO
Tokenizers. Use ``--with-detokenizer`` flag to add the detokenizer model
to the output. By setting ``--clean-up-tokenization-spaces=False`` we
ensure that the detokenizer correctly decodes a code-generation model
output. ``--trust-remote-code`` flag works the same way as passing
``trust_remote_code=True`` to ``AutoTokenizer.from_pretrained``
constructor.

.. code:: ipython3

    !convert_tokenizer $model_id --with-detokenizer -o $tokenizer_dir


.. parsed-literal::

    Loading Huggingface Tokenizer...
    Converting Huggingface Tokenizer to OpenVINO...
    Saved OpenVINO Tokenizer: tokenizer/openvino_tokenizer.xml, tokenizer/openvino_tokenizer.bin
    Saved OpenVINO Detokenizer: tokenizer/openvino_detokenizer.xml, tokenizer/openvino_detokenizer.bin


   ⚠️ If you have any problems with the command above on MacOS, try to
   `install tbb <https://formulae.brew.sh/formula/tbb#default>`__.

The result is two OpenVINO models: ``openvino_tokenizer`` and
``openvino_detokenizer``. Both can be interacted with using
``read_model``, ``compile_model`` and ``save_model``, similar to any
other OpenVINO model.

Convert Tokenizer from HuggingFace Hub with Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

The other method is to pass HuggingFace ``hf_tokenizer`` object to
``convert_tokenizer`` function:

.. code:: ipython3

    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer
    
    
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov_tokenizer, ov_detokenizer




.. parsed-literal::

    (<Model: 'tokenizer'
     inputs[
     <ConstOutput: names[string_input] shape[?] type: string>
     ]
     outputs[
     <ConstOutput: names[input_ids] shape[?,?] type: i64>,
     <ConstOutput: names[attention_mask] shape[?,?] type: i64>
     ]>,
     <Model: 'detokenizer'
     inputs[
     <ConstOutput: names[Parameter_22] shape[?,?] type: i64>
     ]
     outputs[
     <ConstOutput: names[string_output] shape[?] type: string>
     ]>)



That way you get OpenVINO model objects. Use ``save_model`` function
from OpenVINO to reuse converted tokenizers later:

.. code:: ipython3

    from openvino import save_model
    
    
    save_model(ov_tokenizer, tokenizer_dir / "openvino_tokenizer.xml")
    save_model(ov_detokenizer, tokenizer_dir / "openvino_detokenizer.xml")

To use the tokenizer, compile the converted model and input a list of
strings. It’s essential to be aware that not all original tokenizers
support multiple strings (also called batches) as input. This limitation
arises from the requirement for all resulting number sequences to
maintain the same length. To address this, a padding token must be
specified, which will be appended to shorter tokenized strings. In cases
where no padding token is determined in the original tokenizer, OpenVINO
Tokenizers defaults to using :math:`0` for padding. Presently, *only
right-side padding is supported*, typically used for classification
tasks, but not suitable for text generation.

.. code:: ipython3

    from openvino import compile_model
    
    
    tokenizer, detokenizer = compile_model(ov_tokenizer), compile_model(ov_detokenizer)
    test_strings = ["Test", "strings"]
    
    token_ids = tokenizer(test_strings)["input_ids"]
    print(f"Token ids: {token_ids}")
    
    detokenized_text = detokenizer(token_ids)["string_output"]
    print(f"Detokenized text: {detokenized_text}")


.. parsed-literal::

    Token ids: [[   1 4321]
     [   1 6031]]
    Detokenized text: ['Test' 'strings']


We can compare the result of converted (de)tokenizer with the original
one:

.. code:: ipython3

    hf_token_ids = hf_tokenizer(test_strings).input_ids
    print(f"Token ids: {hf_token_ids}")
    
    hf_detokenized_text = hf_tokenizer.batch_decode(hf_token_ids)
    print(f"Detokenized text: {hf_detokenized_text}")


.. parsed-literal::

    Token ids: [[1, 4321], [1, 6031]]


.. parsed-literal::

    2024-07-02 01:03:15.895945: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-07-02 01:03:15.931306: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-07-02 01:03:16.500703: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Detokenized text: ['<s> Test', '<s> strings']


Text Generation Pipeline with OpenVINO Tokenizers
-------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Let’s build a text generation pipeline with OpenVINO Tokenizers and
minimal dependencies. To obtain an OpenVINO model we will use the
Optimum library. The latest version allows you to get a so-called
`stateful
model <https://docs.openvino.ai/2024/openvino-workflow/running-inference/stateful-models.html>`__.

The original ``TinyLlama-1.1B-intermediate-step-1431k-3T`` model is
4.4Gb. To reduce network and disk usage we will load a converted model
which has also been compressed to ``int8``. The original conversion
command is commented.

.. code:: ipython3

    model_dir = Path(Path(model_id).name)
    
    if not model_dir.exists():
        # converting the original model
        # %pip install -U "git+https://github.com/huggingface/optimum-intel.git" "nncf>=2.8.0" onnx
        # %optimum-cli export openvino -m $model_id --task text-generation-with-past $model_dir
    
        # load already converted model
        from huggingface_hub import hf_hub_download
    
        hf_hub_download(
            "chgk13/TinyLlama-1.1B-intermediate-step-1431k-3T",
            filename="openvino_model.xml",
            local_dir=model_dir,
        )
        hf_hub_download(
            "chgk13/TinyLlama-1.1B-intermediate-step-1431k-3T",
            filename="openvino_model.bin",
            local_dir=model_dir,
        )



.. parsed-literal::

    openvino_model.xml:   0%|          | 0.00/2.93M [00:00<?, ?B/s]



.. parsed-literal::

    openvino_model.bin:   0%|          | 0.00/1.10G [00:00<?, ?B/s]


.. code:: ipython3

    import numpy as np
    from tqdm.notebook import trange
    from pathlib import Path
    from openvino_tokenizers import add_greedy_decoding
    from openvino_tokenizers.constants import EOS_TOKEN_ID_NAME
    from openvino import Core
    
    
    core = Core()
    
    # add the greedy decoding subgraph on top of LLM to get the most probable token as an output
    ov_model = add_greedy_decoding(core.read_model(model_dir / "openvino_model.xml"))
    compiled_model = core.compile_model(ov_model)
    infer_request = compiled_model.create_infer_request()

The ``infer_request`` object provides control over the model’s state - a
Key-Value cache that speeds up inference by reducing computations
Multiple inference requests can be created, and each request maintains a
distinct and separate state..

.. code:: ipython3

    text_input = ["Quick brown fox jumped"]
    
    model_input = {name.any_name: output for name, output in tokenizer(text_input).items()}
    
    if "position_ids" in (input.any_name for input in infer_request.model_inputs):
        model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]
    
    # no beam search, set idx to 0
    model_input["beam_idx"] = np.array([0], dtype=np.int32)
    # end of sentence token is that model signifies the end of text generation
    # read EOS token ID from rt_info of tokenizer/detokenizer ov.Model object
    eos_token = ov_tokenizer.get_rt_info(EOS_TOKEN_ID_NAME).value
    
    tokens_result = np.array([[]], dtype=np.int64)
    
    # reset KV cache inside the model before inference
    infer_request.reset_state()
    max_infer = 10
    
    for _ in trange(max_infer):
        infer_request.start_async(model_input)
        infer_request.wait()
    
        # get a prediction for the last token on the first inference
        output_token = infer_request.get_output_tensor().data[:, -1:]
        tokens_result = np.hstack((tokens_result, output_token))
        if output_token[0, 0] == eos_token:
            break
    
        # prepare input for new inference
        model_input["input_ids"] = output_token
        model_input["attention_mask"] = np.hstack((model_input["attention_mask"].data, [[1]]))
        model_input["position_ids"] = np.hstack(
            (
                model_input["position_ids"].data,
                [[model_input["position_ids"].data.shape[-1]]],
            )
        )
    
    text_result = detokenizer(tokens_result)["string_output"]
    print(f"Prompt:\n{text_input[0]}")
    print(f"Generated:\n{text_result[0]}")



.. parsed-literal::

      0%|          | 0/10 [00:00<?, ?it/s]


.. parsed-literal::

    Prompt:
    Quick brown fox jumped
    Generated:
    over the fence.
    
    
    
    
    


Merge Tokenizer into a Model
----------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Packages like ``tensorflow-text`` offer the convenience of integrating
text processing directly into the model, streamlining both distribution
and usage. Similarly, with OpenVINO Tokenizers, you can create models
that combine a converted tokenizer and a model. It’s important to note
that not all scenarios benefit from this merge. In cases where a
tokenizer is used once and a model is inferred multiple times, as seen
in the earlier text generation example, maintaining a separate
(de)tokenizer and model is advisable to prevent unnecessary
tokenization-detokenization cycles during inference. Conversely, if both
a tokenizer and a model are used once in each pipeline inference,
merging simplifies the workflow and aids in avoiding the creation of
intermediate objects:

.. raw:: html

   <center>

.. raw:: html

   </center>

The OpenVINO Python API allows you to avoid this by using the
``share_inputs`` option during inference, but it requires additional
input from a developer every time the model is inferred. Combining the
models and tokenizers simplifies memory management.

.. code:: ipython3

    model_id = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    model_dir = Path(Path(model_id).name)
    
    if not model_dir.exists():
        %pip install -qU git+https://github.com/huggingface/optimum-intel.git onnx
        !optimum-cli export openvino --model $model_id --task text-classification $model_dir
        !convert_tokenizer $model_id -o $model_dir


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    mobileclip 0.1.0 requires torch==1.13.1, but you have torch 2.3.1+cpu which is incompatible.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.18.1+cpu which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    2024-07-02 01:03:42.804844: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    Framework not specified. Using pt to export the model.
    Using framework PyTorch: 2.3.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False
    Detokenizer is not supported, convert tokenizer only.


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    Loading Huggingface Tokenizer...
    Converting Huggingface Tokenizer to OpenVINO...
    Saved OpenVINO Tokenizer: bert-tiny-finetuned-sms-spam-detection/openvino_tokenizer.xml, bert-tiny-finetuned-sms-spam-detection/openvino_tokenizer.bin


.. code:: ipython3

    from openvino import Core, save_model
    from openvino_tokenizers import connect_models
    
    
    core = Core()
    text_input = ["Free money!!!"]
    
    ov_tokenizer = core.read_model(model_dir / "openvino_tokenizer.xml")
    ov_model = core.read_model(model_dir / "openvino_model.xml")
    combined_model = connect_models(ov_tokenizer, ov_model)
    save_model(combined_model, model_dir / "combined_openvino_model.xml")
    
    compiled_combined_model = core.compile_model(combined_model)
    openvino_output = compiled_combined_model(text_input)
    
    print(f"Logits: {openvino_output['logits']}")


.. parsed-literal::

    Logits: [[ 1.2007061 -1.4698029]]


Conclusion
----------

`back to top ⬆️ <#Table-of-contents:>`__

The OpenVINO Tokenizers integrate text processing operations into the
OpenVINO ecosystem. Enabling the conversion of HuggingFace tokenizers
into OpenVINO models, the library allows efficient deployment of deep
learning pipelines across varied environments. The feature of combining
tokenizers and models not only simplifies memory management but also
helps to streamline model usage and deployment.

Links
-----

`back to top ⬆️ <#Table-of-contents:>`__

-  `Installation instructions for different
   environments <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#installation>`__
-  `Supported Tokenizer
   Types <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#supported-tokenizer-types>`__
-  `OpenVINO.GenAI repository with the C++ example of OpenVINO
   Tokenizers
   usage <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/greedy_causal_lm>`__
-  `HuggingFace Tokenizers Comparison
   Table <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#output-match-by-model>`__
