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


**Table of contents:**


-  `Tokenization Basics <#tokenization-basics>`__
-  `Acquiring OpenVINO Tokenizers <#acquiring-openvino-tokenizers>`__

   -  `Convert Tokenizer from HuggingFace Hub with CLI
      Tool <#convert-tokenizer-from_huggingface-hub-with-cli-tool>`__
   -  `Convert Tokenizer from HuggingFace Hub with Python
      API <#convert-tokenizer-from-huggingface-hub-with-python-api>`__

-  `Text Generation Pipeline with OpenVINO
   Tokenizers <#text-generation-pipeline-with-openvino-tokenizers>`__
-  `Text Generation Pipeline with OpenVINO GenAI and OpenVINO
   Tokenizers <#text-generation-pipeline-with-openvino-genai-and-openvino-tokenizers>`__
-  `Merge Tokenizer into a Model <#merge-tokenizer-into-a-model>`__
-  `Conclusion <#conclusion>`__
-  `Links <#links>`__

Tokenization Basics
-------------------



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


This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Acquiring OpenVINO Tokenizers
-----------------------------



OpenVINO Tokenizers Python library allows you to convert HuggingFace
tokenizers into OpenVINO models. To install all required dependencies
use ``pip install openvino-tokenizers[transformers]``.

.. code:: ipython3

    %pip install -Uq pip
    %pip install -q -U "openvino>=2024.3.0" openvino-tokenizers[transformers] openvino-genai
    %pip install "numpy<2.0.0" "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu

.. code:: ipython3

    from pathlib import Path
    
    
    tokenizer_dir = Path("tokenizer/")
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

Convert Tokenizer from HuggingFace Hub with CLI Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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

   ⚠️ Import ``openvino_tokenizers`` will add all tokenizer-related
   operations to OpenVINO, after which you can work with saved
   tokenizers and detokenizers.

.. code:: ipython3

    import openvino as ov
    
    # This import is needed to add all tokenizer-related operations to OpenVINO
    import openvino_tokenizers  # noqa: F401
    
    
    ov.save_model(ov_tokenizer, tokenizer_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, tokenizer_dir / "openvino_detokenizer.xml")

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

    tokenizer, detokenizer = ov.compile_model(ov_tokenizer), ov.compile_model(ov_detokenizer)
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
    Detokenized text: ['<s> Test', '<s> strings']
    

Text Generation Pipeline with OpenVINO Tokenizers
-------------------------------------------------



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
        # Converting the original model
        # %pip install -U "git+https://github.com/huggingface/optimum-intel.git" "nncf>=2.8.0" onnx
        # %optimum-cli export openvino -m $model_id --task text-generation-with-past $model_dir
    
        # Load already converted model
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

.. code:: ipython3

    import numpy as np
    from tqdm.notebook import trange
    from pathlib import Path
    from openvino_tokenizers.constants import EOS_TOKEN_ID_NAME
    
    
    core = ov.Core()
    
    ov_model = core.read_model(model_dir / "openvino_model.xml")
    compiled_model = core.compile_model(ov_model)
    infer_request = compiled_model.create_infer_request()

The ``infer_request`` object provides control over the model’s state - a
Key-Value cache that speeds up inference by reducing computations.
Multiple inference requests can be created, and each request maintains a
distinct and separate state.

.. code:: ipython3

    text_input = ["Quick brown fox jumped"]
    
    model_input = {name.any_name: output for name, output in tokenizer(text_input).items()}
    
    if "position_ids" in (input.any_name for input in infer_request.model_inputs):
        model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]
    
    # No beam search, set idx to 0
    model_input["beam_idx"] = np.array([0], dtype=np.int32)
    
    # End of sentence token is that model signifies the end of text generation
    # Read EOS token ID from rt_info of tokenizer/detokenizer ov.Model object
    eos_token = ov_tokenizer.get_rt_info(EOS_TOKEN_ID_NAME).value
    
    tokens_result = np.array([[]], dtype=np.int64)
    
    # Reset KV cache inside the model before inference
    infer_request.reset_state()
    max_infer = 5
    
    for _ in trange(max_infer):
        infer_request.start_async(model_input)
        infer_request.wait()
    
        output_tensor = infer_request.get_output_tensor()
    
        # Get the most probable token
        token_indices = np.argmax(output_tensor.data, axis=-1)
        output_token = token_indices[:, -1:]
    
        # Concatenate previous tokens result with newly generated token
        tokens_result = np.hstack((tokens_result, output_token))
        if output_token[0, 0] == eos_token:
            break
    
        # Prepare input for the next inference iteration
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

      0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    Prompt:
    Quick brown fox jumped
    Generated:
    over the fence.
    

Text Generation Pipeline with OpenVINO GenAI and OpenVINO Tokenizers
--------------------------------------------------------------------



`OpenVINO GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
is a flavor of OpenVINO, aiming to simplify running inference of
generative AI models. It hides the complexity of the generation process
and minimizes the amount of code required. OpenVINO GenAI depends on
`OpenVINO <https://github.com/openvinotoolkit/openvino>`__ and `OpenVINO
Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__.

Firstly we need to create a pipeline with ``LLMPipeline``.
``LLMPipeline`` is the main object used for text generation using LLM in
OpenVINO GenAI API. You can construct it straight away from the folder
where both converted model and tokenizer are located,
e.g. ``ov_genai.LLMPipeline(model_and_tokenizer_path)``.

As the model and tokenizer are located in different directories, we
create a ``ov_genai.Tokenizer`` object by providing the path to saved
tokenizer. Then we will provide directory with model, tokenizer object
and device for ``LLMPipeline``. Lastly we run ``generate`` method and
get the output in text format.

Additionally, we can configure parameters for decoding. We can get the
default config with ``get_generation_config()``, setup parameters, and
apply the updated version with ``set_generation_config(config)`` or put
config directly to ``generate()``. It’s also possible to specify the
needed options just as inputs in the ``generate()`` method, as shown
below, e.g. we can add ``max_new_tokens`` to stop generation if a
specified number of tokens is generated and the end of generation is not
reached.

Let’s build the same text generation pipeline, but with simplified
Python `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
We will use the same model and tokenizer downloaded in previous steps.

.. code:: ipython3

    import openvino_genai as ov_genai
    
    genai_tokenizer = ov_genai.Tokenizer(str(tokenizer_dir))
    pipe = ov_genai.LLMPipeline(str(model_dir), genai_tokenizer, "CPU")
    
    result = pipe.generate(text_input[0], max_new_tokens=max_infer)
    
    print(f"Prompt:\n{text_input[0]}")
    print(f"Generated:\n{result}")


.. parsed-literal::

    Prompt:
    Quick brown fox jumped
    Generated:
    over the lazy dog.
    

Merge Tokenizer into a Model
----------------------------



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
models and tokenizers simplifies memory management. Moreover, after the
combining models inputs have changed - original model has three inputs
(``input_ids``, ``attention_mask``, ``token_type_ids``) and combined
model has only one input for text input prompt.

.. code:: ipython3

    model_id = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    model_dir = Path(Path(model_id).name)
    
    if not model_dir.exists():
        %pip install -qU git+https://github.com/huggingface/optimum-intel.git "onnx<1.16.2"
        !optimum-cli export openvino --model $model_id --task text-classification $model_dir

.. code:: ipython3

    from openvino_tokenizers import connect_models
    
    
    core = ov.Core()
    text_input = ["Free money!!!"]
    
    ov_tokenizer = core.read_model(model_dir / "openvino_tokenizer.xml")
    ov_model = core.read_model(model_dir / "openvino_model.xml")
    combined_model = connect_models(ov_tokenizer, ov_model)
    ov.save_model(combined_model, model_dir / "combined_openvino_model.xml")
    
    print("Original OpenVINO model inputs:")
    for input in ov_model.inputs:
        print(input)
    
    print("\nCombined OpenVINO model inputs:")
    for input in combined_model.inputs:
        print(input)
    
    compiled_combined_model = core.compile_model(combined_model)
    openvino_output = compiled_combined_model(text_input)
    
    print(f"\nLogits: {openvino_output['logits']}")


.. parsed-literal::

    Original OpenVINO model inputs:
    <Output: names[input_ids] shape[?,?] type: i64>
    <Output: names[attention_mask] shape[?,?] type: i64>
    <Output: names[token_type_ids] shape[?,?] type: i64>
    
    Combined OpenVINO model inputs:
    <Output: names[Parameter_4430] shape[?] type: string>
    
    Logits: [[ 1.2007061 -1.469803 ]]
    

Conclusion
----------



The OpenVINO Tokenizers integrate text processing operations into the
OpenVINO ecosystem. Enabling the conversion of HuggingFace tokenizers
into OpenVINO models, the library allows efficient deployment of deep
learning pipelines across varied environments. The feature of combining
tokenizers and models not only simplifies memory management but also
helps to streamline model usage and deployment.

Links
-----



-  `Installation instructions for different
   environments <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#installation>`__
-  `Supported Tokenizer
   Types <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#supported-tokenizer-types>`__
-  `OpenVINO.GenAI repository with the C++ example of OpenVINO
   Tokenizers
   usage <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/greedy_causal_lm>`__
-  `HuggingFace Tokenizers Comparison
   Table <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#output-match-by-model>`__
