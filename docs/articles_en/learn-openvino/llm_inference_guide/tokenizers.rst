.. {#tokenizers}

OpenVINO Tokenizers
===============================

Tokenization is a required step in generating text using any text models, including LLMs.
Tokenization converts the input text into a sequence of tokens, which the model can understand
and process before running inference. The transformation of a sequence of numbers into a
string is called detokenization.

.. image:: _static/images/tokenization.png
   :align: center

There are two important points in the tokenizer-model relation:

* Every model with text input is paired with a tokenizer and cannot be used without it.
* To reproduce the model accuracy on a specific task, it is essential to use the same tokenizer employed during the model training.

**OpenVINO Tokenizers** is an OpenVINO extension and a Python library designed to streamline
tokenizer conversion for seamless integration into your projects. With OpenVINO Tokenizers you can:

* Add text processing operations to OpenVINO. Both tokenizer and detokenizer are OpenVINO models, meaning that you can work with them as with any regular model: read, compile, save, etc.

* Perform tokenization and detokenization without third-party dependencies.

* Convert Hugging Face tokenizers into OpenVINO model tokenizer and detokenizer for efficient deployment across different environments. See the `conversion example <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#convert-huggingface-tokenizer>`__ for more details.

* Combine OpenVINO models into a single model. Recommended for specific models, like classifiers or RAG Embedders, where both tokenizer and a model are used once in each pipeline inference. For more information, see the `OpenVINO Tokenizers Notebook <https://github.com/openvinotoolkit/openvino_notebooks/blob/master/notebooks/openvino-tokenizers/openvino-tokenizers.ipynb>`__.

* Add greedy decoding pipeline to text generation model.

* Use TensorFlow models, such as TensorFlow Text MUSE model. See the `MUSE model inference example <https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#tensorflow-text-integration>`__ to learn more.  Note that TensorFlow integration requires additional conversion extensions to work with string tensor operations like StringSplit, StaticRexexpReplace, StringLower, and others.

Supported Tokenizers
#####################

.. note::

   OpenVINO Tokenizers can be inferred **only** on a CPU device.

.. list-table::
   :widths: 30 25 20 20
   :header-rows: 1

   * - Hugging Face Tokenizer Type
     - Tokenizer Model Type
     - Tokenizer
     - Detokenizer
   * - Fast
     - WordPiece
     - yes
     - no
   * -
     - BPE
     - yes
     - yes
   * -
     - Unigram
     - no
     - no
   * - Legacy
     - SentencePiece .model
     - yes
     - yes
   * - Custom
     - tiktoken
     - yes
     - yes

.. note::
   The outputs of the converted and the original tokenizer can differ, either decreasing or increasing
   model accuracy on a specific task. You can modify the prompt to mitigate these changes.
   In the `OpenVINO Tokenizers repository <https://github.com/openvinotoolkit/openvino_tokenizers>`__
   you can see the percentage of tests in which the output of the original and converted tokenizer/detokenizer match.

Python Installation
###################


1. Create and activate a virtual environment.

   .. code-block:: python

      python3 -m venv venv

      source venv/bin/activate

2. Install OpenVINO Tokenizers.

   If you have a converted OpenVINO tokenizer:

   .. code-block:: python

      pip install openvino-tokenizers

   If you want to convert Hugging Face tokenizers into OpenVINO tokenizers:

   .. code-block:: python

      pip install openvino-tokenizers[transformers]

   Install pre-release versions, if you want to experiment with latest changes:

   .. code-block:: python

      pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

   Build and install from source:

   .. code-block:: python

      source path/to/installed/openvino/setupvars.sh

            git clone https://github.com/openvinotoolkit/openvino_tokenizers.git

      cd openvino_tokenizers

      pip install --no-deps .

You can also install OpenVINO Tokenizers with Conda distribution. Check `the OpenVINO Tokenizers
repository <https://github.com/openvinotoolkit/openvino_tokenizers.git>`__ for more
information.

C++ Installation
################

You can use converted tokenizers in C++ pipelines with prebuild binaries.

1. Download :doc:`OpenVINO archive distribution <../../get-started/install-openvino>` for your OS and extract the archive.

2. Download `OpenVINO Tokenizers prebuild libraries <https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/>`__. To ensure compatibility, the first three numbers of the OpenVINO Tokenizers version should match the OpenVINO version and OS.

3. Extract OpenVINO Tokenizers archive into OpenVINO installation directory:

.. tab-set::

   .. tab-item:: Linux_x86

      .. code-block:: sh

         <openvino_dir>/runtime/lib/intel64/

   .. tab-item:: Linux_arm64

      .. code-block:: sh

         <openvino_dir>/runtime/lib/aarch64/

   .. tab-item:: Windows

      .. code-block:: sh

         <openvino_dir>\runtime\bin\intel64\Release\

   .. tab-item:: MacOS_x86

      .. code-block:: sh

         <openvino_dir>/runtime/lib/intel64/Release

   .. tab-item:: MacOS_arm64

      .. code-block:: sh

         <openvino_dir>/runtime/lib/arm64/Release/

After that you can add binary extension in the code with:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: sh

         core.add_extension("libopenvino_tokenizers.so")

   .. tab-item:: Windows

      .. code-block:: sh

         core.add_extension("openvino_tokenizers.dll")

   .. tab-item:: MacOS

      .. code-block:: sh

         core.add_extension("libopenvino_tokenizers.dylib") 


If you use the ``2023.3.0.0`` version, the binary extension file is called ``(lib)user_ov_extension.(dll/dylib/so)``.

You can learn how to read and compile converted models in the
:doc:`Model Preparation <../../openvino-workflow/model-preparation>` guide.

Tokenizers Usage
################

1. Convert a Tokenizer to OpenVINO Intermediate Representation (IR)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can convert Hugging Face tokenizers to IR using either a CLI tool bundled with Tokenizers or
Python API. Skip this step if you have a converted OpenVINO tokenizer.

Install dependencies:

.. code-block:: python

   pip install openvino-tokenizers[transformers]

Convert Tokenizers:

.. tab-set::

   .. tab-item:: CLI

      .. code-block:: sh

         !convert_tokenizer $model_id --with-detokenizer -o $tokenizer_dir

   .. tab-item:: Python API

      .. code-block:: python

         from transformers import AutoTokenizer
         from openvino_tokenizers import convert_tokenizer

         hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
         ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
         ov_tokenizer, ov_detokenizer

The result is two OpenVINO models: openvino tokenizer and openvino detokenizer.
Both can be used with ``read_model``, ``compile_model`` and ``save_model``, similar to any other OpenVINO model.

2. Tokenize and Prepare Inputs
+++++++++++++++++++++++++++++++

.. code-block:: python

   text_input = ["Quick brown fox jumped"]

   model_input = {name.any_name: output for name, output in tokenizer(text_input).items()}

   if "position_ids" in (input.any_name for input in infer_request.model_inputs):
      model_input["position_ids"] = np.arange(model_input["input_ids"].shape[1], dtype=np.int64)[np.newaxis, :]

   # no beam search, set idx to 0
   model_input["beam_idx"] = np.array([0], dtype=np.int32)
   # end of sentence token is where the model signifies the end of text generation
   # read EOS token ID from rt_info of tokenizer/detokenizer ov.Model object
   eos_token = ov_tokenizer.get_rt_info(EOS_TOKEN_ID_NAME).value

3. Generate Text
+++++++++++++++++++++++++++

.. code-block:: python

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
         (model_input["position_ids"].data, [[model_input["position_ids"].data.shape[-1]]])
      )

4. Detokenize Output
+++++++++++++++++++++++++++++

.. code-block:: python

   text_result = detokenizer(tokens_result)["string_output"]
   print(f"Prompt:\n{text_input[0]}")
   print(f"Generated:\n{text_result[0]}")


Additional Resources
####################

* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
* `OpenVINO Tokenizers Notebook <https://github.com/openvinotoolkit/openvino_notebooks/blob/master/notebooks/openvino-tokenizers/openvino-tokenizers.ipynb>`__
* `Text generation C++ samples that support most popular models like LLaMA 2 <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__


