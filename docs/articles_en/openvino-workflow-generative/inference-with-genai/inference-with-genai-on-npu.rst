OpenVINO GenAI on NPU
===============================================================================================

.. meta::
   :description: Learn how to use OpenVINO GenAI to execute LLMs and other pipelines on NPU.


This guide will give you extra details on how to use NPU with OpenVINO GenAI.
:doc:`See the installation guide <../../get-started/install-openvino/install-openvino-genai>`
for information on how to start.

Prerequisites
###############################################################################################

Install the required dependencies:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: console

         python3 -m venv npu-env
         npu-env/bin/activate
         pip install nncf==2.18.0 onnx==1.18.0 optimum-intel==1.25.2 transformers==4.51.3
         pip install openvino==2025.3 openvino-tokenizers==2025.3 openvino-genai==2025.3


      For the pre-production version, use the following line, instead:

      .. code-block:: console

         pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


   .. tab-item:: Windows

      .. code-block:: console

         python -m venv npu-env
         npu-env\Scripts\activate
         pip install nncf==2.18.0 onnx==1.18.0 optimum-intel==1.25.2 transformers==4.51.3
         pip install openvino==2025.3 openvino-tokenizers==2025.3 openvino-genai==2025.3


      For the pre-production version, use the following line, instead:

      .. code-block:: console

         pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

.. note::

    With OpenVINO 2025.3, it is highly recommended to use ``transformers==4.51.3`` to
    generate models for Intel NPU. Please expect support for newer transformer versions in
    the future releases.

.. note::

    For systems based on Intel® Core™ Ultra Processors Series 2, more than 16GB of RAM
    may be required to process prompts longer than 1024 tokens with models exceeding 7B parameters,
    such as Llama-2-7B, Mistral-0.2-7B, and Qwen-2-7B.

LLM Inference on NPU
###############################################################################################

Export LLMs from Hugging Face for NPU
***********************************************************************************************

Optimum Intel is the primary way to export Hugging Face models for inference on NPU.
LLMs **must** be exported with the following settings:
* Symmetric weights compression: ``--sym``;
* 4-bit weight format (INT4 or NF4): ``--weight-format int4`` or ``--weight-format nf4``;
* Channel-wise or group-wise weight quantization: ``--group-size -1`` or ``--group-size 128``;
* Maximize the 4-bit weight ratio in the model: ``--ratio 1.0``.

**Group quantization** with group size ``128`` is recommended for smaller models, e.g. up to
4B..5B parameters. Larger models may also work with group-quantization, but normally demonstrate
a better performance with channel-wise quantization.

**Channel-wise quantization** usually performs best but may impact the model accuracy. OpenVINO
Neural Network Compression Framework (NNCF) provides various ways to compensate the quality loss,
e.g. data-aware compression methods or GPTQ.

.. important::

   The NF4 data type is only supported on Intel® Core Ultra Processors Series 2 NPUs (formerly
   codenamed Lunar Lake) and beyond. Please make sure to use channel-wise quantization with NF4.

The full ``optimum-cli`` command examples are shown below:

.. tab-set::

   .. tab-item:: Channel-wise quantization, INT4, data-free

      .. code-block:: console
         :name: channel-wise-data-free-quant

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format int4 --sym --ratio 1.0 --group-size -1 Meta-Llama-3.1-8B-Instruct

   .. tab-item:: Channel-wise quantization, INT4, data-aware

      Use Scale Estimation (``--scale_estimation``) and/or AWQ (``--awq``) to improve accuracy
      for the channel-wise quantized models. Note that these options require a dataset
      (``--dataset <dataset_name>``). Refer to ``optimum-cli`` and NNCF documentation for more details.

      .. code-block:: console
         :name: channel-wise-data-aware-quant

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format int4 --sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2 Meta-Llama-3.1-8B-Instruct

   .. tab-item:: Channel-wise quantization, NF4, data-free

      .. code-block:: console
         :name: channel-wise-data-free-quant-nf4

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format nf4 --sym --group-size -1 --ratio 1.0  Meta-Llama-3.1-8B-Instruct

   .. tab-item:: Group quantization, INT4

      .. code-block:: console
         :name: group-quant

         optimum-cli export openvino -m microsoft/Phi-3.5-mini-instruct --weight-format int4 --sym --ratio 1.0 --group-size 128 Phi-3.5-mini-instruct

   .. important::

      For the channel-wise quantization, the group size argument must be ``-1`` ("minus one"), not ``1``.

There are pre-compressed models on Hugging Face that can be exported as-is, e.g.
- 4-bit (INT4) `GPTQ models <https://huggingface.co/models?other=gptq,4-bit&sort=trending>`__
- `LLMs optimized for NPU <https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0>`__, hosted by OpenVINO.
In this case, the commands are as simple as:

.. code-block:: console

   optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ
   optimum-cli export openvino -m OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov

Run text generation on NPU
***********************************************************************************************

It is typically recommended to install the latest available
`driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

Use the following code snippet to perform generation with OpenVINO GenAI API.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai as ov_genai
         model_path = "TinyLlama"
         pipe = ov_genai.LLMPipeline(model_path, "NPU")
         print(pipe.generate("The Sun is yellow because", max_new_tokens=100))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "openvino/genai/llm_pipeline.hpp"
         #include <iostream>

         int main(int argc, char* argv[]) {
            std::string model_path = "TinyLlama";
            ov::genai::LLMPipeline pipe(models_path, "NPU");
            ov::genai::GenerationConfig config;
            config.max_new_tokens=100;
            std::cout << pipe.generate("The Sun is yellow because", config);
         }


Additional configuration options
***********************************************************************************************

Prompt and response length options
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The LLM pipeline for NPUs leverages the static shape approach, optimizing execution performance,
while potentially introducing certain usage limitations. By default, the LLM pipeline supports
input prompts up to 1024 tokens in length. It also ensures that the generated response contains
at least 150 tokens, unless the generation encounters the end-of-sequence (EOS) token or the
user explicitly sets a lower length limit for the response.

You may configure both the 'maximum input prompt length' and 'minimum response length' using
the following parameters:

* ``MAX_PROMPT_LEN`` - defines the maximum number of tokens that the LLM pipeline can process
  for the input prompt (default: 1024),
* ``MIN_RESPONSE_LEN`` - defines the minimum number of tokens that the LLM pipeline will generate
  in its response (default: 128).

Use the following code snippet to change the default settings:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         pipeline_config = { "MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 512 }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "MAX_PROMPT_LEN",  1024 }, { "MIN_RESPONSE_LEN", 512 } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Performance modes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can configure the NPU pipeline with the ``GENERATE_HINT`` option to switch
between two different performance modes:

* ``FAST_COMPILE`` (default) - enables fast compilation at the expense of performance,
* ``BEST_PERF`` - ensures best possible performance at lower compilation speed.

Use the following code snippet:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         pipeline_config = { "GENERATE_HINT": "BEST_PERF" }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "GENERATE_HINT",  "BEST_PERF" } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Cache compiled models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

By caching compiled models, you may shorten the initialization time of the future pipeline
runs. To do so, specify one of the following options in ``pipeline_config`` for NPU pipeline.

NPUW_CACHE_DIR
-----------------------------------------------------------------------------------------------

``NPUW_CACHE_DIR`` is the most basic option of caching compiled subgraphs without weights and
reusing them for future pipeline runs.


CACHE_DIR
-----------------------------------------------------------------------------------------------

``CACHE_DIR`` operates similarly to the older ``NPUW_CACHE_DIR``, except for the differences below:

* It creates a single ".blob" file and loads it faster.
* Blob type is defined by ``"CACHE_MODE"``. By default it's ``"OPTIMIZE_SIZE"``, in which case NPUW
  produces weightless blob, so either original weights file or ``ov::Model`` object is required
  to load such a blob.
* Optionally, you can cache a blob with weights inside making it much bigger than the default
  weightless blob. To do so, you need to pass ``"CACHE_MODE" : "OPTIMIZE_SPEED"`` in the config.

.. tab-set::

   .. tab-item:: Python example
      :sync: py

      .. code-block:: python

         pipeline_config = { "CACHE_DIR": ".npucache" }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++ example
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap pipeline_config = { { "CACHE_DIR",  ".npucache" } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


'Ahead of time' compilation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Specifying ``EXPORT_BLOB`` and ``BLOB_PATH`` parameters works similarly to ``CACHE_DIR`` but:

* It allows to explicitly specify where to **store** the compiled model.
* For subsequent runs, it requires the same ``BLOB_PATH`` to **import** the compiled model.
* Blob type is defined by ``"CACHE_MODE"``. By default it's ``"OPTIMIZE_SIZE"``, in which case NPUW
  produces weightless blob, so either original weights file or ``ov::Model`` object is required
  to load such a blob.
* To export a blob with weights you need to pass ``"CACHE_MODE" : "OPTIMIZE_SPEED"`` in the config.
* If the blob is exported as weightless you also need to either provide
  ``"WEIGHTS_PATH" : "path\\to\\original\\model.bin"`` or ``"MODEL_PTR" : original ov::Model object``.
* Ahead-of-time import in weightless mode has been optimized to consume less memory than during regular compilation or using ``CACHE_DIR``.

.. tab-set::

   .. tab-item:: Export weightless example

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: python

               pipeline_config = { "EXPORT_BLOB": "YES", "BLOB_PATH": ".npucache\\compiled_model.blob" }
               pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)


         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::AnyMap pipeline_config = { { "EXPORT_BLOB", "YES" }, { "BLOB_PATH", ".npucache\\compiled_model.blob" } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

   .. tab-item:: Import weightless example

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: python

               pipeline_config = { "BLOB_PATH": ".npucache\\compiled_model.blob", "WEIGHTS_PATH": "path\\to\\original\\model.bin" }
               pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)


         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::AnyMap pipeline_config = { { "BLOB_PATH", ".npucache\\compiled_model.blob" }, { "WEIGHTS_PATH", "path\\to\\original\\model.bin" } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

   .. tab-item:: Export with weights example

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: python

               pipeline_config = { "EXPORT_BLOB": "YES", "BLOB_PATH": ".npucache\\compiled_model.blob", "CACHE_MODE" : "OPTIMIZE_SPEED" }
               pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)


         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::AnyMap pipeline_config = { { "EXPORT_BLOB", "YES" }, { "BLOB_PATH", ".npucache\\compiled_model.blob" }, { "CACHE_MODE", "OPTIMIZE_SPEED" } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

   .. tab-item:: Import with weights example

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: python

               pipeline_config = { "BLOB_PATH": ".npucache\\compiled_model.blob" }
               pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)


         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::AnyMap pipeline_config = { { "BLOB_PATH",  ".npucache\\compiled_model.blob" } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Blob encryption
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When exporting NPUW blobs you can also specify encryption and decryption functions for the blob.
In case of weightless blob the whole blob is encrypted, in case of blob with weights everything but
model weights is encrypted.

.. tab-set::

   .. tab-item:: Export example

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::EncryptionCallbacks encryption_callbacks;
               encryption_callbacks.encrypt = [](const std::string& s) { return s; };
               ov::AnyMap pipeline_config = { { "EXPORT_BLOB", "YES" }, { "BLOB_PATH", ".npucache\\compiled_model.blob" }, { "CACHE_ENCRYPTION_CALLBACKS", encryption_callbacks } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

   .. tab-item:: Import example

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::EncryptionCallbacks encryption_callbacks;
               encryption_callbacks.decrypt = [](const std::string& s) { return s; };
               ov::AnyMap pipeline_config = { { "BLOB_PATH", ".npucache\\compiled_model.blob" }, { "WEIGHTS_PATH", "path\\to\\original\\model.bin" }, { "CACHE_ENCRYPTION_CALLBACKS", encryption_callbacks } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Disable memory allocation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In case of execution failures, either silent or with errors, try to update the NPU driver to
`32.0.100.3104 or newer <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.
If the update is not possible, set the ``DISABLE_OPENVINO_GENAI_NPU_L0``
environment variable to disable NPU memory allocation, which might be supported
only on newer drivers for Intel Core Ultra 200V processors.

Set the environment variable in a terminal:

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: console

         export DISABLE_OPENVINO_GENAI_NPU_L0=1

   .. tab-item:: Windows
      :sync: win

      .. code-block:: console

         set DISABLE_OPENVINO_GENAI_NPU_L0=1



Whisper (speech to text) on NPU with OpenVINO GenAI
###############################################################################################

Currently, the Whisper pipeline (using:
`whisper-tiny <https://huggingface.co/openai/whisper-tiny>`__,
`whisper-base <https://huggingface.co/openai/whisper-base>`__,
`whisper-small <https://huggingface.co/openai/whisper-small>`__, or
`whisper-large <https://huggingface.co/openai/whisper-large>`__)
only accepts stateless models. The pipeline will convert stateful models to stateless models automatically or you can manually generate stateless models with the ``--disable-stateful`` flag.
Here is a conversion example:

.. code:: console

   optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny --disable-stateful










Additional Resources
###############################################################################################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `Optimum Intel <https://github.com/huggingface/optimum-intel>`__
