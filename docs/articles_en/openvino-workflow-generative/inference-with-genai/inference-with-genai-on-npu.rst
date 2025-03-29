NPU with OpenVINO GenAI
===============================================================================================

.. meta::
   :description: Learn how to use OpenVINO GenAI to execute LLM models on NPU.


This guide will give you extra details on how to use NPU with OpenVINO GenAI.
:doc:`See the installation guide <../../get-started/install-openvino/install-openvino-genai>`
for information on how to start.

Prerequisites
###############################################################################################

Install required dependencies:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: console

         python3 -m venv npu-env
         npu-env/bin/activate
         pip install  nncf==2.14.1 onnx==1.17.0 optimum-intel==1.21.0
         pip install openvino==2025.1 openvino-tokenizers==2025.1 openvino-genai==2025.1

      For the pre-production version, use the following line, instead:

      .. code-block:: console

         pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


   .. tab-item:: Windows

      .. code-block:: console

         python -m venv npu-env
         npu-env\Scripts\activate
         pip install  nncf==2.14.1 onnx==1.17.0 optimum-intel==1.21.0
         pip install openvino==2025.1 openvino-tokenizers==2025.1 openvino-genai==2025.1

      For the pre-production version, use the following line, instead:

      .. code-block:: console

         pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


Note that for systems based on Intel® Core™ Ultra Processors Series 2, more than 16GB of RAM
may be required to run prompts over 1024 tokens on models exceeding 7B parameters,
such as Llama-2-7B, Mistral-0.2-7B, and Qwen-2-7B.

Make sure your model works with NPU. Some models may not be supported, for example,
**the FLUX.1 pipeline is currently not supported by the device**.

Currently, the Whisper pipeline (using:
`whisper-tiny <https://huggingface.co/openai/whisper-tiny>`__,
`whisper-base <https://huggingface.co/openai/whisper-base>`__,
`whisper-small <https://huggingface.co/openai/whisper-small>`__, or
`whisper-large <https://huggingface.co/openai/whisper-large>`__)
only accepts models generated with the ``--disable-stateful`` flag.
Here is a conversion example:

.. code:: console

   optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny --disable-stateful



Export an LLM model via Hugging Face Optimum-Intel
###############################################################################################

Since **symmetrically-quantized 4-bit (INT4) models are preferred for inference on NPU**, make
sure to export the model with the proper conversion and optimization settings.

| You may export LLMs via Optimum-Intel, using one of two compression methods:
| **group quantization** - for both smaller and larger models,
| **channel-wise quantization** - remarkably effective but for models exceeding 1 billion parameters.

You select one of the methods by setting the ``--group-size`` parameter to either ``128`` or
``-1``, respectively. See the following examples:

.. tab-set::

   .. tab-item:: Channel-wise quantization

      .. tab-set::

         .. tab-item:: Data-free quantization


            .. code-block:: console
               :name: channel-wise-data-free-quant

               optimum-cli export openvino -m meta-llama/Llama-2-7b-chat-hf --weight-format int4 --sym --ratio 1.0 --group-size -1 Llama-2-7b-chat-hf

         .. tab-item:: Data-aware quantization

            If you want to improve accuracy, make sure you:

            1. Update NNCF: ``pip install nncf==2.13``
            2. Use ``--scale_estimation --dataset <dataset_name>`` and accuracy aware quantization ``--awq``:

               .. code-block:: console
                  :name: channel-wise-data-aware-quant

                  optimum-cli export openvino -m meta-llama/Llama-2-7b-chat-hf --weight-format int4 --sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2  Llama-2-7b-chat-hf


      .. important::

         Remember that the negative value of ``-1`` is required here, not ``1``.

   .. tab-item:: Group quantization

      .. code-block:: console
         :name: group-quant

         optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group-size 128 TinyLlama-1.1B-Chat-v1.0



You can also try using 4-bit (INT4)
`GPTQ models <https://huggingface.co/models?other=gptq,4-bit&sort=trending>`__,
which do not require specifying quantization parameters:

.. code-block:: console

   optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ


| Remember, NPU supports GenAI models quantized symmetrically to INT4.
| Below is a list of such models:

* meta-llama/Meta-Llama-3-8B-Instruct
* meta-llama/Llama-3.1-8B
* microsoft/Phi-3-mini-4k-instruct
* Qwen/Qwen2-7B
* mistralai/Mistral-7B-Instruct-v0.2
* openbmb/MiniCPM-1B-sft-bf16
* TinyLlama/TinyLlama-1.1B-Chat-v1.0
* TheBloke/Llama-2-7B-Chat-GPTQ
* Qwen/Qwen2-7B-Instruct-GPTQ-Int4


Run generation using OpenVINO GenAI
###############################################################################################

It is typically recommended to install the latest available
`driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

Use the following code snippet to perform generation with OpenVINO GenAI API.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python
         :emphasize-lines: 4

         import openvino_genai as ov_genai
         model_path = "TinyLlama"
         pipe = ov_genai.LLMPipeline(model_path, "NPU")
         print(pipe.generate("The Sun is yellow because", max_new_tokens=100))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp
         :emphasize-lines: 7, 9

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
###############################################################################################

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
  in its response (default: 150).

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

``CACHE_DIR`` operates similarly to the older ``NPUW_CACHE_DIR``, except for two differences:

* It creates a single ".blob" file and loads it faster.
* It stores all model weights inside the blob, making it much bigger than individual compiled
  schedules for model's subgraphs stored by ``NPUW_CACHE_DIR``.

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
-----------------------------------------------------------------------------------------------

Specifying ``EXPORT_BLOB`` and ``BLOB_PATH`` parameters works similarly to ``CACHE_DIR`` but:

* It allows to explicitly specify where to **store** the compiled model.
* For subsequent runs, it requires the same ``BLOB_PATH`` to **import** the compiled model.

.. tab-set::

   .. tab-item:: Export example

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. code-block:: python

               pipeline_config = { "EXPORT_BLOB": "YES", "BLOB_PATH": ".npucache\\compiled_model.blob" }
               pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)


         .. tab-item:: C++
            :sync: cpp

            .. code-block:: cpp

               ov::AnyMap pipeline_config = { { "EXPORT_BLOB",  "YES" }, { "BLOB_PATH",  ".npucache\\compiled_model.blob" } };
               ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);

   .. tab-item:: Import example

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







Additional Resources
###############################################################################################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
