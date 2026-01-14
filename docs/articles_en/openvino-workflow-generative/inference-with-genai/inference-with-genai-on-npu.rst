OpenVINO GenAI on NPU
===============================================================================================

.. meta::
   :description: Learn how to use OpenVINO GenAI to execute LLMs and other pipelines on NPU.


This guide will give you extra details on how to use NPU with OpenVINO GenAI.
:doc:`See the installation guide <../../get-started/install-openvino/install-openvino-genai>`
for information on how to start.

Prerequisites
###############################################################################################

First, install the required dependencies for the model conversion:

.. tab-set::

   .. tab-item:: Linux

      .. code-block:: console

         python3 -m venv npu-env
         source npu-env/bin/activate
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
    generate models for Intel NPU. Newer Transformers versions will be supported in upcoming releases.

.. note::

    For systems based on Intel® Core™ Ultra Processors Series 2, more than 16GB of RAM
    may be required to process prompts longer than 1024 tokens with models exceeding 7B parameters,
    such as Llama-2-7B, Mistral-0.2-7B, and Qwen-2-7B.

LLM Inference on NPU
###############################################################################################

Export LLMs from Hugging Face
***********************************************************************************************

Optimum Intel is the primary way to export Hugging Face models for inference on NPU.
LLMs **must** be exported with the following settings:

* Symmetric weights compression: ``--sym``;
* 4-bit weight format (INT4 or NF4): ``--weight-format int4`` or ``--weight-format nf4``;
* Channel-wise or group-wise weight quantization: ``--group-size -1`` or ``--group-size 128``;
* Maximize the 4-bit weight ratio in the model: ``--ratio 1.0``.

**Group quantization** (GQ) with group size ``128`` is recommended for smaller models, e.g. up to
4B--5B parameters. Larger models may also work with group-quantization, but normally demonstrate
a better performance with channel-wise quantization.

**Channel-wise quantization** (CW) generally offers the best performance but may reduce model accuracy. OpenVINO
Neural Network Compression Framework (NNCF) provides several methods to compensate for the quality loss,
such as data-aware compression methods or GPTQ.

The full ``optimum-cli`` command examples are shown below:

.. tab-set::

   .. tab-item:: INT4-CW

      INT4 Symmetric channel-wise data-free compression:

      .. code-block:: console
         :name: channel-wise-data-free-quant

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format int4 --sym --ratio 1.0 --group-size -1 Meta-Llama-3.1-8B-Instruct

   .. tab-item:: INT4-CW, data-aware

      INT4 Symmetric data-aware channel-wise compression:

      Use Scale Estimation (``--scale_estimation``) and/or AWQ (``--awq``) to improve accuracy
      for the channel-wise quantized models. Note that these options require a dataset
      (``--dataset <dataset_name>``). Refer to ``optimum-cli`` and NNCF documentation for more details.

      .. code-block:: console
         :name: channel-wise-data-aware-quant

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format int4 --sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2 Meta-Llama-3.1-8B-Instruct

   .. tab-item:: NF4-CW

      NF4 Symmetric data-free channel-wise compression:

      .. code-block:: console
         :name: channel-wise-data-free-quant-nf4

         optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --weight-format nf4 --sym --group-size -1 --ratio 1.0  Meta-Llama-3.1-8B-Instruct

      Usually, NF4-CW provides a better accuracy compared to INT4-CW even with data-free compression. Data-aware methods are also available and can further improve the compressed model accuracy.

   .. tab-item:: INT4-GQ

      INT4 Symmetric data-free group quantization:

      .. code-block:: console
         :name: group-quant

         optimum-cli export openvino -m microsoft/Phi-3.5-mini-instruct --weight-format int4 --sym --ratio 1.0 --group-size 128 Phi-3.5-mini-instruct

      Data-aware methods are also available for the group-quantized models and can further improve the compressed model accuracy.

.. note::

   The NF4 data type is only supported on Intel® Core Ultra Processors Series 2 NPUs (formerly
   codenamed Lunar Lake) and beyond. Use channel-wise quantization with NF4.

.. important::

   For the channel-wise quantization, the group size argument must be ``-1`` ("minus one"), not ``1``.

There are pre-compressed models on Hugging Face that can be exported as-is, for example:

* 4-bit (INT4) `GPTQ models <https://huggingface.co/models?other=gptq,4-bit&sort=trending>`__,
* `LLMs optimized for NPU <https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0>`__, hosted and maintained by OpenVINO.

In this case, the commands are as simple as:

.. code-block:: console

   optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ
   optimum-cli export openvino -m OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov

Run text generation
***********************************************************************************************

It is recommended to install the latest available
`NPU driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.
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

.. important::

   The options described in this article are specific to the NPU device and
   may not work with other devices.

Prompt and response length options
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The LLM pipeline for NPUs leverages the static shape approach, optimizing execution performance,
while potentially introducing certain usage limitations. By default, the LLM pipeline supports
input prompts up to 1024 tokens in length. It also ensures that the generated response contains
at least 128 tokens, unless the generation encounters the end-of-sequence (EOS) token or the
user explicitly sets a lower length limit for the response.

You may configure both the 'maximum input prompt length' and 'minimum response length' using
the following parameters:

* ``MAX_PROMPT_LEN`` -- defines the maximum number of tokens that the LLM pipeline can process
  for the input prompt (default: 1024),
* ``MIN_RESPONSE_LEN`` -- defines the minimum number of tokens that the LLM pipeline can generate
  in its response (default: 128).

The maximum context size for an LLM on NPU is defined as the sum of these two values. By default,
if the input prompt is shorter than ``MAX_PROMPT_LEN`` tokens, time to first
token (TTFT) remains the same as if a full-length prompt was passed. However, a shorter prompt
allows the model to generate more tokens within the available context. For example, if the input
prompt is just 30 tokens, the model can generate up to :math:`1024 + 128 - 30 = 1122` tokens.

OpenVINO 2025.3 has introduced dynamic input prompt support for NPU. The dynamism granularity is
controlled by the new property ``NPUW_LLM_PREFILL_CHUNK_SIZE`` (default: 1024).

If the ``MAX_PROMPT_LEN`` property is set to a value greater than the chunk size, the mechanism
is activated automatically. Set ``PREFILL_HINT`` to ``STATIC`` to disable this feature.

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

In the GenAI LLM chat scenarios, the conversation history is accumulated in the context and may require
a larger ``MAX_PROMPT_LEN`` to handle the history properly.

Performance hints
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can configure the NPU LLM pipeline using the ``PREFILL_HINT`` and ``GENERATE_HINT`` options
to fine-tune performance. These options impact prompt processing (first token)
and text generation (subsequent tokens) behavior, respectively.

``PREFILL_HINT`` -- fine-tunes the prompt processing stage:

* ``DYNAMIC`` (default since OpenVINO 2025.3) -- enables dynamic prompt execution, supports longer prompts.
* ``STATIC`` -- disables dynamic prompt execution, may provide better performance for specific prompt sizes.
  Default behavior before OpenVINO 2025.3.

``GENERATE_HINT`` -- fine-tunes the text generation stage:

* ``FAST_COMPILE`` (default) -- enables fast compilation at the expense of performance,
* ``BEST_PERF`` -- ensures the best possible performance at lower compilation speed.

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


Caching and ahead-of-time compilation
***********************************************************************************************

LLM compilation for NPU happens on-the-fly and may take substantial time. To
improve user experience, the following options are available: OpenVINO Caching and Ahead-of-time
(AoT) compilation.

OpenVINO Caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

By caching compiled models, you can reduce the initialization time for subsequent pipeline
runs. To do so, specify one of the following options in ``pipeline_config`` for the NPU pipeline.

CACHE_DIR
-----------------------------------------------------------------------------------------------

``CACHE_DIR`` is the default OpenVINO caching mechanism. The  ``CACHE_MODE``
hint defines how the cached blob stores weights. ``OPTIMIZE_SPEED`` includes the weights
and allows faster loading for group-quantized models. ``OPTIMIZE_SIZE`` excludes the weights,
producing a weightless blob, and requires the original model to be present on disk.

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

NPUW_CACHE_DIR
-----------------------------------------------------------------------------------------------

``NPUW_CACHE_DIR`` is a legacy NPU-specific weightless caching option. Since OpenVINO 2025.1,
the preferred device-neutral caching mechanism is the OpenVINO caching (``CACHE_DIR``).


Ahead-of-time compilation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Specifying ``EXPORT_BLOB`` and ``BLOB_PATH`` parameters works similarly to ``CACHE_DIR`` but:

* It allows to explicitly specify where to **store** the compiled model.
* For subsequent runs, it requires the same ``BLOB_PATH`` to **import** the compiled model.
* Blob type is also defined by ``CACHE_MODE``.

  * By default, ``OPTIMIZE_SIZE`` is used,  producing a weightless blob. To load this blob, either the original weights file or an ``ov::Model`` object is required.
  * Pass ``OPTIMIZE_SPEED`` to export a blob with full weights.

* If the blob is exported as weightless you also need to either provide
  ``"WEIGHTS_PATH" : "path\\to\\original\\model.bin"`` or ``"MODEL_PTR" : original ov::Model object`` in the config.
* Ahead-of-time import in weightless mode has been optimized to consume less memory than during regular compilation or using ``CACHE_DIR``.

The following snippets demonstrate the functionality:

.. tab-set::

   .. tab-item:: Weightless export

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

   .. tab-item:: Weightless import

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

   .. tab-item:: Full-weight export

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

   .. tab-item:: Full-weight import

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

When exporting NPU LLM blobs, you can also specify encryption and decryption functions for the blob.
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



VLM Inference on NPU
###############################################################################################
VLMs are supported on NPU and can be inferenced in the same way as LLms with GenAI API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import numpy as np
         from PIL import Image
         from openvino import Tensor
         import openvino_genai as ov_genai

         model_path = "Google-Gemma-3-4B-it"
         image_path = "cat.png"
         image = Tensor(np.array(Image.open(image_path).convert("RGB")))

         pipe = ov_genai.VLMPipeline(model_path, "NPU")
         print(pipe.generate("Describe the image",  images=image, max_new_tokens=100))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "load_image.hpp"
         #include <openvino/genai/visual_language/pipeline.hpp>
         #include <iostream>

         bool print_subword(std::string&& subword) {
            return !(std::cout << subword << std::flush);
         }

         int main(int argc, char* argv[]) {
            std::string model_path = "Google-Gemma-3-4B-it";
            std::string image_path = "cat.png";

            std::vector<ov::Tensor> rgbs = utils::load_images(image_path);

            ov::genai::VLMPipeline pipe(model_path, "NPU");
            ov::genai::GenerationConfig config;
            config.max_new_tokens=100;
            std::cout << pipe.generate("Describe the image",
               ov::genai::images(rgbs),
               ov::genai::generation_config(config),
               ov::genai::streamer(print_subword));
         }

Passing config to VLMs
***********************************************************************************************
All the parameters described above (like ``MAX_PROMPT_LEN``, ``MIN_RESPONSE_LEN``, ``CACHE_DIR``, etc.) are also applicable to VLMs.
However, these parameters must be provided in a slightly different way. They should be placed in {"DEVICE_PROPERTIES": {"NPU" : ... } } section of config:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import numpy as np
         from PIL import Image
         from openvino import Tensor
         import openvino_genai as ov_genai

         model_path = "Phi-4-multimodal-instruct"
         image_path = "cat.png"
         image = Tensor(np.array(Image.open(image_path).convert("RGB")))
         pipeline_config = {
            "DEVICE_PROPERTIES": {
               "NPU": {
                  "MAX_PROMPT_LEN": 2048,
                  "MIN_RESPONSE_LEN": 512
               },
            }
         }

         pipe = ov_genai.VLMPipeline(model_path, "NPU", config=pipeline_config)
         print(pipe.generate("Describe the image",  images=image, max_new_tokens=100))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "load_image.hpp"
         #include <openvino/genai/visual_language/pipeline.hpp>
         #include <iostream>

         bool print_subword(std::string&& subword) {
            return !(std::cout << subword << std::flush);
         }

         int main(int argc, char* argv[]) {
            std::string model_path = "Phi-4-multimodal-instruct";
            std::string image_path = "cat.png";

            std::vector<ov::Tensor> rgbs = utils::load_images(image_path);
            ov::AnyMap pipeline_config = {
               {"DEVICE_PROPERTIES", ov::AnyMap{
                  {"NPU", ov::AnyMap{
                     {"MAX_PROMPT_LEN", 2048},
                     {"MIN_RESPONSE_LEN", 512}
                  }}
               }}
            };

            ov::genai::VLMPipeline pipe(model_path, "NPU", pipeline_config);
            ov::genai::GenerationConfig config;
            config.max_new_tokens=100;
            std::cout << pipe.generate("Describe the image",
               ov::genai::images(rgbs),
               ov::genai::generation_config(config),
               ov::genai::streamer(print_subword));
         }



Whisper Inference on NPU
###############################################################################################

OpenAI Whisper support (for
`whisper-tiny <https://huggingface.co/openai/whisper-tiny>`__,
`whisper-base <https://huggingface.co/openai/whisper-base>`__,
`whisper-small <https://huggingface.co/openai/whisper-small>`__, or
`whisper-large <https://huggingface.co/openai/whisper-large>`__ models) was first introduced in
OpenVINO 2024.5. There are no NPU-specific requirements when running the Whisper GenAI pipeline on NPU,
so a standard OpenVINO GenAI sample works without any limitations.


Export Whisper models from Hugging Face
***********************************************************************************************

Prior to OpenVINO 2025.1 the Whisper pipeline
only accepted stateless Whisper models, exported with ``--disable-stateful`` flag:

.. code:: console

   optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny --disable-stateful

Since OpenVINO 2025.1, this is no longer required. Weights can remain in FP16 or be compressed in INT8:

.. code:: console

   optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base-int8 --weight-format int8

Troubleshooting
###############################################################################################

Disabling L0 memory allocation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In case of execution failures, either silent or with errors, try to update the NPU driver to
`32.0.100.3104 or newer <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.
If the update is not possible and you get "out of memory" errors, try setting the
``DISABLE_OPENVINO_GENAI_NPU_L0`` environment variable to disable Level0 memory allocation.

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



Additional Resources
###############################################################################################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `Optimum Intel <https://github.com/huggingface/optimum-intel>`__
