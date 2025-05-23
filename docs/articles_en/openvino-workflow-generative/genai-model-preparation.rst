Generative Model Preparation
===============================================================================

.. meta::
   :description: Learn how to use Hugging Face Hub and Optimum Intel APIs to
                 prepare generative models for inference.



Since generative AI models tend to be big and resource-heavy, it is advisable to
optimize them for efficient inference. This article will show how to prepare
LLM models for inference with OpenVINO by:

* `Downloading Models from Hugging Face <#download-generative-models-from-hugging-face-hub>`__
* `Downloading Models from Model Scope <#download-generative-models-from-model-scope>`__
* `Converting and Optimizing Generative Models <#convert-and-optimize-generative-models>`__



Download Generative Models From Hugging Face Hub
###############################################################################

Pre-converted and pre-optimized models are available in the `OpenVINO Toolkit <https://huggingface.co/OpenVINO>`__
organization, under the `model section <https://huggingface.co/OpenVINO#models>`__, or under
different model collections:

* `LLM: <https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd>`__
* `Speech-to-Text <https://huggingface.co/collections/OpenVINO/speech-to-text-672321d5c070537a178a8aeb>`__
* `Speculative Decoding Draft Models <https://huggingface.co/collections/OpenVINO/speculative-decoding-draft-models-673f5d944d58b29ba6e94161>`__

You can also use the **huggingface_hub** package to download models:

.. code-block:: console

   pip install huggingface_hub
   huggingface-cli download "OpenVINO/phi-2-fp16-ov" --local-dir model_path


The models can be used in OpenVINO immediately after download. No dependencies
are required except **huggingface_hub**.


Download Generative Models From Model Scope
###############################################################################

To download models from `Model Scope <https://www.modelscope.cn/home>`__,
use the **modelscope** package:

.. code-block:: console

   pip install modelscope
   modelscope download --model "Qwen/Qwen2-7b" --local_dir model_path

Models downloaded via Model Scope are available in Pytorch format only and they must
be :doc:`converted to OpenVINO IR <../../openvino-workflow/model-preparation/convert-model-to-ir>`
before inference.

Convert and Optimize Generative Models
###############################################################################

OpenVINO works best with models in the OpenVINO IR format, both in full precision and quantized.
If your selected model has not been pre-optimized, you can easily do it yourself, using a single
**optimum-cli** command. For that, make sure optimum-intel is installed on your system:

.. code-block:: console

   pip install optimum-intel[openvino]


While optimizing models, you can decide to keep the original precision or select one that is lower.

.. tab-set::

   .. tab-item:: Keeping full model precision
      :sync: full-precision

      .. code-block:: console

         optimum-cli export openvino --model <model_id> --weight-format fp16 <exported_model_name>

      Examples:

      .. tab-set::

         .. tab-item:: LLM (text generation)
            :sync: llm-text-gen

            .. code-block:: console

               optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format fp16 ov_llama_2

         .. tab-item:: Diffusion models (text2image)
            :sync: diff-text-img

            .. code-block:: console

               optimum-cli export openvino --model stabilityai/stable-diffusion-xl-base-1.0 --weight-format fp16 ov_SDXL

         .. tab-item:: VLM (Image processing):
            :sync: vlm-img-proc

            .. code-block:: console

               optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code –weight-format fp16 ov_MiniCPM-V-2_6

         .. tab-item:: Whisper models (speech2text):
            :sync: whisp-speech-txt

            .. code-block:: console

               optimum-cli export openvino --trust-remote-code --model openai/whisper-base ov_whisper

   .. tab-item:: Exporting to selected precision
      :sync: low-precision

      .. code-block:: console

         optimum-cli export openvino --model <model_id> --weight-format int4 <exported_model_name>

      Examples:

      .. tab-set::

         .. tab-item:: LLM (text generation)
            :sync: llm-text-gen

            .. code-block:: console

               optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format int4 ov_llama_2

         .. tab-item:: Diffusion models (text2image)
            :sync: diff-text-img

            .. code-block:: console

               optimum-cli export openvino --model stabilityai/stable-diffusion-xl-base-1.0 --weight-format int4 ov_SDXL

         .. tab-item:: VLM (Image processing)
            :sync: vlm-img-proc

            .. code-block:: console

               optimum-cli export openvino -m model_path --task text-generation-with-past --weight-format int4 ov_MiniCPM-V-2_6


.. note::

   Any other ``model_id``, for example ``openbmb/MiniCPM-V-2_6``, or the path
   to a local model file can be used.

   Also, you can specify different data type like ``int8``.


Additional Resources
###############################################################################

* `Full set of optimum-cli parameters <https://huggingface.co/docs/optimum/en/intel/openvino/export>`__
* :doc:`Model conversion in OpenVINO <../../openvino-workflow/model-preparation/convert-model-to-ir>`
* :doc:`Model optimization in OpenVINO <../../openvino-workflow/model-optimization>`
