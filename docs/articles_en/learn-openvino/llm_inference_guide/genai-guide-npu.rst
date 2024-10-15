Run LLMs with OpenVINO GenAI Flavor on NPU
==========================================

.. meta::
   :description: Learn how to use the OpenVINO GenAI flavor to execute LLM models on NPU.

This guide will give you extra details on how to utilize NPU with the GenAI flavor.
:doc:`See the installation guide <../../get-started/install-openvino/install-openvino-genai>`
for information on how to start.

Prerequisites
#############

Install required dependencies:

.. code-block:: console

   python -m venv npu-env
   npu-env\Scripts\activate
   pip install optimum-intel nncf==2.11 onnx==1.16.1
   pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

Export an LLM model via Hugging Face Optimum-Intel
##################################################

A chat-tuned TinyLlama model is used in this example. The following conversion & optimization
settings are recommended when using the NPU:

.. code-block:: python

   optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --group-size 128 --ratio 1.0 TinyLlama

**For models exceeding 1 billion parameters**, it is recommended to use **channel-wise
quantization** that is remarkably effective. For example, you can try the approach with the
llama-2-7b-chat-hf model:

.. code-block:: python

   optimum-cli export openvino -m meta-llama/Llama-2-7b-chat-hf --weight-format int4 --sym --group-size -1 --ratio 1.0 Llama-2-7b-chat-hf


Run generation using OpenVINO GenAI
###################################

It is recommended to install the latest available
`driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__.

Use the following code snippet to perform generation with OpenVINO GenAI API:

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
            ov::genai::LLMPipeline pipe(model_path, "NPU");
            std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100));
         }

Additional configuration options
################################

Prompt and response length options
++++++++++++++++++++++++++++++++++

The LLM pipeline for NPUs leverages the static shape approach, optimizing execution performance,
while potentially introducing certain usage limitations. By default, the LLM pipeline supports
input prompts up to 1024 tokens in length. It also ensures that the generated response contains
at least 150 tokens, unless the generation encounters the end-of-sequence (EOS) token or the
user explicitly sets a lower length limit for the response.

You may configure both the 'maximum input prompt length' and 'minimum response length' using
the following parameters:

* ``MAX_PROMPT_LEN``: Defines the maximum number of tokens that the LLM pipeline can process
  for the input prompt (default: 1024).
* ``MIN_RESPONSE_LEN``: Defines the minimum number of tokens that the LLM pipeline will generate
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


Additional Resources
####################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__