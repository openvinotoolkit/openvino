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

A chat-tuned TinyLlama model is used in this example. The following conversion & optimization settings are recommended when using the NPU:

.. code-block:: python

   optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --group-size 128 --ratio 1.0 TinyLlama

Run generation using OpenVINO GenAI
###################################

Use the following code snippet to perform generation with OpenVINO GenAI API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai as ov_genai
         pipe = ov_genai.LLMPipeline(model_path, "NPU")
         print(pipe.generate("The Sun is yellow because", max_new_tokens=100))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "openvino/genai/llm_pipeline.hpp"
         #include <iostream>

         int main(int argc, char* argv[]) {
            std::string model_path = argv[1];
            ov::genai::LLMPipeline pipe(model_path, "NPU");
            std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100));
         }

Additional configuration options
################################

Compiling models for NPU may take a while. By default, the LLMPipeline for the NPU
is configured for faster compilation, but it may result in lower performance.
To achieve better performance at the expense of compilation time, you may try these settings:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         plugin_config = { "NPU_COMPILATION_MODE_PARAMS": "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm" }
         pipeline_config = { "PREFILL_CONFIG": plugin_config, "GENERATE_CONFIG": plugin_config }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", pipeline_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov::AnyMap plugin_config = { { "NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm" } };
         ov::AnyMap pipeline_config = { { "PREFILL_CONFIG",  plugin_config }, { "GENERATE_CONFIG", plugin_config } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", pipeline_config);


Additional Resources
####################

* :doc:`NPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/npu-device>`
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
