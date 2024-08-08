Run LLMs with OpenVINO GenAI Flavor on NPU
==========================================

.. meta::
   :description: Learn how to use the OpenVINO GenAI flavor to execute LLM models on NPU.

This guide provide some extra details on how to utilize NPU with GenAI
flavor. Please refer to
:doc:`install OpenVINO GenAI <../../get-started/install-openvino/install-openvino-genai>`
for more information on OpenVINO GenAI.

Export an LLM model via Hugging Face Optimum-Intel
##################################################

1. Create python virtual environment and install the correct components for exporting model:

   .. code-block:: text

      # requirements.txt
      transformers>=4.42.4
      openvino==2024.2.0
      openvino-tokenizers==2024.2.0
      nncf==2.11.0
      optimum-intel @ git+https://github.com/huggingface/optimum-intel.git

   .. code-block:: console

      python -m venv export-npu-env
      export-npu-env\Scripts\activate
      pip install -r requirements.txt

2. A chat-tuned TinyLlama model is used in this example. The following conversion & optimization settings are recommended when using NPU:

   .. code-block:: python

      optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --group-size 128 --ratio 1.0 TinyLlama

Run the generation by using OpenVINO GenAI
##########################################

1. Create python virtual environment and install the correct components for running the model on NPU via OpenVINO GenAI:

   .. code-block:: text

      # requirements.txt
      openvino>=2024.3.1
      openvino-tokenizers>=2024.3.1
      openvino-genai>=2024.3.1
      transformers>=4.42.4
      torch>=2.4.0+cpu
      diffusers>=0.29.2
      optimum-intel @ git+https://github.com/huggingface/optimum-intel.git

   .. code-block:: console

      python -m venv run-npu-env
      run-npu-env\Scripts\activate
      pip install -r requirements.txt

2. Perform generation using the new GenAI API

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

Compiling models for NPU may take a while. By default, LLMPipeline for NPU
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
