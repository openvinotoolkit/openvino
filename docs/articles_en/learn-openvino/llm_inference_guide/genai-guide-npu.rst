Run LLMs with OpenVINO GenAI Flavor on NPU
==========================================

.. meta::
   :description: Learn how to use the OpenVINO GenAI flavor to execute LLM models on NPU.

This guide provide some extra details on how to utilize NPU with GenAI
flavor. Please refer to
:doc:`install OpenVINO GenAI <../../get-started/install-openvino/install-openvino-genai>`
for more information on OpenVINO GenAI.

1. Export an LLM model via Hugging Face Optimum-Intel. First, make sure the component versions are correct:

   .. code-block::

      transformers==4.42.4
      openvino==2024.2.0
      openvino-tokenizers==2024.2.0
      nncf==2.11.0
      optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@439d61f79cf55d5d0b28334f577b6ac3c5ced28f

   A chat-tuned TinyLlama model is used in this example. The following conversion & optimization settings are recommended when using NPU:

   .. code-block:: python

      optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4_sym_g128 --ratio 1.0 TinyLlama

2. Perform generation using the new GenAI API:

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

Compiling models for NPU may take a while. By default, LLMPipeline for NPU is configured for faster compilation, but it may result in lower performance. To achieve better performance at the expense of compilation time, you may try these settings:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         plugin_config = { "PREFILL_CONFIG": { "USE_NPUW": "NO" }, "GENERATE_CONFIG": { "USE_NPUW": "NO" } }
         pipe = ov_genai.LLMPipeline(model_path, "NPU", plugin_config)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         std::map<std::string, std::string> cfg = { { "NPU_USE_NPUW", "NO" } };
         ov::AnyMap plugin_config = { { "PREFILL_CONFIG", cfg }, { "GENERATE_CONFIG", cfg } };
         ov::genai::LLMPipeline pipe(model_path, "NPU", plugin_config);


Additional Resources
####################

...
