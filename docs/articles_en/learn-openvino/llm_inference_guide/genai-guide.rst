
OpenVINO GenAI Guide
===============================

This guide will walk through the essential steps for integrating the OpenVINO GenAI API into your application.
The steps below show the initial setup, demonstrate how to load a model,
and illustrate the process of passing the input context to receive generated text.

1.	Export an LLM model via Hugging Face Optimum-Intel. A chat-tuned TinyLlama is used for this example:

.. code-block:: python

   optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code

Optional. Optimize the model:

This model will be in the form of an optimized OpenVINO IR of the fp16 precision.
To make LLM inference more performant we recommend using a lower precision for model weights,
i.e. int4, and compress weights using NNCF during model export directly:

.. code-block:: python

   optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code

2. Perform generation using the new GenAI API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai as ov_genai
         pipe = ov_genai.LLMPipeline(model_path, "CPU")
         print(pipe.generate("The Sun is yellow bacause"))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "openvino/genai/llm_pipeline.hpp"
         #include <iostream>

         int main(int argc, char* argv[]) {
         std::string model_path = argv[1];
         ov::genai::LLMPipeline pipe(model_path, "CPU");//target device is CPU
         std::cout << pipe.generate("The Sun is yellow bacause"); //input context

+OUTPUT

Once the model is exported from Hugging Face Optimum-Intel, it already contains all the necessary
information for execution, including the tokenizer/detokenizer and the generation config
ensuring that its results match Hugging Face generation.





Additional Resources
####################

* `Text generation C++ samples that support most popular models like LLaMA 2 <https://github.com/openvinotoolkit/openvino.genai/tree/master/text_generation/causal_lm/cpp>`__
* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* :doc:`Stateful Models Low-Level Details <../../openvino-workflow/running-inference/stateful-models>`
* :doc:`Working with Textual Data <../../openvino-workflow/running-inference/string-tensors>`


