
OpenVINO GenAI API Guide
===============================

This guide provide the instructions for integrating the OpenVINO GenAI API into your application.
The steps below demonstrate how to load a model and pass the input context to receive generated text.

The examples use a CPU as the target device, however, the GPU support is also available.
Note that the GPU is running only LLM inference, while token selection and tokenization/detokenization remain on the CPU for efficiency.
Tokenizers are represented as a separate model and run on the CPU using the provided inference capabilities.

Before proceeding, make sure that you have installed the OpenVINO GenAI API using :doc:`PyPI or Archive <../../get-started/install-openvino>` distributions.

1.	Export an LLM model via Hugging Face Optimum-Intel. A chat-tuned TinyLlama model is used for this example:

.. code-block:: python

   optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format fp16 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"

*Optional*. Optimize the model:

The model is an optimized OpenVINO IR with fp16 precision. For enhanced LLM performance,
it is recommended to use lower precision for model weights, such as int4, and to compress weights
using NNCF during model export directly:

.. code-block:: python

   optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code

2. Perform generation using the new GenAI API:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai as ov_genai
         pipe = ov_genai.LLMPipeline(model_path, "CPU")
         print(pipe.generate("The Sun is yellow because"))

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         #include "openvino/genai/llm_pipeline.hpp"
         #include <iostream>

         int main(int argc, char* argv[]) {
         std::string model_path = argv[1];
         ov::genai::LLMPipeline pipe(model_path, "CPU");//target device is CPU
         std::cout << pipe.generate("The Sun is yellow because"); //input context

Once the model is exported from Hugging Face Optimum-Intel, it already contains all the necessary
information for execution, including the tokenizer/detokenizer and the generation config
ensuring that its results match Hugging Face generation.

Streaming Options
###########################

For more interactive UIs during generation, streaming of model output tokens is supported. See the example below, where a lambda function outputs words to the console immediately upon generation:

.. tab-set::

   .. tab-item:: C++

      #include "openvino/genai/llm_pipeline.hpp"
      #include <iostream>

      int main(int argc, char* argv[]) {
         std::string model_path = argv[1];
         ov::genai::LLMPipeline pipe(model_path, "CPU");

         auto streamer = [](std::string word) { std::cout << word << std::flush; };
         std::cout << pipe.generate("The Sun is yellow because", streamer);
      }

You can also create your custom streamer for more sophisticated processing:

.. tab-set::

   .. tab-item:: C++

      #include <streamer_base.hpp>

      class CustomStreamer: publict StreamerBase {
      public:
         void put(int64_t token) {/* decode tokens and do process them*/};

         void end() {/* decode tokens and do process them*/};
      };

      int main(int argc, char* argv[]) {
         CustomStreamer custom_streamer;

         std::string model_path = argv[1];
         ov::LLMPipeline pipe(model_path, "CPU");
         cout << pipe.generate("The Sun is yellow bacause", custom_streamer);
      }

Chat Scenarios Optimization
##############################

For chat scenarios where inputs and outputs represent a conversation, maintaining KVCache across inputs
offers optimization benefits. The chat-specific methods **start_chat** and **finish_chat** are used to
mark a conversation session. Simplified Python and C++ examples are provided below:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         import openvino_genai as ov_genai
         pipe = ov_genai.LLMPipeline(model_path)

         config = {'num_groups': 3, 'group_size': 5, 'diversity_penalty': 1.1}
         pipe.set_generation_cofnig(config)

         pipe.start_chat()
         while True:
             print('question:')
             prompt = input()
            if prompt == 'Stop!':
                 break
             print(pipe(prompt))
         pipe.finish_chat()


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         int main(int argc, char* argv[]) {
            std::string prompt;

            std::string model_path = argv[1];
            ov::LLMPipeline pipe(model_path, "CPU");

            pipe.start_chat();
            for (size_t i = 0; i < questions.size(); i++) {
               std::cout << "question:\n";
               std::getline(std::cin, prompt);

               std::cout << pipe(prompt) << std::endl>>;
            }
            pipe.finish_chat();
         }

Optimizing Text Generation with Group Beam Search
#######################################################

Leverage group beam search decoding and configure generation_config for better text generation quality and efficient batch processing in GenAI applications.

Use group beam search decoding:

.. tab-set::

   .. tab-item:: C++

      int main(int argc, char* argv[]) {
         std::string model_path = argv[1];
         ov::LLMPipeline pipe(model_path, "CPU");
         ov::GenerationConfig config = pipe.get_generation_config();
         config.max_new_tokens = 256;
         config.num_groups = 3;
         config.group_size = 5;
         config.diversity_penalty = 1.0f;

         cout << pipe.generate("The Sun is yellow bacause", config);
      }

Specify generation_config to use grouped beam search:

.. tab-set::

   .. tab-item:: C++

      int main(int argc, char* argv[]) {
         std::string prompt;

         std::string model_path = argv[1];
         ov::LLMPipeline pipe(model_path, "CPU");

         ov::GenerationConfig config = pipe.get_generation_config();
         config.max_new_tokens = 256;
         config.num_groups = 3;
         config.group_size = 5;
         config.diversity_penalty = 1.0f;

         auto streamer = [](std::string word) { std::cout << word << std::flush; };

         pipe.start_chat();
         for (size_t i = 0; i < questions.size(); i++) {

            std::cout << "question:\n";
            cout << prompt << endl;

            auto answer = pipe(prompt, config, streamer);
            // no need to print answer, streamer will do that
         }
         pipe.finish_chat();
      }

Comparing with Hugging Face Results
#######################################

Compare and analyze results with those generated by Hugging Face models.

.. tab-set::

   .. tab-item:: Python

      from transformers import AutoTokenizer, AutoModelForCausalLM

      tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

      max_new_tokens = 32
      prompt = 'table is made of'

      encoded_prompt = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
      hf_encoded_output = model.generate(encoded_prompt, max_new_tokens=max_new_tokens, do_sample=False)
      hf_output = tokenizer.decode(hf_encoded_output[0, encoded_prompt.shape[1]:])
      print(f'hf_output: {hf_output}')

      import sys
      sys.path.append('build-Debug/')
      import py_generate_pipeline as genai # set more friendly module name

      pipe = genai.LLMPipeline('text_generation/causal_lm/TinyLlama-1.1B-Chat-v1.0/pytorch/dldt/FP16/')
      ov_output = pipe(prompt, max_new_tokens=max_new_tokens)
      print(f'ov_output: {ov_output}')

      assert hf_output == ov_output



Additional Resources
####################

* `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__



