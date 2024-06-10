Install OpenVINO™ GenAI
====================================

OpenVINO GenAI is a new flavor of OpenVINO, aiming to simplify running inference of generative AI models.
It hides the complexity of the generation process and minimizes the amount of code required.
You can now provide a model and input context directly to OpenVINO, which performs tokenization of the
input text, executes the generation loop on the selected device, and returns the generated text.
For a quickstart guide, refer to the :doc:`GenAI API Guide <../../learn-openvino/llm_inference_guide/genai-guide>`.

To see GenAI in action, check the Jupyter notebooks:
`LLM-powered Chatbot <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/README.md>`__ and
`LLM Instruction-following pipeline <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-question-answering/README.md>`__

The OpenVINO GenAI flavor is available for installation via Archive and PyPI distributions:

Archive Installation
###############################

To install the GenAI flavor of OpenVINO from an archive file, follow the standard installation steps for your system
but instead of using the vanilla package file, download the one with OpenVINO GenAI:

.. tab-set::

   .. tab-item:: x86_64
      :sync: x86-64

      .. tab-set::

         .. tab-item:: Ubuntu 24.04
            :sync: ubuntu-24

            .. code-block:: sh


               curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/linux/l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_2024.1.0.tgz
               tar -xf openvino_2024.1.0.tgz
               sudo mv l_openvino_toolkit_ubuntu24_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_2024.1.0

         .. tab-item:: Ubuntu 22.04
            :sync: ubuntu-22

            .. code-block:: sh

               curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/linux/l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64.tgz --output openvino_2024.1.0.tgz
               tar -xf openvino_2024.1.0.tgz
               sudo mv l_openvino_toolkit_ubuntu22_2024.1.0.15008.f4afc983258_x86_64 /opt/intel/openvino_2024.1.0

Here are the full guides:
:doc:`Linux <install-openvino-archive-linux>`,
:doc:`Windows <install-openvino-archive-windows>`, and
:doc:`macOS <install-openvino-archive-macos>`.

If OpenVINO GenAI is installed via archive distribution or built from source, you will need to install additional python dependencies (for example, `optimum-cli` for simplified model downloading and exporting):

.. code-block:: sh

   # (Optional) Clone OpenVINO GenAI repository
   git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
   cd openvino.genai
   # Install python dependencies
   python -m pip install ./thirdparty/openvino_tokenizers/[transformers] --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
   python -m pip install --upgrade-strategy eager -r ./samples/cpp/requirements.txt


PyPI Installation
###############################

To install the GenAI flavor of OpenVINO via PyPI, follow the standard :doc:`installation steps <install-openvino-pip>`,
but use the *openvino-genai* package instead of *openvino*:

.. code-block:: python

   python -m pip install openvino-genai


Supported Models
#######################################

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Architecture
     - Models
     - Example Hugging Face Models
   * - ``ChatGLMModel``
     - ChatGLM
     - `THUDM/chatglm2-6b <https://huggingface.co/THUDM/chatglm2-6b>`__
       `THUDM/chatglm3-6b <https://huggingface.co/THUDM/chatglm3-6b>`__
   * - ``GemmaForCausalLM``
     - Gemma
     - `google/gemma-2b-it <https://huggingface.co/google/gemma-2b-it>`__
   * - ``GPTNeoXForCausalLM``
     - Dolly
       RedPajama
     - `databricks/dolly-v2-3b <https://huggingface.co/databricks/dolly-v2-3b>`__
       `ikala/redpajama-3b-chat <https://huggingface.co/ikala/redpajama-3b-chat>`__
   * - ``LlamaForCausalLM``
     - Llama 2
       OpenLLaMA
       TinyLlama
     - `meta-llama/Llama-2-13b-chat-hf <https://huggingface.co/meta-llama/Llama-2-13b-chat-hf>`__
       `meta-llama/Llama-2-13b-hf <https://huggingface.co/meta-llama/Llama-2-13b-hf>`__
       `meta-llama/Llama-2-7b-chat-hf <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__
       `meta-llama/Llama-2-7b-hf <https://huggingface.co/meta-llama/Llama-2-7b-hf>`__
       `meta-llama/Llama-2-70b-chat-hf <https://huggingface.co/meta-llama/Llama-2-70b-chat-hf>`__
       `meta-llama/Llama-2-70b-hf <https://huggingface.co/meta-llama/Llama-2-70b-hf>`__
       `microsoft/Llama2-7b-WhoIsHarryPotter <https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter>`__
       `openlm-research/open_llama_13b <https://huggingface.co/openlm-research/open_llama_13b>`__
       `openlm-research/open_llama_3b <https://huggingface.co/openlm-research/open_llama_3b>`__
       `openlm-research/open_llama_3b_v2 <https://huggingface.co/openlm-research/open_llama_3b_v2>`__
       `openlm-research/open_llama_7b <https://huggingface.co/openlm-research/open_llama_7b>`__
       `openlm-research/open_llama_7b_v2 <https://huggingface.co/openlm-research/open_llama_7b_v2>`__
       `TinyLlama/TinyLlama-1.1B-Chat-v1.0 <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
   * - ``MistralForCausalLM``
     - Mistral
       Notus
       Zephyr
     - `mistralai/Mistral-7B-v0.1 <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__
       `argilla/notus-7b-v1 <https://huggingface.co/argilla/notus-7b-v1>`__
       `HuggingFaceH4/zephyr-7b-beta <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__
   * - ``PhiForCausalLM``
     - Phi
     - `microsoft/phi-2 <https://huggingface.co/microsoft/phi-2>`__
       `microsoft/phi-1_5 <https://huggingface.co/microsoft/phi-1_5>`__
   * - ``QWenLMHeadModel``
     - Qwen
     - `Qwen/Qwen-7B-Chat <https://huggingface.co/Qwen/Qwen-7B-Chat>`__
       `Qwen/Qwen-7B-Chat-Int4 <https://huggingface.co/Qwen/Qwen-7B-Chat-Int4>`__
       `Qwen/Qwen1.5-7B-Chat <https://huggingface.co/Qwen/Qwen1.5-7B-Chat>`__
       `Qwen/Qwen1.5-7B-Chat-GPTQ-Int4 <https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4>`__

The pipeline can work with other similar topologies produced by Optimum Intel with the same model
signature. After conversion, the model must have the following inputs:

* ``input_ids`` contains the tokens.
* ``attention_mask`` is populated with 1s.
* ``beam_idx`` is used to select beams.
* ``position_ids`` (optional) encodes the position of the currently generating token in the sequence
and a single ``logits`` output.

.. note::

   Models should belong to the same family and have the same tokenizers.
   Some models may require access request submission on the Hugging Face page to be downloaded.



