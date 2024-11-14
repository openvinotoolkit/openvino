Create an LLM-powered Chatbot using OpenVINO Generate API
=========================================================

In the rapidly evolving world of artificial intelligence (AI), chatbots
have emerged as powerful tools for businesses to enhance customer
interactions and streamline operations. Large Language Models (LLMs) are
artificial intelligence systems that can understand and generate human
language. They use deep learning algorithms and massive amounts of data
to learn the nuances of language and produce coherent and relevant
responses. While a decent intent-based chatbot can answer basic,
one-touch inquiries like order management, FAQs, and policy questions,
LLM chatbots can tackle more complex, multi-touch questions. LLM enables
chatbots to provide support in a conversational manner, similar to how
humans do, through contextual memory. Leveraging the capabilities of
Language Models, chatbots are becoming increasingly intelligent, capable
of understanding and responding to human language with remarkable
accuracy.

Previously, we already discussed how to build an instruction-following
pipeline using OpenVINO, please check out `this
tutorial <llm-question-answering-with-output.html>`__ for
reference. In this tutorial, we consider how to use the power of
OpenVINO for running Large Language Models for chat. We will use a
pre-trained model from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. The `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format. To simplify the user
experience, we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai>`__ for
generation pipeline.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to 4-bit or 8-bit data types using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a chat inference pipeline with `OpenVINO Generate
   API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
-  Run chat pipeline with `Gradio <https://www.gradio.app/>`__.

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Convert model using Optimum-CLI
   tool <#convert-model-using-optimum-cli-tool>`__

   -  `Weights Compression using
      Optimum-CLI <#weights-compression-using-optimum-cli>`__

-  `Select device for inference <#select-device-for-inference>`__
-  `Instantiate pipeline with OpenVINO Generate
   API <#instantiate-pipeline-with-openvino-generate-api>`__
-  `Run Chatbot <#run-chatbot>`__

   -  `Advanced generation options <#advanced-generation-options>`__

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    import os

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    %pip install -Uq pip
    %pip uninstall -q -y optimum optimum-intel
    %pip install -q -U "openvino>=2024.3.0" openvino-tokenizers[transformers] openvino-genai
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "git+https://github.com/openvinotoolkit/nncf.git"\
    "torch>=2.1"\
    "datasets" \
    "accelerate" \
    "gradio>=4.19" \
    "transformers>=4.43.1" \
    "onnx<=1.16.1; sys_platform=='win32'" "einops" "transformers_stream_generator" "tiktoken" "bitsandbytes"

.. code:: ipython3

    import os
    from pathlib import Path
    import requests
    import shutil

    # fetch model configuration

    config_shared_path = Path("../../utils/llm_config.py")
    config_dst_path = Path("llm_config.py")

    if not config_dst_path.exists():
        if config_shared_path.exists():
            try:
                os.symlink(config_shared_path, config_dst_path)
            except Exception:
                shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)
    elif not os.path.islink(config_dst_path):
        print("LLM config will be updated")
        if config_shared_path.exists():
            shutil.copy(config_shared_path, config_dst_path)
        else:
            r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
            with open("llm_config.py", "w", encoding="utf-8") as f:
                f.write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.
Model conversion and optimization is time- and memory-consuming process.
For your convenience, we provide a
`collection <https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd>`__
of optimized models on HuggingFace hub. You can skip the model
conversion step by selecting one of the available on HuggingFace hub
model. If you want to reproduce optimization process locally, please
unset **Use preconverted models** checkbox.

   **Note**: conversion of some models can require additional actions
   from user side and at least 64GB RAM for conversion.

`Weight
compression <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
is a technique for enhancing the efficiency of models, especially those
with large memory requirements. This method reduces the model’s memory
footprint, a crucial factor for Large Language Models (LLMs). We provide
several options for model weight compression:

-  **FP16** reducing model binary size on disk using ``save_model`` with
   enabled compression weights to FP16 precision. This approach is
   available in OpenVINO from scratch and is the default behavior.
-  **INT8** is an 8-bit weight-only quantization provided by
   `NNCF <https://github.com/openvinotoolkit/nncf>`__: This method
   compresses weights to an 8-bit integer data type, which balances
   model size reduction and accuracy, making it a versatile option for a
   broad range of applications.
-  **INT4** is an 4-bit weight-only quantization provided by
   `NNCF <https://github.com/openvinotoolkit/nncf>`__. involves
   quantizing weights to an unsigned 4-bit integer symmetrically around
   a fixed zero point of eight (i.e., the midpoint between zero and 15).
   in case of **symmetric quantization** or asymmetrically with a
   non-fixed zero point, in case of **asymmetric quantization**
   respectively. Compared to INT8 compression, INT4 compression improves
   performance even more, but introduces a minor drop in prediction
   quality. INT4 it ideal for situations where speed is prioritized over
   an acceptable trade-off against accuracy.
-  **INT4 AWQ** is an 4-bit activation-aware weight quantization.
   `Activation-aware Weight
   Quantization <https://arxiv.org/abs/2306.00978>`__ (AWQ) is an
   algorithm that tunes model weights for more accurate INT4
   compression. It slightly improves generation quality of compressed
   LLMs, but requires significant additional time for tuning weights on
   a calibration dataset. We will use ``wikitext-2-raw-v1/train`` subset
   of the
   `Wikitext <https://huggingface.co/datasets/Salesforce/wikitext>`__
   dataset for calibration.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to see available models options

.. raw:: html

   </summary>

-  **tiny-llama-1b-chat** - This is the chat model finetuned on top of
   `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>`__.
   The TinyLlama project aims to pretrain a 1.1B Llama model on 3
   trillion tokens with the adoption of the same architecture and
   tokenizer as Llama 2. This means TinyLlama can be plugged and played
   in many open-source projects built upon Llama. Besides, TinyLlama is
   compact with only 1.1B parameters. This compactness allows it to
   cater to a multitude of applications demanding a restricted
   computation and memory footprint. More details about model can be
   found in `model
   card <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`__
-  **mini-cpm-2b-dpo** - MiniCPM is an End-Size LLM developed by
   ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding
   embeddings. After Direct Preference Optimization (DPO) fine-tuning,
   MiniCPM outperforms many popular 7b, 13b and 70b models. More details
   can be found in
   `model_card <https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16>`__.
-  **llama-3.2-1B-instruct** - 1B parameters model from LLama3.2
   collection of instruction-tuned multilingual models. Llama 3.2
   instruction-tuned text only models are optimized for multilingual
   dialogue use cases, including agentic retrieval and summarization
   tasks. They outperform many of the available open source and closed
   chat models on common industry benchmarks. More details can be found
   in `model
   card <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model


       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-3.2-3B-instruct** - 3B parameters model from LLama3.2
   collection of instruction-tuned multilingual models. Llama 3.2
   instruction-tuned text only models are optimized for multilingual
   dialogue use cases, including agentic retrieval and summarization
   tasks. They outperform many of the available open source and closed
   chat models on common industry benchmarks. More details can be found
   in `model
   card <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`__
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model


       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **gemma-2b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 2B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-2b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit
   `HuggingFace model
   card <https://huggingface.co/google/gemma-2b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model


       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **gemma-2-2b-it** - Gemma2 is the second generation of a Gemma family
   of lightweight, state-of-the-art open models from Google, built from
   the same research and technology used to create the Gemini models.
   They are text-to-text, decoder-only large language models, available
   in English, with open weights, pre-trained variants, and
   instruction-tuned variants. Gemma models are well-suited for a
   variety of text generation tasks, including question answering,
   summarization, and reasoning. This model is instruction-tuned version
   of 2B parameters model. More details about model can be found in
   `model card <https://huggingface.co/google/gemma-2-2b-it>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/google/gemma-2-2b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model


       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **phi-3-mini-instruct** - The Phi-3-Mini is a 3.8B parameters,
   lightweight, state-of-the-art open model trained with the Phi-3
   datasets that includes both synthetic data and the filtered publicly
   available websites data with a focus on high-quality and reasoning
   dense properties. More details about model can be found in `model
   card <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__,
   `Microsoft blog <https://aka.ms/phi3blog-april>`__ and `technical
   report <https://aka.ms/phi3-tech-report>`__.
-  **phi-3.5-mini-instruct** - Phi-3.5-mini is a lightweight,
   state-of-the-art open model built upon datasets used for Phi-3 -
   synthetic data and filtered publicly available websites - with a
   focus on very high-quality, reasoning dense data. The model belongs
   to the Phi-3 model family and supports 128K token context length. The
   model underwent a rigorous enhancement process, incorporating both
   supervised fine-tuning, proximal policy optimization, and direct
   preference optimization to ensure precise instruction adherence and
   robust safety measures. More details about model can be found in
   `model
   card <https://huggingface.co/microsoft/Phi-3.5-mini-instruct>`__,
   `Microsoft blog <https://aka.ms/phi3.5-techblog>`__ and `technical
   report <https://arxiv.org/abs/2404.14219>`__.
-  **red-pajama-3b-chat** - A 2.8B parameter pre-trained language model
   based on GPT-NEOX architecture. It was developed by Together Computer
   and leaders from the open-source AI community. The model is
   fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability.
   More details about model can be found in `HuggingFace model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-  **gemma-7b-it** - Gemma is a family of lightweight, state-of-the-art
   open models from Google, built from the same research and technology
   used to create the Gemini models. They are text-to-text, decoder-only
   large language models, available in English, with open weights,
   pre-trained variants, and instruction-tuned variants. Gemma models
   are well-suited for a variety of text generation tasks, including
   question answering, summarization, and reasoning. This model is
   instruction-tuned version of 7B parameters model. More details about
   model can be found in `model
   card <https://huggingface.co/google/gemma-7b-it>`__. >\ **Note**: run
   model with demo, you will need to accept license agreement. >You must
   be a registered user in Hugging Face Hub. Please visit
   `HuggingFace model
   card <https://huggingface.co/google/gemma-7b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **gemma-2-9b-it** - Gemma2 is the second generation of a Gemma family
   of lightweight, state-of-the-art open models from Google, built from
   the same research and technology used to create the Gemini models.
   They are text-to-text, decoder-only large language models, available
   in English, with open weights, pre-trained variants, and
   instruction-tuned variants. Gemma models are well-suited for a
   variety of text generation tasks, including question answering,
   summarization, and reasoning. This model is instruction-tuned version
   of 9B parameters model. More details about model can be found in
   `model card <https://huggingface.co/google/gemma-2-9b-it>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/google/gemma-2-2b-it>`__, carefully read
   terms of usage and click accept button. You will need to use an
   access token for the code below to run. For more information on
   access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model


       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-2-7b-chat** - LLama 2 is the second generation of LLama
   models developed by Meta. Llama 2 is a collection of pre-trained and
   fine-tuned generative text models ranging in scale from 7 billion to
   70 billion parameters. llama-2-7b-chat is 7 billions parameters
   version of LLama 2 finetuned and optimized for dialogue use case.
   More details about model can be found in the
   `paper <https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
   `repository <https://github.com/facebookresearch/llama>`__ and
   `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-3-8b-instruct** - Llama 3 is an auto-regressive language
   model that uses an optimized transformer architecture. The tuned
   versions use supervised fine-tuning (SFT) and reinforcement learning
   with human feedback (RLHF) to align with human preferences for
   helpfulness and safety. The Llama 3 instruction tuned models are
   optimized for dialogue use cases and outperform many of the available
   open source chat models on common industry benchmarks. More details
   about model can be found in `Meta blog
   post <https://ai.meta.com/blog/meta-llama-3/>`__, `model
   website <https://llama.meta.com/llama3>`__ and `model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **llama-3.1-8b-instruct** - The Llama 3.1 instruction tuned text only
   models (8B, 70B, 405B) are optimized for multilingual dialogue use
   cases and outperform many of the available open source and closed
   chat models on common industry benchmarks. More details about model
   can be found in `Meta blog
   post <https://ai.meta.com/blog/meta-llama-3-1/>`__, `model
   website <https://llama.meta.com>`__ and `model
   card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__.
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       # login to huggingfacehub to get access to pretrained model

       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **qwen2.5-0.5b-instruct/qwen2.5-1.5b-instruct/qwen2.5-3b-instruct/qwen2.5-7b-instruct/qwen2.5-14b-instruct**
   - Qwen2.5 is the latest series of Qwen large language models.
   Comparing with Qwen2, Qwen2.5 series brings significant improvements
   in coding, mathematics and general knowledge skills. Additionally, it
   brings long-context and multiple languages support including Chinese,
   English, French, Spanish, Portuguese, German, Italian, Russian,
   Japanese, Korean, Vietnamese, Thai, Arabic, and more. For more
   details, please refer to
   `model_card <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__,
   `blog <https://qwenlm.github.io/blog/qwen2.5/>`__,
   `GitHub <https://github.com/QwenLM/Qwen2.5>`__, and
   `Documentation <https://qwen.readthedocs.io/en/latest/>`__.
-  **qwen-7b-chat** - Qwen-7B is the 7B-parameter version of the large
   language model series, Qwen (abbr. Tongyi Qianwen), proposed by
   Alibaba Cloud. Qwen-7B is a Transformer-based large language model,
   which is pretrained on a large volume of data, including web texts,
   books, codes, etc. For more details about Qwen, please refer to the
   `GitHub <https://github.com/QwenLM/Qwen>`__ code repository.
-  **chatglm3-6b** - ChatGLM3-6B is the latest open-source model in the
   ChatGLM series. While retaining many excellent features such as
   smooth dialogue and low deployment threshold from the previous two
   generations, ChatGLM3-6B employs a more diverse training dataset,
   more sufficient training steps, and a more reasonable training
   strategy. ChatGLM3-6B adopts a newly designed `Prompt
   format <https://github.com/THUDM/ChatGLM3/blob/main/PROMPT_en.md>`__,
   in addition to the normal multi-turn dialogue. You can find more
   details about model in the `model
   card <https://huggingface.co/THUDM/chatglm3-6b>`__
-  **mistral-7b** - The Mistral-7B-v0.1 Large Language Model (LLM) is a
   pretrained generative text model with 7 billion parameters. You can
   find more details about model in the `model
   card <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__,
   `paper <https://arxiv.org/abs/2310.06825>`__ and `release blog
   post <https://mistral.ai/news/announcing-mistral-7b/>`__.
-  **zephyr-7b-beta** - Zephyr is a series of language models that are
   trained to act as helpful assistants. Zephyr-7B-beta is the second
   model in the series, and is a fine-tuned version of
   `mistralai/Mistral-7B-v0.1 <https://huggingface.co/mistralai/Mistral-7B-v0.1>`__
   that was trained on on a mix of publicly available, synthetic
   datasets using `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. You can find more
   details about model in `technical
   report <https://arxiv.org/abs/2310.16944>`__ and `HuggingFace model
   card <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__.
-  **neural-chat-7b-v3-1** - Mistral-7b model fine-tuned using Intel
   Gaudi. The model fine-tuned on the open source dataset
   `Open-Orca/SlimOrca <https://huggingface.co/datasets/Open-Orca/SlimOrca>`__
   and aligned with `Direct Preference Optimization (DPO)
   algorithm <https://arxiv.org/abs/2305.18290>`__. More details can be
   found in `model
   card <https://huggingface.co/Intel/neural-chat-7b-v3-1>`__ and `blog
   post <https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3>`__.
-  **notus-7b-v1** - Notus is a collection of fine-tuned models using
   `Direct Preference Optimization
   (DPO) <https://arxiv.org/abs/2305.18290>`__. and related
   `RLHF <https://huggingface.co/blog/rlhf>`__ techniques. This model is
   the first version, fine-tuned with DPO over zephyr-7b-sft. Following
   a data-first approach, the only difference between Notus-7B-v1 and
   Zephyr-7B-beta is the preference dataset used for dDPO. Proposed
   approach for dataset creation helps to effectively fine-tune Notus-7b
   that surpasses Zephyr-7B-beta and Claude 2 on
   `AlpacaEval <https://tatsu-lab.github.io/alpaca_eval/>`__. More
   details about model can be found in `model
   card <https://huggingface.co/argilla/notus-7b-v1>`__.
-  **youri-7b-chat** - Youri-7b-chat is a Llama2 based model. `Rinna
   Co., Ltd. <https://rinna.co.jp/>`__ conducted further pre-training
   for the Llama2 model with a mixture of English and Japanese datasets
   to improve Japanese task capability. The model is publicly released
   on Hugging Face hub. You can find detailed information at the
   `rinna/youri-7b-chat project
   page <https://huggingface.co/rinna/youri-7b>`__.
-  **baichuan2-7b-chat** - Baichuan 2 is the new generation of
   large-scale open-source language models launched by `Baichuan
   Intelligence inc <https://www.baichuan-ai.com/home>`__. It is trained
   on a high-quality corpus with 2.6 trillion tokens and has achieved
   the best performance in authoritative Chinese and English benchmarks
   of the same size.
-  **internlm2-chat-1.8b** - InternLM2 is the second generation InternLM
   series. Compared to the previous generation model, it shows
   significant improvements in various capabilities, including
   reasoning, mathematics, and coding. More details about model can be
   found in `model repository <https://huggingface.co/internlm>`__.
-  **glm-4-9b-chat** - GLM-4-9B is the open-source version of the latest
   generation of pre-trained models in the GLM-4 series launched by
   Zhipu AI. In the evaluation of data sets in semantics, mathematics,
   reasoning, code, and knowledge, GLM-4-9B and its human
   preference-aligned version GLM-4-9B-Chat have shown superior
   performance beyond Llama-3-8B. In addition to multi-round
   conversations, GLM-4-9B-Chat also has advanced features such as web
   browsing, code execution, custom tool calls (Function Call), and long
   text reasoning (supporting up to 128K context). More details about
   model can be found in `model
   card <https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README_en.md>`__,
   `technical report <https://arxiv.org/pdf/2406.12793>`__ and
   `repository <https://github.com/THUDM/GLM-4>`__

.. raw:: html

   </details>

.. code:: ipython3

    from llm_config import get_llm_selection_widget

    form, lang, model_id_widget, compression_variant, use_preconverted = get_llm_selection_widget()

    form




.. parsed-literal::

    Box(children=(Box(children=(Label(value='Language:'), Dropdown(options=('English', 'Chinese', 'Japanese'), val…



.. code:: ipython3

    model_configuration = model_id_widget.value
    model_id = model_id_widget.label
    print(f"Selected model {model_id} with {compression_variant.value} compression")


.. parsed-literal::

    Selected model qwen2-0.5b-instruct with INT4 compression


Convert model using Optimum-CLI tool
------------------------------------



`Optimum Intel <https://huggingface.co/docs/optimum/intel/index>`__
is the interface between the
`Transformers <https://huggingface.co/docs/transformers/index>`__ and
`Diffusers <https://huggingface.co/docs/diffusers/index>`__ libraries
and OpenVINO to accelerate end-to-end pipelines on Intel architectures.
It provides ease-to-use cli interface for exporting models to `OpenVINO
Intermediate Representation
(IR) <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
format.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to read more about Optimum CLI usage

.. raw:: html

   </summary>

The command bellow demonstrates basic command for model export with
``optimum-cli``

::

   optimum-cli export openvino --model <model_id_or_path> --task <task> <out_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. For LLMs it is recommended to use
``text-generation-with-past``. If model initialization requires to use
remote code, ``--trust-remote-code`` flag additionally should be passed.

.. raw:: html

   </details>

Weights Compression using Optimum-CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



You can also apply fp16, 8-bit or 4-bit weight compression on the
Linear, Convolutional and Embedding layers when exporting your model
with the CLI.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to read more about weights compression with Optimum CLI

.. raw:: html

   </summary>

Setting ``--weight-format`` to respectively fp16, int8 or int4. This
type of optimization allows to reduce the memory footprint and inference
latency. By default the quantization scheme for int8/int4 will be
`asymmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization>`__,
to make it
`symmetric <https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization>`__
you can add ``--sym``.

For INT4 quantization you can also specify the following arguments :

- The ``--group-size`` parameter will define the group size to use for
  quantization, -1 it will results in per-column quantization.
- The ``--ratio`` parameter controls the ratio between 4-bit and 8-bit
  quantization. If set to 0.9, it means that 90% of the layers will be
  quantized to int4 while 10% will be quantized to int8.

Smaller group_size and ratio values usually improve accuracy at the
sacrifice of the model size and inference latency. You can enable AWQ to
be additionally applied during model export with INT4 precision using
``--awq`` flag and providing dataset name with ``--dataset``\ parameter
(e.g. ``--dataset wikitext2``)

   **Note**: Applying AWQ requires significant memory and time.

..

   **Note**: It is possible that there will be no matching patterns in
   the model to apply AWQ, in such case it will be skipped.

.. raw:: html

   </details>

.. code:: ipython3

    from llm_config import convert_and_compress_model

    model_dir = convert_and_compress_model(model_id, model_configuration, compression_variant.value, use_preconverted.value)


.. parsed-literal::

    ✅ INT4 qwen2-0.5b-instruct model already converted and can be found in qwen2/INT4_compressed_weights


Let’s compare model size for different compression types

.. code:: ipython3

    from llm_config import compare_model_size

    compare_model_size(model_dir)


.. parsed-literal::

    Size of model with INT4 compressed weights is 358.86 MB


Select device for inference
---------------------------



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="CPU", exclude=["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



The cell below demonstrates how to instantiate model based on selected
variant of model weights and inference device

Instantiate pipeline with OpenVINO Generate API
-----------------------------------------------



`OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__
can be used to create pipelines to run an inference with OpenVINO
Runtime.

Firstly we need to create a pipeline with ``LLMPipeline``.
``LLMPipeline`` is the main object used for text generation using LLM in
OpenVINO GenAI API. You can construct it straight away from the folder
with the converted model. We will provide directory with model and
device for ``LLMPipeline``. Then we run ``generate`` method and get the
output in text format. Additionally, we can configure parameters for
decoding. We can get the default config with
``get_generation_config()``, setup parameters, and apply the updated
version with ``set_generation_config(config)`` or put config directly to
``generate()``. It’s also possible to specify the needed options just as
inputs in the ``generate()`` method, as shown below, e.g. we can add
``max_new_tokens`` to stop generation if a specified number of tokens is
generated and the end of generation is not reached. We will discuss some
of the available generation parameters more deeply later.

.. code:: ipython3

    from openvino_genai import LLMPipeline

    print(f"Loading model from {model_dir}\n")


    pipe = LLMPipeline(str(model_dir), device.value)

    generation_config = pipe.get_generation_config()

    input_prompt = "The Sun is yellow bacause"
    print(f"Input text: {input_prompt}")
    print(pipe.generate(input_prompt, max_new_tokens=10))


.. parsed-literal::

    Loading model from qwen2/INT4_compressed_weights

    Input text: The Sun is yellow bacause
     it is made of hydrogen and oxygen atoms. The


Run Chatbot
-----------



Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to see how pipeline works

.. raw:: html

   </summary>

The diagram below illustrates how the chatbot pipeline works

.. figure:: https://github.com/user-attachments/assets/9c9b56e1-01a6-48d8-aa46-222a88e25066
   :alt: llm_diagram

   llm_diagram

As you can see, user input question passed via tokenizer to apply
chat-specific formatting (chat template) and turn the provided string
into the numeric format. `OpenVINO
Tokenizers <https://github.com/openvinotoolkit/openvino_tokenizers>`__
are used for these purposes inside ``LLMPipeline``. You can find more
detailed info about tokenization theory and OpenVINO Tokenizers in this
`tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-tokenizers/openvino-tokenizers.ipynb>`__.
Then tokenized input passed to LLM for making prediction of next token
probability. The way the next token will be selected over predicted
probabilities is driven by the selected decoding methodology. You can
find more information about the most popular decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The sampler’s
goal is to select the next token id is driven by generation
configuration. Next, we apply stop generation condition to check the
generation is finished or not (e.g. if we reached the maximum new
generated tokens or the next token id equals to end of the generation).
If the end of the generation is not reached, then new generated token id
is used as the next iteration input, and the generation cycle repeats
until the condition is not met. When stop generation criteria are met,
then OpenVINO Detokenizer decodes generated token ids to text answer.

The difference between chatbot and instruction-following pipelines is
that the model should have “memory” to find correct answers on the chain
of connected questions. OpenVINO GenAI uses ``KVCache`` representation
for maintain a history of conversation. By default, ``LLMPipeline``
resets ``KVCache`` after each ``generate`` call. To keep conversational
history, we should move LLMPipeline to chat mode using ``start_chat()``
method.

More info about OpenVINO LLM inference can be found in `LLM Inference
Guide <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html>`__

.. raw:: html

   </details>

Advanced generation options
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here to see detailed description of advanced options

.. raw:: html

   </summary>

| There are several parameters that can control text generation quality,
  \* ``Temperature`` is a parameter used to control the level of
  creativity in AI-generated text. By adjusting the ``temperature``, you
  can influence the AI model’s probability distribution, making the text
  more focused or diverse.
| Consider the following example: The AI model has to complete the
  sentence “The cat is \____.” with the following token probabilities:

| playing: 0.5
| sleeping: 0.25
| eating: 0.15
| driving: 0.05
| flying: 0.05

-  **Low temperature** (e.g., 0.2): The AI model becomes more focused
   and deterministic, choosing tokens with the highest probability, such
   as “playing.”

   -  **Medium temperature** (e.g., 1.0): The AI model maintains a
      balance between creativity and focus, selecting tokens based on
      their probabilities without significant bias, such as “playing,”
      “sleeping,” or “eating.”
   -  **High temperature** (e.g., 2.0): The AI model becomes more
      adventurous, increasing the chances of selecting less likely
      tokens, such as “driving” and “flying.”

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is an another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.
-  ``Repetition Penalty`` This parameter can help penalize tokens based
   on how frequently they occur in the text, including the input prompt.
   A token that has already appeared five times is penalized more
   heavily than a token that has appeared only one time. A value of 1
   means that there is no penalty and values larger than 1 discourage
   repeated tokens.

.. raw:: html

   </details>

.. code:: ipython3

    if not Path("gradio_helper_genai.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-chatbot/gradio_helper_genai.py")
        open("gradio_helper_genai.py", "w").write(r.text)

    from gradio_helper_genai import make_demo

    demo = make_demo(pipe, model_configuration, model_id, lang.value)

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
