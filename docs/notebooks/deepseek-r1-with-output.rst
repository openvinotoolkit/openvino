LLM reasoning with DeepSeek-R1 distilled models
===============================================

`DeepSeek-R1 <https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf>`__
is an open-source reasoning model developed by DeepSeek to address tasks
requiring logical inference, mathematical problem-solving, and real-time
decision-making. With DeepSeek-R1, you can follow its logic, making it
easier to understand and, if necessary, challenge its output. This
capability gives reasoning models an edge in fields where outcomes need
to be explainable, like research or complex decision-making.

Distillation in AI creates smaller, more efficient models from larger
ones, preserving much of their reasoning power while reducing
computational demands. DeepSeek applied this technique to create a suite
of distilled models from R1, using Qwen and Llama architectures. That
allows us to try DeepSeek-R1 capability locally on usual laptops.

In this tutorial, we consider how to run DeepSeek-R1 distilled models
using OpenVINO.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Convert model using Optimum-CLI
   tool <#convert-model-using-optimum-cli-tool>`__

   -  `Weights Compression using
      Optimum-CLI <#weights-compression-using-optimum-cli>`__

-  `Instantiate pipeline with OpenVINO Generate
   API <#instantiate-pipeline-with-openvino-generate-api>`__
-  `Run Chatbot <#run-chatbot>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install required dependencies

.. code:: ipython3

    import os
    import platform

    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

    %pip install -q -U "openvino>=2024.6.0" openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "nncf==2.14.1"\
    "torch>=2.1"\
    "datasets" \
    "accelerate" \
    "gradio>=4.19" \
    "transformers>=4.43.1" \
    "huggingface-hub>=0.26.5" \
    "einops" "tiktoken"

    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"

.. code:: ipython3

    import requests
    from pathlib import Path

    if not Path("llm_config.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("llm_config.py", "w").write(r.text)

    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("deepseek-r1.ipynb")

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of LLM solutions:

-  **DeepSeek-R1-Distill-Llama-8B** is a distilled model based on
   `Llama-3.1-8B <https://huggingface.co/meta-llama/Llama-3.1-8B>`__,
   that prioritizes high performance and advanced reasoning
   capabilities, particularly excelling in tasks requiring mathematical
   and factual precision. Check `model
   card <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`__
   for more info.
-  **DeepSeek-R1-Distill-Qwen-1.5B** is the smallest DeepSeek-R1
   distilled model based on
   `Qwen2.5-Math-1.5B <https://huggingface.co/Qwen/Qwen2.5-Math-1.5B>`__.
   Despite its compact size, the model demonstrates strong capabilities
   in solving basic mathematical tasks, at the same time its programming
   capabilities are limited. Check `model
   card <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__
   for more info.
-  **DeepSeek-R1-Distill-Qwen-7B** is a distilled model based on
   `Qwen-2.5-Math-7B <https://huggingface.co/Qwen/Qwen2.5-Math-7B>`__.
   The model demonstrates a good balance between mathematical and
   factual reasoning and can be less suited for complex coding tasks.
   Check `model
   card <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__
   for more info.
-  **DeepSeek-R1-Distil-Qwen-14B** is a distilled model based on
   `Qwen2.5-14B <https://huggingface.co/Qwen/Qwen2.5-14B>`__ that has
   great competence in factual reasoning and solving complex
   mathematical tasks. Check `model
   card <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-15B>`__
   for more info.

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
-  **INT4 NPU-friendly** is an 4-bit channel-wise quantization. This
   approach is
   `recommended <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide-npu.html>`__
   for LLM inference using NPU.

.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget(default="CPU")

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    from llm_config import get_llm_selection_widget

    form, lang, model_id_widget, compression_variant, _ = get_llm_selection_widget(device=device.value)

    form




.. parsed-literal::

    Box(children=(Box(children=(Label(value='Language:'), Dropdown(options=('English', 'Chinese'), value='English'…



.. code:: ipython3

    model_configuration = model_id_widget.value
    model_id = model_id_widget.label
    print(f"Selected model {model_id} with {compression_variant.value} compression")


.. parsed-literal::

    Selected model DeepSeek-R1-Distill-Llama-8B with INT4 compression


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

    model_dir = convert_and_compress_model(model_id, model_configuration, compression_variant.value)


.. parsed-literal::

    ✅ INT4 DeepSeek-R1-Distill-Llama-8B model already converted and can be found in DeepSeek-R1-Distill-Llama-8B/INT4_compressed_weights


.. code:: ipython3

    from llm_config import compare_model_size

    compare_model_size(model_dir)


.. parsed-literal::

    Size of model with INT4 compressed weights is 5081.91 MB


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
decoding. We can create the default config with
``ov_genai.GenerationConfig()``, setup parameters, and apply the updated
version with ``set_generation_config(config)`` or put config directly to
``generate()``. It’s also possible to specify the needed options just as
inputs in the ``generate()`` method, as shown below, e.g. we can add
``max_new_tokens`` to stop generation if a specified number of tokens is
generated and the end of generation is not reached. We will discuss some
of the available generation parameters more deeply later. Generation
process for long response may be time consuming, for accessing partial
result as soon as it is generated without waiting when whole process
finished, Streaming API can be used. Token streaming is the mode in
which the generative system returns the tokens one by one as the model
generates them. This enables showing progressive generations to the user
rather than waiting for the whole generation. Streaming is an essential
aspect of the end-user experience as it reduces latency, one of the most
critical aspects of a smooth experience. In code below, we implement
simple streamer for printing output result. For more advanced streamer
example please check openvino.genai
`sample <https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/multinomial_causal_lm>`__.

.. code:: ipython3

    import openvino_genai as ov_genai
    import sys

    print(f"Loading model from {model_dir}\n")


    pipe = ov_genai.LLMPipeline(str(model_dir), device.value)
    if "genai_chat_template" in model_configuration:
        pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 128


    def streamer(subword):
        print(subword, end="", flush=True)
        sys.stdout.flush()
        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False


    input_prompt = "What is OpenVINO?"
    print(f"Input text: {input_prompt}")
    result = pipe.generate(input_prompt, generation_config, streamer)


.. parsed-literal::

    Loading model from DeepSeek-R1-Distill-Llama-8B/INT4_compressed_weights

    Input text: What is OpenVINO?
     It's an open-source model optimization tool that accelerates AI deployment across various platforms. It supports multiple frameworks and platforms, providing tools for quantization, pruning, and knowledge distillation. OpenVINO is designed to help developers reduce the computational requirements of AI models, making them more efficient and deployable on resource-constrained environments.

    What is OpenVINO? It's an open-source model optimization tool that accelerates AI deployment across various platforms. It supports multiple frameworks and platforms, providing tools for quantization, pruning, and knowledge distillation. OpenVINO is designed to help developers reduce the computational requirements of AI models, making them more

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

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/deepseek-r1/gradio_helper.py")
        open("gradio_helper_genai.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(pipe, model_configuration, model_id, lang.value, device.value == "NPU")

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/
