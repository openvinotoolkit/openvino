Structure Extraction with NuExtract and OpenVINO
================================================

.. figure:: https://github.com/user-attachments/assets/70dd93cc-da36-4c53-8891-78c0f9a41f20
   :alt: image

   image

`NuExtract <https://huggingface.co/numind/NuExtract>`__ model is a
text-to-JSON Large Language Model (LLM) that allows to extract
arbitrarily complex information from text and turns it into structured
data.

LLM stands for “Large Language Model” which refers to a type of
artificial intelligence model that is designed to understand and
generate human-like text based on the input it receives. LLMs are
trained on large datasets of text to learn patterns, grammar, and
semantic relationships, allowing them to generate coherent and
contextually relevant responses. One core capability of Large Language
Models (LLMs) is to follow natural language instructions.
Instruction-following models are capable of generating text in response
to prompts and are often used for tasks like writing assistance,
chatbots, and content generation.

In this tutorial, we consider how to run a structure extraction text
generation pipeline using NuExtract model and OpenVINO. We will use
pre-trained models from the `Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. The `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library
converts the models to OpenVINO™ IR format. To simplify the user
experience, we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai>`__ for
generation inference pipeline.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__
-  Compress model weights to INT8 and INT4 with `OpenVINO
   NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a structure extraction inference pipeline with `Generate
   API <https://github.com/openvinotoolkit/openvino.genai>`__
-  Launch interactive Gradio demo with structure extraction pipeline


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Select model for inference <#select-model-for-inference>`__
-  `Download and convert model to OpenVINO IR via Optimum Intel
   CLI <#download-and-convert-model-to-openvino-ir-via-optimum-intel-cli>`__
-  `Compress model weights <#compress-model-weights>`__

   -  `Weights Compression using Optimum Intel
      CLI <#weights-compression-using-optimum-intel-cli>`__

-  `Select device for inference and model
   variant <#select-device-for-inference-and-model-variant>`__
-  `Create a structure extraction inference
   pipeline <#create-a-structure-extraction-inference-pipeline>`__
-  `Run interactive structure extraction demo with
   Gradio <#run-interactive-structure-extraction-demo-with-gradio>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip uninstall -q -y optimum optimum-intel
    %pip install -Uq "openvino>=2024.3.0" "openvino-genai"
    %pip install -q "torch>=2.1" "nncf>=2.12" "transformers>=4.40.0" "accelerate" "gradio>=4.19" "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu

.. code:: ipython3

    import os
    from pathlib import Path
    import requests
    import shutil
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import download_file
    
    # Fetch llm_config.py
    llm_config_shared_path = Path("../../utils/llm_config.py")
    llm_config_dst_path = Path("llm_config.py")
    
    if not llm_config_dst_path.exists():
        if llm_config_shared_path.exists():
            try:
                os.symlink(llm_config_shared_path, llm_config_dst_path)
            except Exception:
                shutil.copy(llm_config_shared_path, llm_config_dst_path)
        else:
            download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
    elif not os.path.islink(llm_config_dst_path):
        print("LLM config will be updated")
        if llm_config_shared_path.exists():
            shutil.copy(llm_config_shared_path, llm_config_dst_path)
        else:
            download_file(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")

Select model for inference
--------------------------



The tutorial supports different models, you can select one from the
provided options to compare the quality of open source solutions.
>\ **Note**: conversion of some models can require additional actions
from user side and at least 64GB RAM for conversion.

NuExtract model has several versions:

-  **NuExtract-tiny** - This is a version of
   `Qwen1.5-0.5 <https://huggingface.co/Qwen/Qwen1.5-0.5B>`__ model with
   0.5 billion parameters. More details about the model can be found in
   `model card <https://huggingface.co/numind/NuExtract-tiny>`__.
-  **NuExtract** - This is a version of
   `phi-3-mini <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>`__
   model with 3.8 billion parameters. More details about the model can
   be found in `model card <https://huggingface.co/numind/NuExtract>`__.
-  **NuExtract-large** - This is a version of
   `phi-3-small <https://huggingface.co/microsoft/Phi-3-small-8k-instruct>`__
   model with 7 billion parameters. More details about the model can be
   found in `model
   card <https://huggingface.co/numind/NuExtract-large>`__.

All NuExtract models are fine-tuned on a private high-quality synthetic
dataset for information extraction.

.. code:: ipython3

    from llm_config import get_llm_selection_widget
    
    models = {
        "NuExtract_tiny": {"model_id": "numind/NuExtract-tiny"},
        "NuExtract": {"model_id": "numind/NuExtract"},
        "NuExtract_large": {"model_id": "numind/NuExtract-large"},
    }
    
    form, _, model_dropdown, compression_dropdown, _ = get_llm_selection_widget(languages=None, models=models, show_preconverted_checkbox=False)
    
    form




.. parsed-literal::

    Box(children=(Box(children=(Label(value='Model:'), Dropdown(options={'NuExtract_tiny': {'model_id': 'numind/Nu…



.. code:: ipython3

    model_name = model_dropdown.label
    model_config = model_dropdown.value
    print(f"Selected model {model_name} with {compression_dropdown.value} compression")


.. parsed-literal::

    Selected model NuExtract_tiny with INT4 compression
    

Download and convert model to OpenVINO IR via Optimum Intel CLI
---------------------------------------------------------------



Listed model are available for downloading via the `HuggingFace
hub <https://huggingface.co/models>`__. We will use optimum-cli
interface for exporting it into OpenVINO Intermediate Representation
(IR) format.

Optimum CLI interface for converting models supports export to OpenVINO
(supported starting optimum-intel 1.12 version). General command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where ``--model`` argument is model id from HuggingFace Hub or local
directory with model (saved using ``.save_pretrained`` method),
``--task`` is one of `supported
task <https://huggingface.co/docs/optimum/exporters/task_manager>`__
that exported model should solve. If ``--task`` is not specified, the
task will be auto-inferred based on the model. If model initialization
requires to use remote code, ``--trust-remote-code`` flag additionally
should be passed. Full list of supported arguments available via
``--help`` For more details and examples of usage, please check `optimum
documentation <https://huggingface.co/docs/optimum/intel/inference#export>`__.

Compress model weights
----------------------



The Weights Compression algorithm is aimed at compressing the weights of
the models and can be used to optimize the model footprint and
performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM). Compared to INT8 compression, INT4 compression improves
performance even more but introduces a minor drop in prediction quality.

Weights Compression using Optimum Intel CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Optimum Intel supports weight compression via NNCF out of the box. For
8-bit compression we pass ``--weight-format int8`` to ``optimum-cli``
command line. For 4 bit compression we provide ``--weight-format int4``
and some other options containing number of bits and other compression
parameters. An example of this approach usage you can find in
`llm-chatbot notebook <llm-chatbot-with-output.html>`__

   **Note**: This tutorial involves conversion model for FP16 and
   INT4/INT8 weights compression scenarios. It may be memory and
   time-consuming in the first run. You can manually control the
   compression precision below. **Note**: There may be no speedup for
   INT4/INT8 compressed models on dGPU

.. code:: ipython3

    from llm_config import convert_and_compress_model
    
    model_dir = convert_and_compress_model(model_name, model_config, compression_dropdown.value, use_preconverted=False)


.. parsed-literal::

    ⌛ NuExtract_tiny conversion to INT4 started. It may takes some time.
    


**Export command:**



``optimum-cli export openvino --model numind/NuExtract-tiny --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 NuExtract_tiny/INT4_compressed_weights``


.. parsed-literal::

    Framework not specified. Using pt to export the model.
    Using framework PyTorch: 2.3.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> True
    We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
    /home/ytarkan/miniconda3/envs/ov_notebooks_env/lib/python3.9/site-packages/optimum/exporters/openvino/model_patcher.py:489: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if sequence_length != 1:
    /home/ytarkan/miniconda3/envs/ov_notebooks_env/lib/python3.9/site-packages/transformers/models/qwen2/modeling_qwen2.py:110: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if seq_len > self.max_seq_len_cached:
    

.. parsed-literal::

    [2KMixed-Precision assignment [90m━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m [36m168/168[0m • [36m0:00:01[0m • [36m0:00:00[0m• [36m0:00:01[0m
    [?25hINFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 47% (47 / 169)              │ 20% (46 / 168)                         │
    ├────────────────┼─────────────────────────────┼────────────────────────���───────────────┤
    │              4 │ 53% (122 / 169)             │ 80% (122 / 168)                        │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    [2KApplying Weight Compression [90m━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m [36m169/169[0m • [36m0:00:05[0m • [36m0:00:00[0m• [36m0:00:01[0m
    [?25h

.. parsed-literal::

    Set tokenizer padding side to left for `text-generation-with-past` task.
    Replacing `(?!\S)` pattern to `(?:$|[^\S])` in RegexSplit operation
    

.. parsed-literal::

    ✅ INT4 NuExtract_tiny model converted and can be found in NuExtract_tiny/INT4_compressed_weights
    

Let’s compare model size for different compression types

.. code:: ipython3

    from llm_config import compare_model_size
    
    compare_model_size(model_dir)


.. parsed-literal::

    Size of model with INT4 compressed weights is 347.03 MB
    

Select device for inference and model variant
---------------------------------------------



   **Note**: There may be no speedup for INT4/INT8 compressed models on
   dGPU.

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget(default="CPU", exclude=["NPU"])
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU')



Create a structure extraction inference pipeline
------------------------------------------------



Firstly we will prepare input prompt for NuExtract model by introducing
``prepare_input()`` function. This function combines the main text, a
JSON schema and optional examples into a single string that adheres to
model’s specific input requirements.

``prepare_input()`` function accepts the following parameters: 1.
``text``: This is the primary text from which you want to extract
information. 2. ``schema``: A JSON schema string that defines the
structure of the information you want to extract. This acts as a
template, guiding NuExtract model on what data to look for and how to
format the output. 3. ``examples``: An optional list of example strings.
These can be used to provide the model with sample extractions,
potentially improving accuracy for complex or ambiguous cases.

.. code:: ipython3

    import json
    from typing import List
    
    
    def prepare_input(text: str, schema: str, examples: List[str] = ["", "", ""]) -> str:
        schema = json.dumps(json.loads(schema), indent=4)
        input_llm = "<|input|>\n### Template:\n" + schema + "\n"
        for example in examples:
            if example != "":
                input_llm += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n"
    
        input_llm += "### Text:\n" + text + "\n<|output|>\n"
        return input_llm

To simplify user experience we will use `OpenVINO Generate
API <https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md>`__.
We will create pipeline with ``LLMPipeline``. ``LLMPipeline`` is the
main object used for decoding. You can construct it straight away from
the folder with the converted model. It will automatically load the
``main model``, ``tokenizer``, ``detokenizer`` and default
``generation configuration``. After that we will configure parameters
for decoding. We can get default config with
``get_generation_config()``, setup parameters and apply the updated
version with ``set_generation_config(config)`` or put config directly to
``generate()``. It’s also possible to specify the needed options just as
inputs in the ``generate()`` method, as shown below. Then we just run
``generate`` method and get the output in text format. We do not need to
encode input prompt according to model expected template or write
post-processing code for logits decoder, it will be done easily with
LLMPipeline.

.. code:: ipython3

    from openvino_genai import LLMPipeline
    
    pipe = LLMPipeline(model_dir.as_posix(), device.value)
    
    
    def run_structure_extraction(text: str, schema: str) -> str:
        input = prepare_input(text, schema)
        return pipe.generate(input, max_new_tokens=200)

To run structure extraction inference pipeline we need to provide
example text for data extraction and define output structure in a JSON
schema format:

.. code:: ipython3

    text = """We introduce Mistral 7B, a 7-billion-parameter language model engineered for
    superior performance and efficiency. Mistral 7B outperforms the best open 13B
    model (Llama 2) across all evaluated benchmarks, and the best released 34B
    model (Llama 1) in reasoning, mathematics, and code generation. Our model
    leverages grouped-query attention (GQA) for faster inference, coupled with sliding
    window attention (SWA) to effectively handle sequences of arbitrary length with a
    reduced inference cost. We also provide a model fine-tuned to follow instructions,
    Mistral 7B - Instruct, that surpasses Llama 2 13B - chat model both on human and
    automated benchmarks. Our models are released under the Apache 2.0 license.
    Code: https://github.com/mistralai/mistral-src
    Webpage: https://mistral.ai/news/announcing-mistral-7b/"""
    
    schema = """{
        "Model": {
            "Name": "",
            "Number of parameters": "",
            "Number of max token": "",
            "Architecture": []
        },
        "Usage": {
            "Use case": [],
            "Licence": ""
        }
    }"""
    
    output = run_structure_extraction(text, schema)
    print(output)


.. parsed-literal::

    {
        "Model": {
            "Name": "Mistral 7B",
            "Number of parameters": "7-billion",
            "Number of max token": "",
            "Architecture": [
                "grouped-query attention",
                "sliding window attention"
            ]
        },
        "Usage": {
            "Use case": [
                "reasoning",
                "mathematics",
                "code generation"
            ],
           "Licence": "Apache 2.0"
        }
    }
    
    

Run interactive structure extraction demo with Gradio
-----------------------------------------------------



.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/nuextract-structure-extraction/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=run_structure_extraction)
    
    try:
        demo.launch(height=800)
    except Exception:
        demo.launch(share=True, height=800)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/

.. code:: ipython3

    # Uncomment and run this cell for stopping gradio interface
    # demo.close()
