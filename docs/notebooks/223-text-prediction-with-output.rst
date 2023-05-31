Text Prediction with OpenVINO™
==============================

This notebook shows text prediction with OpenVINO. This notebook can
work in two different modes, Text Generation and Conversation, which the
user can select via selecting the model in the Model Selection Section.
We use three models
`GPT-2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`__,
`GPT-Neo <https://zenodo.org/record/5297715#.ZAmpsXZBztU>`__, and
`PersonaGPT <https://arxiv.org/abs/2110.12949v1>`__, which are a part of
the Generative Pre-trained Transformer (GPT) family. GPT-2 and GPT-Neo
can be used for text generation, whereas PersonaGPT is trained for the
downstream task of conversation.

GPT-2 and GPT-Neo are pre-trained on a large corpus of English text
using unsupervised training. They both display a broad set of
capabilities, including the ability to generate conditional synthetic
text samples of unprecedented quality, where we prime the model with an
input and have it generate a lengthy continuation.

More Details about the models are provided on their huggingface cards:

-  `GPT-2 <https://huggingface.co/gpt2>`__
-  `GPT-Neo <https://huggingface.co/EleutherAI/gpt-neo-125M>`__

PersonaGPT is an open-domain conversational agent that can decode
*personalized* and *controlled* responses based on user input. It is
built on the pretrained
`DialoGPT-medium <https://github.com/microsoft/DialoGPT>`__ model,
following the `GPT-2 <https://github.com/openai/gpt-2>`__ architecture.
PersonaGPT is fine-tuned on the
`Persona-Chat <https://arxiv.org/pdf/1801.07243>`__ dataset. The model
is available from
`HuggingFace <https://huggingface.co/af1tang/personaGPT>`__. PersonaGPT
displays a broad set of capabilities, including the ability to take on
personas, where we prime the model with few facts and have it generate
based upon that, it can also be used for creating a chatbot on a
knowledge base.

The following image illustrates the complete demo pipeline used for text
generation:

.. figure:: https://user-images.githubusercontent.com/91228207/163990722-d2713ede-921e-4594-8b00-8b5c1a4d73b5.jpeg
   :alt: image2

   image2

This is a demonstration in which the user can type the beginning of the
text and the network will generate a further. This procedure can be
repeated as many times as the user desires.

For Text Generation, The model input is tokenized text, which serves as
the initial condition for text generation. Then, logits from the models’
inference results are obtained, and the token with the highest
probability is selected using the top-k sampling strategy and joined to
the input sequence. This procedure repeats until the end of the sequence
token is received or the specified maximum length is reached. After
that, tokenized IDs are decoded to text.

The following image illustrates the demo pipeline for conversation:

.. figure:: https://user-images.githubusercontent.com/95569637/226101538-e204aebd-a34f-4c8b-b90c-5363ba41c080.jpeg
   :alt: image2

   image2

For Conversation, User Input is tokenized with eos_token concatenated in
the end. Then, the text gets generated as detailed above. The Generated
response is added to the history with the eos_token at the end.
Additional user input is added to the history, and the sequence is
passed back into the model.

Model Selection
---------------

Select the Model to be used for text generation, GPT-2 and GPT-Neo are
used for text generation wheras PersonaGPT is used for Conversation.

.. code:: ipython3

    # Install Gradio for Interactive Inference
    !pip install gradio


.. parsed-literal::

    Requirement already satisfied: gradio in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (3.11.0)
    Requirement already satisfied: aiohttp in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.8.4)
    Requirement already satisfied: fastapi in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.95.2)
    Requirement already satisfied: ffmpy in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.3.0)
    Requirement already satisfied: fsspec in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (2023.5.0)
    Requirement already satisfied: h11<0.13,>=0.11 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.12.0)
    Requirement already satisfied: httpx in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.24.1)
    Requirement already satisfied: jinja2 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.1.2)
    Requirement already satisfied: markdown-it-py[linkify,plugins] in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (2.2.0)
    Requirement already satisfied: matplotlib in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.5.2)
    Requirement already satisfied: numpy in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (1.23.4)
    Requirement already satisfied: orjson in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.8.14)
    Requirement already satisfied: pandas in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (1.3.5)
    Requirement already satisfied: paramiko in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.2.0)
    Requirement already satisfied: pillow in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (9.5.0)
    Requirement already satisfied: pycryptodome in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (3.18.0)
    Requirement already satisfied: pydantic in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (1.10.8)
    Requirement already satisfied: pydub in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.25.1)
    Requirement already satisfied: python-multipart in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.0.6)
    Requirement already satisfied: pyyaml in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (6.0)
    Requirement already satisfied: requests in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (2.31.0)
    Requirement already satisfied: uvicorn in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (0.22.0)
    Requirement already satisfied: websockets>=10.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from gradio) (11.0.3)
    Requirement already satisfied: attrs>=17.3.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (23.1.0)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (3.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (1.3.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from aiohttp->gradio) (1.3.1)
    Requirement already satisfied: starlette<0.28.0,>=0.27.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from fastapi->gradio) (0.27.0)
    Requirement already satisfied: typing-extensions>=4.2.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pydantic->gradio) (4.6.2)
    Requirement already satisfied: certifi in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from httpx->gradio) (2023.5.7)
    Requirement already satisfied: httpcore<0.18.0,>=0.15.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from httpx->gradio) (0.15.0)
    Requirement already satisfied: idna in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from httpx->gradio) (3.4)
    Requirement already satisfied: sniffio in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from httpx->gradio) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jinja2->gradio) (2.1.2)
    Requirement already satisfied: mdurl~=0.1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from markdown-it-py[linkify,plugins]->gradio) (0.1.2)
    Requirement already satisfied: linkify-it-py<3,>=1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from markdown-it-py[linkify,plugins]->gradio) (2.0.2)
    Requirement already satisfied: mdit-py-plugins in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from markdown-it-py[linkify,plugins]->gradio) (0.3.5)
    Requirement already satisfied: cycler>=0.10 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (4.39.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (23.1)
    Requirement already satisfied: pyparsing>=2.2.1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->gradio) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pandas->gradio) (2023.3)
    Requirement already satisfied: bcrypt>=3.2 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from paramiko->gradio) (4.0.1)
    Requirement already satisfied: cryptography>=3.3 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from paramiko->gradio) (40.0.2)
    Requirement already satisfied: pynacl>=1.5 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from paramiko->gradio) (1.5.0)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->gradio) (1.26.16)
    Requirement already satisfied: click>=7.0 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from uvicorn->gradio) (8.1.3)
    Requirement already satisfied: cffi>=1.12 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from cryptography>=3.3->paramiko->gradio) (1.15.1)
    Requirement already satisfied: anyio==3.* in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from httpcore<0.18.0,>=0.15.0->httpx->gradio) (3.7.0)
    Requirement already satisfied: exceptiongroup in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from anyio==3.*->httpcore<0.18.0,>=0.15.0->httpx->gradio) (1.1.1)
    Requirement already satisfied: uc-micro-py in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from linkify-it-py<3,>=1->markdown-it-py[linkify,plugins]->gradio) (1.0.2)
    Requirement already satisfied: six>=1.5 in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib->gradio) (1.16.0)
    Requirement already satisfied: pycparser in /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from cffi>=1.12->cryptography>=3.3->paramiko->gradio) (2.21)


.. code:: ipython3

    from gradio import Blocks, Chatbot, Textbox, Row, Column
    import ipywidgets as widgets
    
    style = {'description_width': 'initial'}
    model_name = widgets.Select(
        options=['GPT-2', 'GPT-Neo', 'PersonaGPT (Converastional)'],
        value='GPT-Neo',
        description='Select Model:',
        disabled=False
    )
    
    widgets.VBox([model_name])




.. parsed-literal::

    VBox(children=(Select(description='Select Model:', index=1, options=('GPT-2', 'GPT-Neo', 'PersonaGPT (Converas…



Load Model
----------

Download the Selected Model and Tokenizer from Huggingface

.. code:: ipython3

    from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel
    
    if model_name.value == "PersonaGPT (Converastional)":
        pt_model = GPT2LMHeadModel.from_pretrained('af1tang/personaGPT')
        tokenizer = GPT2Tokenizer.from_pretrained('af1tang/personaGPT')
    elif model_name.value == 'GPT-2':
        pt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name.value == 'GPT-Neo':
        pt_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
        tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')

Convert Pytorch Model to OpenVINO IR
------------------------------------

.. figure:: https://user-images.githubusercontent.com/29454499/211261803-784d4791-15cb-4aea-8795-0969dfbb8291.png
   :alt: conversion_pipeline

   conversion_pipeline

For starting work with GPT-Neo model using OpenVINO, model should be
converted to OpenVINO Intermediate Represenation (IR) format.
HuggingFace provides gpt-neo model in PyTorch format, which supported in
OpenVINO via conversion to ONNX. We use HuggingFace transformers
library’s oonx module to export model to ONNX.
``transformers.onnx.export`` accepts preprocessing function for input
sample generation (tokenizer in our case),an instance of model, ONNX
export configuration, ONNX opset version for export and output path.
More information about transformers export to ONNX can be found in
HuggingFace
`documentation <https://huggingface.co/docs/transformers/serialization>`__.

While ONNX models are directly supported by OpenVINO runtime, it can be
useful to convert them to IR format to take advantage of OpenVINO
optimization tools and features. ``mo.convert_model`` python function
can be used for converting model using `OpenVINO Model
Optimizer <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Python_API.html>`__.
The function returns instance of OpenVINO Model class, which is ready to
use in Python interface but can also be serialized to OpenVINO IR format
for future execution using ``openvino.runtime.serialize``. In our case,
``compress_to_fp16`` parameter is enabled for compression model weights
to fp16 precision and also specified dynamic input shapes with possible
shape range (from 1 token to maximum length defined in our processing
function) for optimization of memory consumption.

.. code:: ipython3

    from pathlib import Path
    from openvino.runtime import serialize
    from openvino.tools import mo
    from transformers.onnx import export, FeaturesManager
    
    
    # define path for saving onnx model
    onnx_path = Path("model/text_generator.onnx")
    onnx_path.parent.mkdir(exist_ok=True)
    
    # define path for saving openvino model
    model_path = onnx_path.with_suffix(".xml")
    
    # get model onnx config function for output feature format casual-lm
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(pt_model, feature='causal-lm')
    
    # fill onnx config based on pytorch model config
    onnx_config = model_onnx_config(pt_model.config)
    
    # convert model to onnx
    onnx_inputs, onnx_outputs = export(preprocessor=tokenizer,model=pt_model,config=onnx_config,opset=onnx_config.default_onnx_opset,output=onnx_path)
    
    # convert model to openvino
    if model_name.value == "PersonaGPT (Converastional)":
        ov_model = mo.convert_model(onnx_path, compress_to_fp16=True, input="input_ids[1,1..1000],attention_mask[1,1..1000]")
    else:
        ov_model = mo.convert_model(onnx_path, compress_to_fp16=True, input="input_ids[1,1..128],attention_mask[1,1..128]")
    
    # serialize openvino model
    serialize(ov_model, str(model_path))


.. parsed-literal::

    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:555: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if batch_size <= 0:
    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    Warning: One or more of the values of the Constant can't fit in the float16 data type. Those values were casted to the nearest limit value, the model can produce incorrect results.
    Warning: One or more of the values of the Constant can't fit in the float16 data type. Those values were casted to the nearest limit value, the model can produce incorrect results.


Load the model
~~~~~~~~~~~~~~

We start by building an OpenVINO Core object. Then we read the network
architecture and model weights from the .xml and .bin files,
respectively. Finally, we compile the model for the desired device.
Because we use the dynamic shapes feature, which is only available on
CPU, we must use ``CPU`` for the device. Dynamic shapes support on GPU
is coming soon.

Since the text recognition model has a dynamic input shape, you cannot
directly switch device to ``GPU`` for inference on integrated or
discrete Intel GPUs. In order to run inference on iGPU or dGPU with this
model, you will need to resize the inputs to this model to use a fixed
size and then try running the inference on ``GPU`` device.

.. code:: ipython3

    from openvino.runtime import Core
    
    # initialize openvino core
    core = Core()
    
    # read the model and corresponding weights from file
    model = core.read_model(model_path)
    
    # compile the model for CPU devices
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    # get output tensors
    output_key = compiled_model.output(0)

Input keys are the names of the input nodes and output keys contain
names of the output nodes of the network. In the case of GPT-Neo, we
have ``batch size`` and ``sequence length`` as inputs and
``batch size``, ``sequence length`` and ``vocab size`` as outputs.

Pre-Processing
--------------

NLP models often take a list of tokens as a standard input. A token is a
word or a part of a word mapped to an integer. To provide the proper
input, we use a vocabulary file to handle the mapping. So first let’s
load the vocabulary file.

Define tokenization
-------------------

.. code:: ipython3

    from typing import List, Tuple
    
    
    # this function converts text to tokens
    def tokenize(text: str) -> Tuple[List[int], List[int]]:
        """
        tokenize input text using GPT2 tokenizer
    
        Parameters:
          text, str - input text
        Returns:
          input_ids - np.array with input token ids
          attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
        """
    
        inputs = tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

``eos_token`` is special token, which means that generation is finished.
We store the index of this token in order to use this index as padding
at later stage.

.. code:: ipython3

    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.decode(eos_token_id)

Define Softmax layer
~~~~~~~~~~~~~~~~~~~~

A softmax function is used to convert top-k logits into a probability
distribution.

.. code:: ipython3

    import numpy as np
    
    
    def softmax(x : np.array) -> np.array:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        summation = e_x.sum(axis=-1, keepdims=True)
        return e_x / summation

Set the minimum sequence length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the minimum sequence length is not reached, the following code will
reduce the probability of the ``eos`` token occurring. This continues
the process of generating the next words.

.. code:: ipython3

    def process_logits(cur_length: int, scores: np.array, eos_token_id : int, min_length : int = 0) -> np.array:
        """
        Reduce probability for padded indices.
    
        Parameters:
          cur_length: Current length of input sequence.
          scores: Model output logits.
          eos_token_id: Index of end of string token in model vocab.
          min_length: Minimum length for applying postprocessing.
    
        Returns:
          Processed logits with reduced probability for padded indices.
        """
        if cur_length < min_length:
            scores[:, eos_token_id] = -float("inf")
        return scores

Top-K sampling
~~~~~~~~~~~~~~

In Top-K sampling, we filter the K most likely next words and
redistribute the probability mass among only those K next words.

.. code:: ipython3

    def get_top_k_logits(scores : np.array, top_k : int) -> np.array:
        """
        Perform top-k sampling on the logits scores.
    
        Parameters:
          scores: np.array, model output logits.
          top_k: int, number of elements with the highest probability to select.
    
        Returns:
          np.array, shape (batch_size, sequence_length, vocab_size),
            filtered logits scores where only the top-k elements with the highest
            probability are kept and the rest are replaced with -inf
        """
        filter_value = -float("inf")
        top_k = min(max(top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                     fill_value=filter_value).filled()
        return filtred_scores

Main Processing Function
~~~~~~~~~~~~~~~~~~~~~~~~

Generating the predicted sequence.

.. code:: ipython3

    def generate_sequence(input_ids : List[int], attention_mask : List[int], max_sequence_length : int = 128,
                          eos_token_id : int = eos_token_id, dynamic_shapes : bool = True) -> List[int]:
        """
        Generates a sequence of tokens using a pre-trained language model.
    
        Parameters:
          input_ids: np.array, tokenized input ids for model
          attention_mask: np.array, attention mask for model
          max_sequence_length: int, maximum sequence length for stopping iteration
          eos_token_id: int, index of the end-of-sequence token in the model's vocabulary
          dynamic_shapes: bool, whether to use dynamic shapes for inference or pad model input to max_sequence_length
    
        Returns:
          np.array, the predicted sequence of token ids
        """
        while True:
            cur_input_len = len(input_ids[0])
            if not dynamic_shapes:
                pad_len = max_sequence_length - cur_input_len
                model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
                model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
            else:
                model_input_ids = input_ids
                model_input_attention_mask = attention_mask
            outputs = compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[output_key]
            next_token_logits = outputs[:, cur_input_len - 1, :]
            # pre-process distribution
            next_token_scores = process_logits(cur_input_len,
                                               next_token_logits, eos_token_id)
            top_k = 20
            next_token_scores = get_top_k_logits(next_token_scores, top_k)
            # get next token id
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1,
                                           p=probs[0], replace=True)
            # break the loop if max length or end of text token is reached
            if cur_input_len == max_sequence_length or next_tokens[0] == eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)
        return input_ids

Inference with GPT-Neo/GPT-2
----------------------------

The ``text`` variable below is the input used to generate a predicted
sequence.

.. code:: ipython3

    import time
    if not model_name.value == "PersonaGPT (Converastional)":
        text = "Deep learning is a type of machine learning that uses neural networks"
        input_ids, attention_mask = tokenize(text)
    
        start = time.perf_counter()
        output_ids = generate_sequence(input_ids, attention_mask)
        end = time.perf_counter()
        output_text = " "
        # Convert IDs to words and make the sentence from it
        for i in output_ids[0]:
            output_text += tokenizer.batch_decode([i])[0]
        print(f"Generation took {end - start:.3f} s")
        print(f"Input Text:  {text}")
        print()
        print(f"{model_name.value}: {output_text}")
    else:
        print("Selected Model is PersonaGPT. Please select GPT-Neo or GPT-2 in the first cell to generate text sequences")


.. parsed-literal::

    Generation took 5.323 s
    Input Text:  Deep learning is a type of machine learning that uses neural networks
    
    GPT-Neo:  Deep learning is a type of machine learning that uses neural networks to learn new ways of conveying information. Although many people are trying to learn more about how to make a given decision, learning the right word or phrase and passing it on to another person is a common technique. This technique is called a ��learning agent.�� As a result, we often hear the word ��learn��, ��learn a��, ��learn a bad��, ��learn�� or ��learn a good word�� used to describe our thinking on the job. When you hear these words or phrases in


Conversation with PersonaGPT using OpenVINO™
============================================

User Input is tokenized with eos_token concatenated in the end. Model
input is tokenized text, which serves as initial condition for
generation, then logits from model inference result should be obtained
and token with the highest probability is selected using top-k sampling
strategy and joined to input sequence. The procedure repeats until end
of sequence token will be recived or specified maximum length is
reached. After that, decoding token ids to text using tokenized should
be applied.

The Generated response is added to the history with the eos_token at the
end. Further User Input is added to it and agin passed into the model.

Converse Function
-----------------

Wrapper on generate sequence function to support conversation

.. code:: ipython3

    def converse(input: str, history: List[int], eos_token: str = eos_token,
                 eos_token_id: int = eos_token_id) -> Tuple[str, List[int]]:
        """
        Converse with the Model.
    
        Parameters:
          input: Text input given by the User
          history: Chat History, ids of tokens of chat occured so far
          eos_token: end of sequence string
          eos_token_id: end of sequence index from vocab
        Returns:
          response: Text Response generated by the model
          history: Chat History, Ids of the tokens of chat occured so far,including the tokens of generated response
        """
    
        # Get Input Ids of the User Input
        new_user_input_ids, _ = tokenize(input + eos_token)
    
        # append the new user input tokens to the chat history, if history exists
        if len(history) == 0:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = np.concatenate([history, new_user_input_ids[0]])
            bot_input_ids = np.expand_dims(bot_input_ids, axis=0)
    
        # Create Attention Mask
        bot_attention_mask = np.ones_like(bot_input_ids)
    
        # Generate Response from the model
        history = generate_sequence(bot_input_ids, bot_attention_mask, max_sequence_length=1000)
    
        # Add the eos_token to mark end of sequence
        history = np.append(history[0], eos_token_id)
    
        # convert the tokens to text, and then split the responses into lines and retrieve the response from the Model
        response = ''.join(tokenizer.batch_decode(history)).split(eos_token)[-2]
        return response, history

Conversation Class
------------------

.. code:: ipython3

    class Conversation:
        def __init__(self):
            # Initialize Empty History
            self.history = []
            self.messages = []
    
        def chat(self, input_text):
            """
            Wrapper Over Converse Function.
            Parameters:
                input_text: Text input given by the User
            Returns:
                response: Text Response generated by the model
            """
            response, self.history = converse(input_text, self.history)
            self.messages.append(f"Person: {input_text}")
            self.messages.append(f"PersonaGPT: {response}")
            return response

Conversation with PersonaGPT
----------------------------

This notebook provides two styles of inference, Plain and Interactive.
The style of inference can be selected in the next cell.

.. code:: ipython3

    style = {'description_width': 'initial'}
    interactive_mode = widgets.Select(
        options=['Plain', 'Interactive'],
        value='Plain',
        description='Inference Style:',
        disabled=False
    )
    
    widgets.VBox([interactive_mode])




.. parsed-literal::

    VBox(children=(Select(description='Inference Style:', options=('Plain', 'Interactive'), value='Plain'),))



.. code:: ipython3

    if model_name.value == "PersonaGPT (Converastional)":
        if interactive_mode.value == 'Plain':
            conversation = Conversation()
            user_prompt = None
            pre_written_prompts = ["Hi,How are you?", "What are you doing?", "I like to dance,do you?", "Can you recommend me some books?"]
            # Number of responses generated by model
            n_prompts = 10
            for i in range(n_prompts):
                # Uncomment for taking User Input
                # user_prompt = input()
                if not user_prompt:
                    user_prompt = pre_written_prompts[i % len(pre_written_prompts)]
                conversation.chat(user_prompt)
                print(conversation.messages[-2])
                print(conversation.messages[-1])
                user_prompt = None
        else:
            def add_text(history, text):
                history = history + [(text, None)]
                return history, ""
    
            conversation = Conversation()
    
            def bot(history):
                conversation.chat(history[-1][0])
                response = conversation.messages[-1]
                history[-1][1] = response
                return history
    
            with Blocks() as demo:
                chatbot = Chatbot([], elem_id="chatbot").style()
    
                with Row():
                    with Column():
                        txt = Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter, or upload an image",
                        ).style(container=False)
    
                txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                    bot, chatbot, chatbot
                )
    
            demo.launch()
    else:
        print("Selected Model is not PersonaGPT, Please select PersonaGPT in the first cell to have a conversation")


.. parsed-literal::

    Selected Model is not PersonaGPT, Please select PersonaGPT in the first cell to have a conversation

