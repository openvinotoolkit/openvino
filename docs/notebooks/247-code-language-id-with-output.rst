Programming Language Classification with OpenVINO
=================================================

Overview
--------

This tutorial will be divided in 2 parts:

1. Create a simple inference pipeline with a pre-trained model using the OpenVINO™ IR format.
2. Conduct `post-training quantization <https://docs.openvino.ai/2023.1/ptq_introduction.html>`__ 
   on a pre-trained model using Hugging Face Optimum and benchmark performance.

Feel free to use the notebook outline in Jupyter or your IDE for easy
navigation.

Introduction
------------

Task
~~~~

**Programming language classification** is the task of identifying which
programming language is used in an arbitrary code snippet. This can be
useful to label new data to include in a dataset, and potentially serve
as an intermediary step when input snippets need to be process based on
their programming language.

It is a relatively easy machine learning task given that each
programming language has its own formal symbols, syntax, and grammar.
However, there are some potential edge cases: - **Ambiguous short
snippets**: For example, TypeScript is a superset of JavaScript, meaning
it does everything JavaScript can and more. For a short input snippet,
it might be impossible to distinguish between the two. Given we know
TypeScript is a superset, and the model doesn’t, we should default to
classifying the input as JavaScript in a post-processing step. -
**Nested programming languages**: Some languages are typically used in
tandem. For example, most HTML contains CSS and JavaScript, and it is
not uncommon to see SQL nested in other scripting languages. For such
input, it is unclear what the expected output class should be. -
**Evolving programming language**: Even though programming languages are
formal, their symbols, syntax, and grammar can be revised and updated.
For example, the walrus operator (``:=``) was a symbol distinctively
used in Golang, but was later introduced in Python 3.8.

Model
~~~~~

The classification model that will be used in this notebook is
`CodeBERTa-language-id <https://huggingface.co/huggingface/CodeBERTa-language-id>`__
by HuggingFace. This model was fine-tuned from the masked language
modeling model
`CodeBERTa-small-v1 <https://huggingface.co/huggingface/CodeBERTa-small-v1>`__
trained on the
`CodeSearchNet <https://huggingface.co/huggingface/CodeBERTa-small-v1>`__
dataset (Husain, 2019).

It supports 6 programming languages: - Go - Java - JavaScript - PHP -
Python - Ruby

Part 1: Inference pipeline with OpenVINO
----------------------------------------

For this section, we will use the `HuggingFace Optimum <https://huggingface.co/docs/optimum/index>`__ library, which
aims to optimize inference on specific hardware and integrates with the
OpenVINO toolkit. The code will be very similar to the 
`HuggingFace Transformers <https://huggingface.co/docs/transformers/index>`__, but
will allow to automatically convert models to the OpenVINO™ IR format.

Install prerequisites
~~~~~~~~~~~~~~~~~~~~~

First, complete the `repository installation steps <../notebooks_installation.html>`__.

Then, the following cell will install: - HuggingFace Optimum with
OpenVINO support - HuggingFace Evaluate to benchmark results

.. code:: ipython3

    !pip install -q "diffusers>=0.17.1" "openvino-dev>=2023.0.0" "nncf>=2.5.0" "gradio" "onnx>=1.11.0" "onnxruntime>=1.14.0" "optimum-intel>=1.9.1" "transformers>=4.31.0" "evaluate"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 23.3 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pytorch-lightning 1.6.5 requires protobuf<=3.20.1, but you have protobuf 4.24.0 which is incompatible.
    

Imports
~~~~~~~

The import ``OVModelForSequenceClassification`` from Optimum is
equivalent to ``AutoModelForSequenceClassification`` from Transformers

.. code:: ipython3

    from functools import partial
    from pathlib import Path
    
    import pandas as pd
    from datasets import load_dataset, Dataset
    import evaluate
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from optimum.intel import OVModelForSequenceClassification  
    from optimum.intel.openvino import OVConfig, OVQuantizer
    from huggingface_hub.utils import RepositoryNotFoundError


.. parsed-literal::

    2023-08-16 01:03:40.095980: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-08-16 01:03:40.129769: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-08-16 01:03:40.709247: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Setting up HuggingFace cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resources from HuggingFace will be downloaded in the local folder
``./model`` (next to this notebook) instead of the device global cache
for easy cleanup. Learn more
`here <https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup>`__.

.. code:: ipython3

    MODEL_NAME = "CodeBERTa-language-id"
    MODEL_ID = f"huggingface/{MODEL_NAME}"
    MODEL_LOCAL_PATH = Path("./model").joinpath(MODEL_NAME)

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    from openvino.runtime import Core
    
    core = Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Download resources
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # try to load resources locally
    try:
        model = OVModelForSequenceClassification.from_pretrained(MODEL_LOCAL_PATH, device=device.value)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
        print(f"Loaded resources from local path: {MODEL_LOCAL_PATH.absolute()}")
    
    # if not found, download from HuggingFace Hub then save locally
    except (RepositoryNotFoundError, OSError):
        print("Downloading resources from HuggingFace Hub")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(MODEL_LOCAL_PATH)
    
        # export=True is needed to convert the PyTorch model to OpenVINO
        model = OVModelForSequenceClassification.from_pretrained(MODEL_ID, export=True, device=device.value)
        model.save_pretrained(MODEL_LOCAL_PATH)
        print(f"Ressources cached locally at: {MODEL_LOCAL_PATH.absolute()}")


.. parsed-literal::

    Downloading resources from HuggingFace Hub


.. parsed-literal::

    Framework not specified. Using pt to export to ONNX.
    Some weights of the model checkpoint at huggingface/CodeBERTa-language-id were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False
    Compiling the model...
    Set CACHE_DIR to /tmp/tmpsl_db7y_/model_cache


.. parsed-literal::

    Ressources cached locally at: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/notebooks/247-code-language-id/model/CodeBERTa-language-id


Create inference pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    code_classification_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


.. parsed-literal::

    Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
    pip install xformers.


Inference on new input
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # change input snippet to test model
    input_snippet = "df['speed'] = df.distance / df.time"
    output = code_classification_pipe(input_snippet)
    
    print(f"Input snippet:\n  {input_snippet}\n")
    print(f"Predicted label: {output[0]['label']}")
    print(f"Predicted score: {output[0]['score']:.2}")


.. parsed-literal::

    Input snippet:
      df['speed'] = df.distance / df.time
    
    Predicted label: python
    Predicted score: 0.81


Part 2: OpenVINO post-training quantization with HuggingFace Optimum
--------------------------------------------------------------------

In this section, we will quantize a trained model. At a high-level, this
process consists of using lower precision numbers in the model, which
results in a smaller model size and faster inference at the cost of a
potential marginal performance degradation. `Learn more <https://docs.openvino.ai/2023.1/ptq_introduction.html>`__.

The HuggingFace Optimum library supports post-training quantization for
OpenVINO. `Learn more <https://huggingface.co/docs/optimum/main/en/intel/index>`__.

Define constants and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    QUANTIZED_MODEL_LOCAL_PATH = MODEL_LOCAL_PATH.with_name(f"{MODEL_NAME}-quantized")
    DATASET_NAME = "code_search_net"
    LABEL_MAPPING = {"go": 0, "java": 1, "javascript": 2, "php": 3, "python": 4, "ruby": 5}
    
    
    def preprocess_function(examples: dict, tokenizer):
        """Preprocess inputs by tokenizing the `func_code_string` column"""
        return tokenizer(
            examples["func_code_string"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    
    
    def map_labels(example: dict) -> dict:
        """Convert string labels to integers"""
        label_mapping = {"go": 0, "java": 1, "javascript": 2, "php": 3, "python": 4, "ruby": 5}
        example["language"] = label_mapping[example["language"]]
        return example 
    
    
    def get_dataset_sample(dataset_split: str, num_samples: int) -> Dataset:
        """Create a sample with equal representation of each class without downloading the entire data"""
        labels = ["go", "java", "javascript", "php", "python", "ruby"]
        example_per_label = num_samples // len(labels)
    
        examples = []
        for label in labels:
            subset = load_dataset("code_search_net", split=dataset_split, name=label, streaming=True)
            subset = subset.map(map_labels)
            examples.extend([example for example in subset.shuffle().take(example_per_label)])
        
        return Dataset.from_list(examples)

Load resources
~~~~~~~~~~~~~~

.. note::

   The base model is loaded using ``AutoModelForSequenceClassification`` from ``Transformers``.


.. code:: ipython3

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    
    quantizer = OVQuantizer.from_pretrained(base_model)
    quantization_config = OVConfig()


.. parsed-literal::

    Some weights of the model checkpoint at huggingface/CodeBERTa-language-id were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


Load calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~

The ``get_dataset_sample()`` function will sample up to ``num_samples``,
with an equal number of examples across the 6 programming languages.

.. note::

   Uncomment the method below to download and use the full dataset (5+ Gb).


.. code:: ipython3

    calibration_sample = get_dataset_sample(dataset_split="train", num_samples=120)
    calibration_sample = calibration_sample.map(partial(preprocess_function, tokenizer=tokenizer))
    
    # calibration_sample = quantizer.get_calibration_dataset(
    #     DATASET_NAME,
    #     preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
    #     num_samples=120,
    #     dataset_split="train",
    #     preprocess_batch=True,
    # )


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)



.. parsed-literal::

    Map:   0%|          | 0/120 [00:00<?, ? examples/s]


Quantize model
~~~~~~~~~~~~~~

Calling ``quantizer.quantize(...)`` will iterate through the calibration
dataset to quantize and save the model

.. code:: ipython3

    quantizer.quantize(
        quantization_config=quantization_config,
        calibration_dataset=calibration_sample,
        save_directory=QUANTIZED_MODEL_LOCAL_PATH,
    )


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 12 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/NNCFEmbedding[token_type_embeddings]/embedding_0
    INFO:nncf:Not adding activation input quantizer for operation: 11 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/NNCFEmbedding[word_embeddings]/embedding_0
    INFO:nncf:Not adding activation input quantizer for operation: 3 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/ne_0
    INFO:nncf:Not adding activation input quantizer for operation: 4 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/int_0
    INFO:nncf:Not adding activation input quantizer for operation: 5 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/cumsum_0
    INFO:nncf:Not adding activation input quantizer for operation: 13 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/__add___2
    INFO:nncf:Not adding activation input quantizer for operation: 6 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/type_as_0
    INFO:nncf:Not adding activation input quantizer for operation: 7 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 8 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/__mul___0
    INFO:nncf:Not adding activation input quantizer for operation: 9 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/long_0
    INFO:nncf:Not adding activation input quantizer for operation: 10 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/__add___1
    INFO:nncf:Not adding activation input quantizer for operation: 14 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/NNCFEmbedding[position_embeddings]/embedding_0
    INFO:nncf:Not adding activation input quantizer for operation: 15 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/__iadd___0
    INFO:nncf:Not adding activation input quantizer for operation: 16 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 17 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEmbeddings[embeddings]/Dropout[dropout]/dropout_0
    INFO:nncf:Not adding activation input quantizer for operation: 30 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 33 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 39 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 40 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 45 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 46 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[0]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 59 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 62 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 68 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 69 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 74 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 75 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[1]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 88 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 91 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 97 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 98 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 103 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 104 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[2]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 117 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 120 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 126 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 127 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 132 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 133 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[3]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 146 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 149 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 155 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 156 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 161 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 162 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[4]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 175 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaAttention[attention]/RobertaSelfAttention[self]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 178 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaAttention[attention]/RobertaSelfAttention[self]/matmul_1
    INFO:nncf:Not adding activation input quantizer for operation: 184 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaAttention[attention]/RobertaSelfOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 185 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaAttention[attention]/RobertaSelfOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Not adding activation input quantizer for operation: 190 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaOutput[output]/__add___0
    INFO:nncf:Not adding activation input quantizer for operation: 191 RobertaForSequenceClassification/RobertaModel[roberta]/RobertaEncoder[encoder]/ModuleList[layer]/RobertaLayer[5]/RobertaOutput[output]/NNCFLayerNorm[LayerNorm]/layer_norm_0
    INFO:nncf:Collecting tensor statistics |█               | 33 / 300
    INFO:nncf:Collecting tensor statistics |███             | 66 / 300
    INFO:nncf:Collecting tensor statistics |█████           | 99 / 300
    INFO:nncf:Compiling and loading torch extension: quantized_functions_cpu...
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    INFO:nncf:Finished loading torch extension: quantized_functions_cpu


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-475/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/torch/nncf_network.py:938: FutureWarning: Old style of accessing NNCF-specific attributes and methods on NNCFNetwork objects is deprecated. Access the NNCF-specific attrs through the NNCFInterface, which is set up as an `nncf` attribute on the compressed model object.
    For instance, instead of `compressed_model.get_graph()` you should now write `compressed_model.nncf.get_graph()`.
    The old style will be removed after NNCF v2.5.0
      warning_deprecated(
    Using framework PyTorch: 1.13.1+cpu
    Overriding 1 configuration item(s)
    	- use_cache -> False


.. parsed-literal::

    WARNING:nncf:You are setting `forward` on an NNCF-processed model object.
    NNCF relies on custom-wrapping the `forward` call in order to function properly.
    Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behaviour.
    If you need to replace the underlying forward function of the original model so that NNCF should be using that instead of the original forward function that NNCF saved during the compressed model creation, you can do this by calling:
    model.nncf.set_original_unbound_forward(fn)
    if `fn` has an unbound 0-th `self` argument, or
    with model.nncf.temporary_bound_original_forward(fn): ...
    if `fn` already had 0-th `self` argument bound or never had it in the first place.
    WARNING:nncf:You are setting `forward` on an NNCF-processed model object.
    NNCF relies on custom-wrapping the `forward` call in order to function properly.
    Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behaviour.
    If you need to replace the underlying forward function of the original model so that NNCF should be using that instead of the original forward function that NNCF saved during the compressed model creation, you can do this by calling:
    model.nncf.set_original_unbound_forward(fn)
    if `fn` has an unbound 0-th `self` argument, or
    with model.nncf.temporary_bound_original_forward(fn): ...
    if `fn` already had 0-th `self` argument bound or never had it in the first place.


.. parsed-literal::

    Configuration saved in model/CodeBERTa-language-id-quantized/openvino_config.json


Load quantized model
~~~~~~~~~~~~~~~~~~~~

.. note::

   The argument ``export=True`` is not required since the quantized model is already in the OpenVINO format.

.. code:: ipython3

    quantized_model = OVModelForSequenceClassification.from_pretrained(QUANTIZED_MODEL_LOCAL_PATH, device=device.value)
    quantized_code_classification_pipe = pipeline("text-classification", model=quantized_model, tokenizer=tokenizer)


.. parsed-literal::

    Compiling the model...
    Set CACHE_DIR to model/CodeBERTa-language-id-quantized/model_cache


Inference on new input using quantized model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    input_snippet = "df['speed'] = df.distance / df.time"
    output = quantized_code_classification_pipe(input_snippet)
    
    print(f"Input snippet:\n  {input_snippet}\n")
    print(f"Predicted label: {output[0]['label']}")
    print(f"Predicted score: {output[0]['score']:.2}")


.. parsed-literal::

    Input snippet:
      df['speed'] = df.distance / df.time
    
    Predicted label: python
    Predicted score: 0.82


Load evaluation set
~~~~~~~~~~~~~~~~~~~

.. note::

   Uncomment the method below to download and use the full dataset (5+ Gb).


.. code:: ipython3

    validation_sample = get_dataset_sample(dataset_split="validation", num_samples=120)
    
    # validation_sample = load_dataset(DATASET_NAME, split="validation")

Evaluate model
~~~~~~~~~~~~~~

.. code:: ipython3

    # This class is needed due to a current limitation of the Evaluate library with multiclass metrics
    # ref: https://discuss.huggingface.co/t/combining-metrics-for-multiclass-predictions-evaluations/21792/16
    class ConfiguredMetric:
        def __init__(self, metric, *metric_args, **metric_kwargs):
            self.metric = metric
            self.metric_args = metric_args
            self.metric_kwargs = metric_kwargs
        
        def add(self, *args, **kwargs):
            return self.metric.add(*args, **kwargs)
        
        def add_batch(self, *args, **kwargs):
            return self.metric.add_batch(*args, **kwargs)
    
        def compute(self, *args, **kwargs):
            return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)
    
        @property
        def name(self):
            return self.metric.name
    
        def _feature_names(self):
            return self.metric._feature_names()

First, an ``Evaluator`` object for ``text-classification`` and a set of
``EvaluationModule`` are instantiated. Then, the evaluator
``.compute()`` method is called on both the base
``code_classification_pipe`` and the quantized
``quantized_code_classification_pipeline``. Finally, results are
displayed.

.. code:: ipython3

    code_classification_evaluator = evaluate.evaluator("text-classification")
    # instantiate an object that can contain multiple `evaluate` metrics
    metrics = evaluate.combine([
        ConfiguredMetric(evaluate.load('f1'), average='macro'),
    ])
    
    base_results = code_classification_evaluator.compute(
        model_or_pipeline=code_classification_pipe,
        data=validation_sample,
        input_column="func_code_string",
        label_column="language",
        label_mapping=LABEL_MAPPING,
        metric=metrics,
    )
    
    quantized_results = code_classification_evaluator.compute(
        model_or_pipeline=quantized_code_classification_pipe,
        data=validation_sample,
        input_column="func_code_string",
        label_column="language",
        label_mapping=LABEL_MAPPING,
        metric=metrics,
    )
    
    results_df = pd.DataFrame.from_records([base_results, quantized_results], index=["base", "quantized"])
    results_df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>f1</th>
          <th>total_time_in_seconds</th>
          <th>samples_per_second</th>
          <th>latency_in_seconds</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>base</th>
          <td>1.0</td>
          <td>2.322593</td>
          <td>51.666392</td>
          <td>0.019355</td>
        </tr>
        <tr>
          <th>quantized</th>
          <td>1.0</td>
          <td>2.647466</td>
          <td>45.326357</td>
          <td>0.022062</td>
        </tr>
      </tbody>
    </table>
    </div>



Additional resources
--------------------

-  `Grammatical Error Correction with
   OpenVINO <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/214-grammar-correction/214-grammar-correction.ipynb>`__
-  `Quantize a Hugging Face Question-Answering Model with
   OpenVINO <https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb>`__\ \*\*

Clean up
--------

Uncomment and run cell below to delete all resources cached locally in
./model

.. code:: ipython3

    # import os
    # import shutil
    
    # try:
    #     shutil.rmtree(path=QUANTIZED_MODEL_LOCAL_PATH)
    #     shutil.rmtree(path=MODEL_LOCAL_PATH)
    #     os.remove(path="./compressed_graph.dot")
    #     os.remove(path="./original_graph.dot")
    # except FileNotFoundError:
    #     print("Directory was already deleted")
