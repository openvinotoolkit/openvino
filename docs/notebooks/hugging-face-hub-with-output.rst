Hugging Face Model Hub with OpenVINO™
=======================================

The Hugging Face (HF) `Model Hub <https://huggingface.co/models>`__ is a
central repository for pre-trained deep learning models. It allows
exploration and provides access to thousands of models for a wide range
of tasks, including text classification, question answering, and image
classification. Hugging Face provides Python packages that serve as APIs
and tools to easily download and fine tune state-of-the-art pretrained
models, namely
`transformers <https://github.com/huggingface/transformers>`__ and
`diffusers <https://github.com/huggingface/diffusers>`__ packages.

|image0|

Throughout this notebook we will learn: 1. How to load a HF pipeline
using the ``transformers`` package and then convert it to OpenVINO. 2.
How to load the same pipeline using Optimum Intel package.


**Table of contents:**


-  `Converting a Model from the HF Transformers
   Package <#converting-a-model-from-the-hf-transformers-package>`__

   -  `Installing Requirements <#installing-requirements>`__
   -  `Imports <#imports>`__
   -  `Initializing a Model Using the HF Transformers
      Package <#initializing-a-model-using-the-hf-transformers-package>`__
   -  `Original Model inference <#original-model-inference>`__
   -  `Converting the Model to OpenVINO IR
      format <#converting-the-model-to-openvino-ir-format>`__
   -  `Converted Model Inference <#converted-model-inference>`__

-  `Converting a Model Using the Optimum Intel
   Package <#converting-a-model-using-the-optimum-intel-package>`__

   -  `Install Requirements for
      Optimum <#install-requirements-for-optimum>`__
   -  `Import Optimum <#import-optimum>`__
   -  `Initialize and Convert the Model Automatically using OVModel
      class <#initialize-and-convert-the-model-automatically-using-ovmodel-class>`__
   -  `Convert model using Optimum CLI
      interface <#convert-model-using-optimum-cli-interface>`__
   -  `The Optimum Model Inference <#the-optimum-model-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://github.com/huggingface/optimum-intel/raw/main/readme_logo.png

Converting a Model from the HF Transformers Package
---------------------------------------------------



Hugging Face transformers package provides API for initializing a model
and loading a set of pre-trained weights using the model text handle.
Discovering a desired model name is straightforward with `HF website’s
Models page <https://huggingface.co/models>`__, one can choose a model
solving a particular machine learning problem and even sort the models
by popularity and novelty.

Installing Requirements
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "transformers>=4.33.0" "torch>=2.1.0"
    %pip install -q ipywidgets
    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Imports
~~~~~~~



.. code:: ipython3

    from pathlib import Path
    
    import numpy as np
    import torch
    
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer

Initializing a Model Using the HF Transformers Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



We will use `roberta text sentiment
classification <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest>`__
model in our example, it is a transformer-based encoder model pretrained
in a special way, please refer to the model card to learn more.

Following the instructions on the model page, we use
``AutoModelForSequenceClassification`` to initialize the model and
perform inference with it. To find more information on HF pipelines and
model initialization please refer to `HF
tutorials <https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt#behind-the-pipeline>`__.

.. code:: ipython3

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, return_dict=True)
    
    # The torchscript=True flag is used to ensure the model outputs are tuples
    # instead of ModelOutput (which causes JIT errors).
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, torchscript=True)


.. parsed-literal::

    Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


Original Model inference
~~~~~~~~~~~~~~~~~~~~~~~~



Let’s do a classification of a simple prompt below.

.. code:: ipython3

    text = "HF models run perfectly with OpenVINO!"
    
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0]
    scores = torch.softmax(scores, dim=0).numpy(force=True)
    
    
    def print_prediction(scores):
        for i, descending_index in enumerate(scores.argsort()[::-1]):
            label = model.config.id2label[descending_index]
            score = np.round(float(scores[descending_index]), 4)
            print(f"{i+1}) {label} {score}")
    
    
    print_prediction(scores)


.. parsed-literal::

    1) positive 0.9485
    2) neutral 0.0484
    3) negative 0.0031


Converting the Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the OpenVINO `Model
conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
to convert the model (this one is implemented in PyTorch) to OpenVINO
Intermediate Representation (IR).

Note how we reuse our real ``encoded_input``, passing it to the
``ov.convert_model`` function. It will be used for model tracing.

.. code:: ipython3

    import openvino as ov
    
    save_model_path = Path("./models/model.xml")
    
    if not save_model_path.exists():
        ov_model = ov.convert_model(model, example_input=dict(encoded_input))
        ov.save_model(ov_model, save_model_path)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


Converted Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~



First, we pick a device to do the model inference

.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



OpenVINO model IR must be compiled for a specific device prior to the
model inference.

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    
    compiled_model = core.compile_model(save_model_path, device.value)
    
    # Compiled model call is performed using the same parameters as for the original model
    scores_ov = compiled_model(encoded_input.data)[0]
    
    scores_ov = torch.softmax(torch.tensor(scores_ov[0]), dim=0).detach().numpy()
    
    print_prediction(scores_ov)


.. parsed-literal::

    1) positive 0.9483
    2) neutral 0.0485
    3) negative 0.0031


Note the prediction of the converted model match exactly the one of the
original model.

This is a rather simple example as the pipeline includes just one
encoder model. Contemporary state of the art pipelines often consist of
several model, feel free to explore other OpenVINO tutorials: 1. `Stable
Diffusion v2 <stable-diffusion-v2-with-output.html>`__ 2. `Zero-shot Image
Classification with OpenAI
CLIP <clip-zero-shot-image-classification-with-output.html>`__ 3. `Controllable Music
Generation with MusicGen <music-generation-with-output.html>`__

The workflow for the ``diffusers`` package is exactly the same. The
first example in the list above relies on the ``diffusers``.

Converting a Model Using the Optimum Intel Package
--------------------------------------------------



Optimum Intel is the interface between the Transformers and
Diffusers libraries and the different tools and libraries provided by
Intel to accelerate end-to-end pipelines on Intel architectures.

Among other use cases, Optimum Intel provides a simple interface to
optimize your Transformers and Diffusers models, convert them to the
OpenVINO Intermediate Representation (IR) format and run inference using
OpenVINO Runtime.

Install Requirements for Optimum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git" onnx


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Import Optimum
~~~~~~~~~~~~~~



Documentation for Optimum Intel states: >You can now easily perform
inference with OpenVINO Runtime on a variety of Intel processors (see
the full list of supported devices). For that, just replace the
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

You can find more information in `Optimum Intel
documentation <https://huggingface.co/docs/optimum/intel/inference>`__.

.. code:: ipython3

    from optimum.intel.openvino import OVModelForSequenceClassification


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    2024-10-23 01:23:29.312809: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 01:23:29.347879: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 01:23:29.943155: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Initialize and Convert the Model Automatically using OVModel class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To load a Transformers model and convert it to the OpenVINO format on
the fly, you can set ``export=True`` when loading your model. The model
can be saved in OpenVINO format using ``save_pretrained`` method and
specifying a directory for storing the model as an argument. For the
next usage, you can avoid the conversion step and load the saved early
model from disk using ``from_pretrained`` method without export
specification. We also specified ``device`` parameter for compiling the
model on the specific device, if not provided, the default device will
be used. The device can be changed later in runtime using
``model.to(device)``, please note that it may require some time for
model compilation on a newly selected device. In some cases, it can be
useful to separate model initialization and compilation, for example, if
you want to reshape the model using ``reshape`` method, you can postpone
compilation, providing the parameter ``compile=False`` into
``from_pretrained`` method, compilation can be performed manually using
``compile`` method or will be performed automatically during first
inference run.

.. code:: ipython3

    model = OVModelForSequenceClassification.from_pretrained(MODEL, export=True, device=device.value)
    
    # The save_pretrained() method saves the model weights to avoid conversion on the next load.
    model.save_pretrained("./models/optimum_model")


.. parsed-literal::

    Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


Convert model using Optimum CLI interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Alternatively, you can use the Optimum CLI interface for converting
models (supported starting optimum-intel 1.12 version). General command
format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where task is task to export the model for, if not specified, the task
will be auto-inferred based on the model. Available tasks depend on the
model, but are among: [‘default’, ‘fill-mask’, ‘text-generation’,
‘text2text-generation’, ‘text-classification’, ‘token-classification’,
‘multiple-choice’, ‘object-detection’, ‘question-answering’,
‘image-classification’, ‘image-segmentation’, ‘masked-im’,
‘semantic-segmentation’, ‘automatic-speech-recognition’,
‘audio-classification’, ‘audio-frame-classification’,
‘automatic-speech-recognition’, ‘audio-xvector’, ‘image-to-text’,
‘stable-diffusion’, ‘zero-shot-object-detection’]. For decoder models,
use ``xxx-with-past`` to export the model using past key values in the
decoder.

You can find a mapping between tasks and model classes in Optimum
TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.

Additionally, you can specify weights compression using
``--weight-format`` argument with one of following options: ``fp32``,
``fp16``, ``int8`` and ``int4``. Fro int8 and int4 nncf will be used for
weight compression.

Full list of supported arguments available via ``--help``

.. code:: ipython3

    !optimum-cli export openvino --help


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    2024-10-23 01:23:44.450300: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    usage: optimum-cli export openvino [-h] -m MODEL [--task TASK]
                                       [--framework {pt,tf}] [--trust-remote-code]
                                       [--weight-format {fp32,fp16,int8,int4,mxfp4}]
                                       [--library {transformers,diffusers,timm,sentence_transformers,open_clip}]
                                       [--cache_dir CACHE_DIR]
                                       [--pad-token-id PAD_TOKEN_ID]
                                       [--ratio RATIO] [--sym]
                                       [--group-size GROUP_SIZE]
                                       [--dataset DATASET] [--all-layers] [--awq]
                                       [--scale-estimation] [--gptq]
                                       [--sensitivity-metric SENSITIVITY_METRIC]
                                       [--num-samples NUM_SAMPLES]
                                       [--disable-stateful]
                                       [--disable-convert-tokenizer]
                                       output
    
    optional arguments:
      -h, --help            show this help message and exit
    
    Required arguments:
      -m MODEL, --model MODEL
                            Model ID on huggingface.co or path on disk to load
                            model from.
      output                Path indicating the directory where to store the
                            generated OV model.
    
    Optional arguments:
      --task TASK           The task to export the model for. If not specified,
                            the task will be auto-inferred based on the model.
                            Available tasks depend on the model, but are among:
                            ['audio-xvector', 'text-to-audio', 'fill-mask',
                            'feature-extraction', 'text-generation', 'zero-shot-
                            image-classification', 'text-to-image', 'text2text-
                            generation', 'zero-shot-object-detection', 'automatic-
                            speech-recognition', 'text-classification', 'semantic-
                            segmentation', 'masked-im', 'image-to-text',
                            'sentence-similarity', 'object-detection',
                            'inpainting', 'audio-frame-classification', 'image-to-
                            image', 'token-classification', 'question-answering',
                            'audio-classification', 'image-segmentation', 'image-
                            classification', 'mask-generation', 'multiple-choice',
                            'depth-estimation']. For decoder models, use `xxx-
                            with-past` to export the model using past key values
                            in the decoder.
      --framework {pt,tf}   The framework to use for the export. If not provided,
                            will attempt to use the local checkpoint's original
                            framework or what is available in the environment.
      --trust-remote-code   Allows to use custom code for the modeling hosted in
                            the model repository. This option should only be set
                            for repositories you trust and in which you have read
                            the code, as it will execute on your local machine
                            arbitrary code present in the model repository.
      --weight-format {fp32,fp16,int8,int4,mxfp4}
                            The weight format of the exported model.
      --library {transformers,diffusers,timm,sentence_transformers,open_clip}
                            The library used to load the model before export. If
                            not provided, will attempt to infer the local
                            checkpoint's library
      --cache_dir CACHE_DIR
                            The path to a directory in which the downloaded model
                            should be cached if the standard cache should not be
                            used.
      --pad-token-id PAD_TOKEN_ID
                            This is needed by some models, for some tasks. If not
                            provided, will attempt to use the tokenizer to guess
                            it.
      --ratio RATIO         A parameter used when applying 4-bit quantization to
                            control the ratio between 4-bit and 8-bit
                            quantization. If set to 0.8, 80% of the layers will be
                            quantized to int4 while 20% will be quantized to int8.
                            This helps to achieve better accuracy at the sacrifice
                            of the model size and inference latency. Default value
                            is 1.0.
      --sym                 Whether to apply symmetric quantization
      --group-size GROUP_SIZE
                            The group size to use for quantization. Recommended
                            value is 128 and -1 uses per-column quantization.
      --dataset DATASET     The dataset used for data-aware compression or
                            quantization with NNCF. You can use the one from the
                            list ['wikitext2','c4','c4-new'] for language models
                            or ['conceptual_captions','laion/220k-GPT4Vision-
                            captions-from-LIVIS','laion/filtered-wit'] for
                            diffusion models.
      --all-layers          Whether embeddings and last MatMul layers should be
                            compressed to INT4. If not provided an weight
                            compression is applied, they are compressed to INT8.
      --awq                 Whether to apply AWQ algorithm. AWQ improves
                            generation quality of INT4-compressed LLMs, but
                            requires additional time for tuning weights on a
                            calibration dataset. To run AWQ, please also provide a
                            dataset argument. Note: it's possible that there will
                            be no matching patterns in the model to apply AWQ, in
                            such case it will be skipped.
      --scale-estimation    Indicates whether to apply a scale estimation
                            algorithm that minimizes the L2 error between the
                            original and compressed layers. Providing a dataset is
                            required to run scale estimation. Please note, that
                            applying scale estimation takes additional memory and
                            time.
      --gptq                Indicates whether to apply GPTQ algorithm that
                            optimizes compressed weights in a layer-wise fashion
                            to minimize the difference between activations of a
                            compressed and original layer. Please note, that
                            applying GPTQ takes additional memory and time.
      --sensitivity-metric SENSITIVITY_METRIC
                            The sensitivity metric for assigning quantization
                            precision to layers. Can be one of the following:
                            ['weight_quantization_error',
                            'hessian_input_activation',
                            'mean_activation_variance', 'max_activation_variance',
                            'mean_activation_magnitude'].
      --num-samples NUM_SAMPLES
                            The maximum number of samples to take from the dataset
                            for quantization.
      --disable-stateful    Disable stateful converted models, stateless models
                            will be generated instead. Stateful models are
                            produced by default when this key is not used. In
                            stateful models all kv-cache inputs and outputs are
                            hidden in the model and are not exposed as model
                            inputs and outputs. If --disable-stateful option is
                            used, it may result in sub-optimal inference
                            performance. Use it when you intentionally want to use
                            a stateless model, for example, to be compatible with
                            existing OpenVINO native inference code that expects
                            kv-cache inputs and outputs in the model.
      --disable-convert-tokenizer
                            Do not add converted tokenizer and detokenizer
                            OpenVINO models.


The command line export for model from example above with FP16 weights
compression:

.. code:: ipython3

    !optimum-cli export openvino --model $MODEL --task text-classification --weight-format fp16 models/optimum_model/fp16


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    2024-10-23 01:23:49.987001: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Tokenizer won't be converted.


After export, model will be available in the specified directory and can
be loaded using the same OVModelForXXX class.

.. code:: ipython3

    model = OVModelForSequenceClassification.from_pretrained("models/optimum_model/fp16", device=device.value)

There are some models in the Hugging Face Models Hub, that are already
converted and ready to run! You can filter those models out by library
name, just type OpenVINO, or follow `this
link <https://huggingface.co/models?library=openvino&sort=trending>`__.

The Optimum Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~



Model inference is exactly the same as for the original model!

.. code:: ipython3

    output = model(**encoded_input)
    scores = output[0][0]
    scores = torch.softmax(scores, dim=0).numpy(force=True)
    
    print_prediction(scores)


.. parsed-literal::

    1) positive 0.9483
    2) neutral 0.0485
    3) negative 0.0031


You can find more examples of using Optimum Intel here: 1. `Accelerate
Inference of Sparse Transformer
Models <sparsity-optimization-with-output.html>`__ 2.
`Grammatical Error Correction with
OpenVINO <grammar-correction-with-output.html>`__ 3. `Stable
Diffusion v2.1 using Optimum-Intel
OpenVINO <stable-diffusion-v2-with-output.html>`__
4. `Image generation with Stable Diffusion
XL <stable-diffusion-xl-with-output.html>`__ 5. `Instruction following using
Databricks Dolly 2.0 <dolly-2-instruction-following-with-output.html>`__ 6. `Create
LLM-powered Chatbot using OpenVINO <llm-chatbot-with-output.html>`__ 7. `Document
Visual Question Answering Using Pix2Struct and
OpenVINO <pix2struct-docvqa-with-output.html>`__ 8. `Automatic speech recognition
using Distil-Whisper and OpenVINO <distil-whisper-asr-with-output.html>`__
