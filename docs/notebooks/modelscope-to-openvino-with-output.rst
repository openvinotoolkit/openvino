Convert models from ModelScope to OpenVINO
==========================================

.. image:: https://camo.githubusercontent.com/bbda58b4f77b80d9206e3410b533ca5a2582b81070e7dd283ee12fd0d442bd2b/68747470733a2f2f6d6f64656c73636f70652e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f6d6f64656c73636f70652e676966

`ModelScope <https://www.modelscope.cn/home>`__ is a
“Model-as-a-Service” (MaaS) platform that seeks to bring together most
advanced machine learning models from the AI community, and to
streamline the process of leveraging AI models in real applications.
Hundreds of models are made publicly available on ModelScope (700+ and
counting), covering the latest development in areas such as NLP, CV,
Audio, Multi-modality, and AI for Science, etc. Many of these models
represent the SOTA in their specific fields, and made their open-sourced
debut on ModelScope.

This tutorial covers how to use the modelscope ecosystem within
OpenVINO.

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert models from ModelScope using OpenVINO Model Conversion
   API <#convert-models-from-modelscope-using-openvino-model-conversion-api>`__

   -  `Select inference device for image
      classification <#select-inference-device-for-image-classification>`__
   -  `Run Image classification <#run-image-classification>`__

-  `Convert ModelScope models using Optimum
   Intel <#convert-modelscope-models-using-optimum-intel>`__

   -  `Select inference device for text
      classification <#select-inference-device-for-text-classification>`__
   -  `Perform text classification <#perform-text-classification>`__

-  `Convert ModelScope models for usage with OpenVINO
   GenAI <#convert-modelscope-models-for-usage-with-openvino-genai>`__

   -  `Select inference device for text
      generation <#select-inference-device-for-text-generation>`__
   -  `Run OpenVINO GenAI pipeline <#run-openvino-genai-pipeline>`__

Prerequisites
-------------



.. code:: ipython3

    import platform
    
    %pip install -q "torch>=2.1.1" "torchvision"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q modelscope addict oss2 simplejson sortedcontainers pillow opencv-python "datasets<=3.0.0"
    %pip install -q "transformers>=4.45" "git+https://github.com/huggingface/optimum-intel.git"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -qU "openvino>=2024.5.0" "openvino-tokenizers>=2024.5.0" "openvino-genai>=2024.5.0" "nncf>=2.14.0"
    
    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("modelscope-to-openvino.ipynb")

Convert models from ModelScope using OpenVINO Model Conversion API
------------------------------------------------------------------



Modelscope package provides API for initializing a model and loading a
set of pre-trained weights using the model text handle. Discovering a
desired model name is straightforward with `Modelscope models web
page <https://www.modelscope.cn/models>`__, one can choose a model
solving a particular machine learning problem and even sort the models
by popularity and novelty.

OpenVINO supports various types of models and frameworks via conversion
to OpenVINO Intermediate Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original model instance and example input for tracing and returns
``ov.Model`` representing this model in OpenVINO framework. Converted
model can be used for saving on disk using ``ov.save_model`` function or
directly loading on device using ``core.complie_model``.

As example, we will use
`tinynas <https://www.modelscope.cn/models/iic/cv_tinynas_classification>`__
image classification model. The code bellow demonstrates how to load
this model using Modelscope pipelines interface, convert it to OpenVINO
IR and then perform image classification on specified device.

.. code:: ipython3

    from pathlib import Path
    
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    import openvino as ov
    import torch
    import gc
    
    
    cls_model_id = "iic/cv_tinynas_classification"
    cls_model_path = Path(cls_model_id.split("/")[-1]) / "openvino_model.xml"
    
    if not cls_model_path.exists():
        # load Modelcope pipeline with model
        image_classification = pipeline(Tasks.image_classification, model=cls_model_id)
        # convert model to OpenVINO
        ov_model = ov.convert_model(image_classification.model, example_input=torch.zeros((1, 3, 224, 224)), input=[1, 3, 224, 224])
        # save OpenVINO model on disk for next usage
        ov.save_model(ov_model, cls_model_path)
        del ov_model
        del image_classification
        gc.collect();


.. parsed-literal::

    2024-11-12 19:08:10.199148: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-12 19:08:10.212253: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1731424090.226654 1605757 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1731424090.230976 1605757 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-11-12 19:08:10.246563: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

Select inference device for image classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    cv_cls_device = device_widget("CPU")
    
    cv_cls_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Run Image classification
~~~~~~~~~~~~~~~~~~~~~~~~



Model inference interface remains compatible with pipeline preprocessing
and postprocessing, so you can reuse these part of pipeline, but for
providing standalone experience, we will demonstrate how to use model
without pipeline. The code bellow defines utilities for image
preprocessing and postprocessing.

.. code:: ipython3

    from notebook_utils import download_file
    from PIL import Image
    from torchvision import transforms
    
    # prepare input data and output lables
    img_url = "https://pailitao-image-recog.oss-cn-zhangjiakou.aliyuncs.com/mufan/img_data/maas_test_data/dog.png"
    img_path = Path("dog.png")
    
    labels_url = "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/imagenet_2012.txt"
    
    labels_path = Path("imagenet_2012.txt")
    
    if not img_path.exists():
        download_file(img_url)
    
    if not labels_path.exists():
        download_file(labels_url)
    
    image = Image.open(img_path)
    imagenet_classes = labels_path.open("r").read().splitlines()
    
    
    # prepare image preprocessing
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms_normalize,
    ]
    transformer = transforms.Compose(transform_list)
    
    # compile model
    core = ov.Core()
    
    ov_model = core.compile_model(cls_model_path, cv_cls_device.value)

Now, when we make all necessary preparations, we can run model
inference.

.. code:: ipython3

    import numpy as np
    
    # preprocess input
    image_tensor = transformer(image)
    
    # run model inference
    result = ov_model(image_tensor.unsqueeze(0))[0]
    
    # postprocess results
    label_id = np.argmax(result[0])
    score = result[0][label_id]
    
    label = imagenet_classes[label_id]
    
    # visualize results
    display(image)
    print(f"Predicted label: {label}, score {score}")



.. image:: modelscope-to-openvino-with-output_files/modelscope-to-openvino-with-output_12_0.png


.. parsed-literal::

    Predicted label: n02099601 golden retriever, score 8.060977935791016
    

Convert ModelScope models using Optimum Intel
---------------------------------------------



For models compatible with the `HuggingFace
Transformers <https://huggingface.co/docs/transformers/index>`__
library, we can use `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ integration
to convert and run model. Optimum Intel is the interface between the
Transformers and Diffusers libraries and the different tools and
libraries provided by Intel to accelerate end-to-end pipelines on Intel
architectures.

Optimum Intel provides a simple interface for optimizing your
Transformers and Diffusers models, converting them to the OpenVINO
Intermediate Representation (IR) format, and running inference using
OpenVINO Runtime, among other use cases. For running ModelScope models
using this interface we should download model from hub first. There are
several ways how to download models from Modelscope Hub, one of them is
usage of ``modelscope.snapshot_download`` function. This function
accepts model id from hub and optionally local directory (if not
provided, model will be downloaded to cache directory).

After that, we can load model to Optimum Intel interface replacing the
``AutoModelForXxx`` class from transformers with the corresponding
``OVModelForXxx``. Model conversion will be performed on the fly. For
avoiding next time conversion, we can save model on disk using
``save_pretrained`` method and in the next time pass directory with
already converted model as argument in ``from_pretrained`` method. We
also specified ``device`` parameter for compiling the model on the
specific device, if not provided, the default device will be used. The
device can be changed later in runtime using ``model.to(device)``,
please note that it may require some time for model compilation on a
newly selected device. In some cases, it can be useful to separate model
initialization and compilation, for example, if you want to reshape the
model using ``reshape`` method, you can postpone compilation, providing
the parameter ``compile=False`` into ``from_pretrained`` method,
compilation can be performed manually using ``compile`` method or will
be performed automatically during first inference run.

As example, we will use
`nlp_bert_sentiment-analysis_english-base <https://modelscope.cn/models/iic/nlp_bert_sentiment-analysis_english-base>`__.
This model was trained for classification input text on 3 sentiment
categories: negative, positive and neutral. In transformers,
``AutoModelForSequenceClassification`` should be used for model
initialization, so for usage model with OpenVINO, it is enough just
replace ``AutoModelForSequenceClassification`` to
``OVModelForSequenceClassification``.

.. code:: ipython3

    from modelscope import snapshot_download
    
    text_model_id = "iic/nlp_bert_sentiment-analysis_english-base"
    text_model_path = Path(text_model_id.split("/")[-1])
    ov_text_model_path = text_model_path / "ov"
    
    
    if not text_model_path.exists():
        snapshot_download(text_model_id, local_dir=text_model_path)

Select inference device for text classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    text_cls_device = device_widget("CPU", "NPU")
    
    text_cls_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Perform text classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from transformers import AutoTokenizer
    from optimum.intel.openvino import OVModelForSequenceClassification
    
    
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    
    if not ov_text_model_path.exists():
        # model will be automatically exported to OpenVINO format during loading
        ov_model = OVModelForSequenceClassification.from_pretrained(text_model_path, text_cls_device.value)
        ov_model.save_pretrained(ov_text_model_path)
        # save converted model using save_pretrained for avoid conversion in next time
        tokenizer.save_pretrained(ov_text_model_path)
    else:
        # load converted model directly if availa ble
        ov_model = OVModelForSequenceClassification.from_pretrained(ov_text_model_path, device=text_cls_device.value)
    
    # prepare input
    input_text = "Good night."
    input_data = tokenizer(input_text, return_tensors="pt")
    
    # run model inference
    output = ov_model(**input_data)
    # postprocess results
    predicted_label_id = output.logits[0].argmax().item()
    
    predicted_label = ov_model.config.id2label[predicted_label_id]
    
    print(f"predicted label: {predicted_label}")


.. parsed-literal::

    predicted label: Positive
    

Convert ModelScope models for usage with OpenVINO GenAI
-------------------------------------------------------



OpenVINO™ GenAI is a library of the most popular Generative AI model
pipelines, optimized execution methods, and samples that run on top of
highly performant `OpenVINO
Runtime <https://github.com/openvinotoolkit/openvino>`__.

This library is friendly to PC and laptop execution, and optimized for
resource consumption. It requires no external dependencies to run
generative models as it already includes all the core functionality
(e.g. tokenization via openvino-tokenizers).

You can also load and run models from ModelScope with OpenVINO GenAI
`supported
pipelines <https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#supported-generative-ai-scenarios>`__.

This inference approach is also based on model representation obtained
using Optimum Intel and also requires to download ModelScope model
first. As example we will be
`qwen2.5-1.5b-instruct <https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct>`__
model for text generation, that is part of powerful Qwen2 LLMs family.
If in previous chapter we are focused with usage python API for
downloading and converting models, in this one - we are also considering
CLI usage for the same actions.

Downloading ModelScope models using CLI can be performed using following
command:

.. code:: bash

   modelscope download <model_id> --local_dir <model_local_dir>

where ``<model_id>`` is model id from Hub and ``<model_local_dir>`` is
output directory for model saving.

``optimum-cli`` provides command line interface for exporting models
using Optimum. General OpenVINO export command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where task is task to export the model for. Available tasks depend on
the model, but are among: [‘default’, ‘fill-mask’, ‘text-generation’,
‘text2text-generation’, ‘text-classification’, ‘token-classification’,
‘multiple-choice’, ‘object-detection’, ‘question-answering’,
‘image-classification’, ‘image-segmentation’, ‘masked-im’,
‘semantic-segmentation’, ‘automatic-speech-recognition’,
‘audio-classification’, ‘audio-frame-classification’,
‘automatic-speech-recognition’, ‘audio-xvector’, ‘image-to-text’,
‘stable-diffusion’, ‘zero-shot-object-detection’].

You can find a mapping between tasks and model classes in Optimum
TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.

Additionally, you can specify weights compression using
``--weight-format`` argument with one of following options: ``fp32``,
``fp16``, ``int8`` and ``int4``. For int8 and int4 nncf will be used for
weight compression. For models that required remote code execution,
``--trust-remote-code`` flag should be provided.

Full list of supported arguments available via ``--help``

.. code:: ipython3

    from IPython.display import Markdown, display
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    llm_path = Path("Qwen2.5-1.5B-Instruct")
    ov_llm_path = llm_path / "ov"
    download_command = f"modelscope download {model_id} --local_dir {llm_path}"
    display(Markdown("**Download command:**"))
    display(Markdown(f"`{download_command}`"))
    
    if not llm_path.exists():
        !{download_command}



**Download command:**



``modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir Qwen2.5-1.5B-Instruct``


.. code:: ipython3

    export_command = f"optimum-cli export openvino -m {llm_path} --task text-generation-with-past --weight-format int4 {ov_llm_path}"
    display(Markdown("**Export command:**"))
    display(Markdown(f"`{export_command}`"))
    
    if not ov_llm_path.exists():
        !{export_command}



**Export command:**



``optimum-cli export openvino -m Qwen2.5-1.5B-Instruct --task text-generation-with-past --weight-format int4 Qwen2.5-1.5B-Instruct/ov``


Select inference device for text generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from notebook_utils import device_widget
    
    llm_device = device_widget("CPU")
    
    llm_device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



Run OpenVINO GenAI pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~



For running text generation using OpenVINO GenAI, we should use
``LLMPipeline`` class initialized with providing converted model
directory and inference device. You can find more detailed example how
to use OpenVINO GenAI ``LLMPipeline`` for chatbot scenario in this
`tutorial <llm-chatbot-generate-api-with-output.html>`__.

.. code:: ipython3

    import openvino_genai as ov_genai
    
    
    def streamer(subword):
        print(subword, end="", flush=True)
        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False
    
    
    llm_pipe = ov_genai.LLMPipeline(ov_llm_path, llm_device.value)
    
    llm_pipe.generate("The Sun is yellow because", max_new_tokens=200, streamer=streamer)


.. parsed-literal::

     it has a spectrum of colors, and you are also looking at it. What color would the sun be if you could see its light without being able to see any other objects? If we imagine that someone had never seen or heard about the sun before, what would they expect to see?
    
    1. **Color of the Sun**: The sun appears yellow when viewed from Earth due to the way our atmosphere scatters sunlight. This phenomenon occurs as follows:
    
       - **Sunlight Scattering**: When sunlight passes through the Earth's atmosphere, different wavelengths (colors) of light travel at slightly different speeds due to their varying energies.
       - **Air Mass Height**: At higher altitudes where air density decreases with altitude, shorter wavelength (blue) photons have more energy and thus escape faster into space compared to longer wavelength (red) photons which remain in the atmosphere longer.
       - **Sky Color**: As a result, blue light is scattered more than red light by molecules in the upper layers of the atmosphere



.. parsed-literal::

    " it has a spectrum of colors, and you are also looking at it. What color would the sun be if you could see its light without being able to see any other objects? If we imagine that someone had never seen or heard about the sun before, what would they expect to see?\n\n1. **Color of the Sun**: The sun appears yellow when viewed from Earth due to the way our atmosphere scatters sunlight. This phenomenon occurs as follows:\n\n   - **Sunlight Scattering**: When sunlight passes through the Earth's atmosphere, different wavelengths (colors) of light travel at slightly different speeds due to their varying energies.\n   - **Air Mass Height**: At higher altitudes where air density decreases with altitude, shorter wavelength (blue) photons have more energy and thus escape faster into space compared to longer wavelength (red) photons which remain in the atmosphere longer.\n   - **Sky Color**: As a result, blue light is scattered more than red light by molecules in the upper layers of the atmosphere"



.. code:: ipython3

    import gc
    
    del llm_pipe
    gc.collect();
