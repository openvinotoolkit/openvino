OpenVINO™ Model conversion
==========================

This notebook shows how to convert a model from original framework
format to OpenVINO Intermediate Representation (IR).


**Table of contents:**


-  `OpenVINO IR format <#openvino-ir-format>`__
-  `Fetching example models <#fetching-example-models>`__
-  `Conversion <#conversion>`__

   -  `Setting Input Shapes <#setting-input-shapes>`__
   -  `Compressing a Model to FP16 <#compressing-a-model-to-fp16>`__
   -  `Convert Models from memory <#convert-models-from-memory>`__

-  `Migration from Legacy conversion
   API <#migration-from-legacy-conversion-api>`__

   -  `Specifying Layout <#specifying-layout>`__
   -  `Changing Model Layout <#changing-model-layout>`__
   -  `Specifying Mean and Scale
      Values <#specifying-mean-and-scale-values>`__
   -  `Reversing Input Channels <#reversing-input-channels>`__
   -  `Cutting Off Parts of a Model <#cutting-off-parts-of-a-model>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    # Required imports. Please execute this cell first.
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    "openvino>=2024.4.0" "requests" "tqdm" "transformers>=4.31" "onnx!=1.16.2" "torch>=2.1" "torchvision" "tensorflow_hub" "tensorflow"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


OpenVINO IR format
------------------



OpenVINO `Intermediate Representation
(IR) <https://docs.openvino.ai/2024/documentation/openvino-ir-format.html>`__
is the proprietary model format of OpenVINO. It is produced after
converting a model with model conversion API. Model conversion API
translates the frequently used deep learning operations to their
respective similar representation in OpenVINO and tunes them with the
associated weights and biases from the trained model. The resulting IR
contains two files: an ``.xml`` file, containing information about
network topology, and a ``.bin`` file, containing the weights and biases
binary data.

There are two ways to convert a model from the original framework format
to OpenVINO IR: Python conversion API and OVC command-line tool. You can
choose one of them based on whichever is most convenient for you.

OpenVINO conversion API supports next model formats: ``PyTorch``,
``TensorFlow``, ``TensorFlow Lite``, ``ONNX``, and ``PaddlePaddle``.
These model formats can be read, compiled, and converted to OpenVINO IR,
either automatically or explicitly.

For more details, refer to `Model
Preparation <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
documentation.

.. code:: ipython3

    # OVC CLI tool parameters description
    
    ! ovc --help


.. parsed-literal::

    usage: ovc INPUT_MODEL... [-h] [--output_model OUTPUT_MODEL]
               [--compress_to_fp16 [True | False]] [--version] [--input INPUT]
               [--output OUTPUT] [--extension EXTENSION] [--verbose]
    
    positional arguments:
      INPUT_MODEL           Input model file(s) from TensorFlow, ONNX,
                            PaddlePaddle. Use openvino.convert_model in Python to
                            convert models from PyTorch.
    
    optional arguments:
      -h, --help            show this help message and exit
      --output_model OUTPUT_MODEL
                            This parameter is used to name output .xml/.bin files
                            of converted model. Model name or output directory can
                            be passed. If output directory is passed, the
                            resulting .xml/.bin files are named by original model
                            name.
      --compress_to_fp16 [True | False]
                            Compress weights in output OpenVINO model to FP16. To
                            turn off compression use "--compress_to_fp16=False"
                            command line parameter. Default value is True.
      --version             Print ovc version and exit.
      --input INPUT         Information of model input required for model
                            conversion. This is a comma separated list with
                            optional input names and shapes. The order of inputs
                            in converted model will match the order of specified
                            inputs. The shape is specified as comma-separated
                            list. Example, to set `input_1` input with shape
                            [1,100] and `sequence_len` input with shape [1,?]:
                            "input_1[1,100],sequence_len[1,?]", where "?" is a
                            dynamic dimension, which means that such a dimension
                            can be specified later in the runtime. If the
                            dimension is set as an integer (like 100 in [1,100]),
                            such a dimension is not supposed to be changed later,
                            during a model conversion it is treated as a static
                            value. Example with unnamed inputs: "[1,100],[1,?]".
      --output OUTPUT       One or more comma-separated model outputs to be
                            preserved in the converted model. Other outputs are
                            removed. If `output` parameter is not specified then
                            all outputs from the original model are preserved. Do
                            not add :0 to the names for TensorFlow. The order of
                            outputs in the converted model is the same as the
                            order of specified names. Example: ovc model.onnx
                            output=out_1,out_2
      --extension EXTENSION
                            Paths or a comma-separated list of paths to libraries
                            (.so or .dll) with extensions.
      --verbose             Print detailed information about conversion.


Fetching example models
-----------------------



This notebook uses two models for conversion examples:

-  `Distilbert <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
   NLP model from Hugging Face
-  `Resnet50 <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
   CV classification model from torchvision

.. code:: ipython3

    from pathlib import Path
    
    # create a directory for models files
    MODEL_DIRECTORY_PATH = Path("model")
    MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

Fetch
`distilbert <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>`__
NLP model from Hugging Face and export it in ONNX format:

.. code:: ipython3

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.onnx import export, FeaturesManager
    
    ONNX_NLP_MODEL_PATH = MODEL_DIRECTORY_PATH / "distilbert.onnx"
    
    # download model
    hf_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # get model onnx config function for output feature format sequence-classification
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(hf_model, feature="sequence-classification")
    # fill onnx config based on pytorch model config
    onnx_config = model_onnx_config(hf_model.config)
    
    # export to onnx format
    export(
        preprocessor=tokenizer,
        model=hf_model,
        config=onnx_config,
        opset=onnx_config.default_onnx_opset,
        output=ONNX_NLP_MODEL_PATH,
    )


.. parsed-literal::

    2024-10-22 22:40:21.522113: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-22 22:40:21.555890: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-22 22:40:22.084160: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/distilbert/modeling_distilbert.py:215: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      mask, torch.tensor(torch.finfo(scores.dtype).min)




.. parsed-literal::

    (['input_ids', 'attention_mask'], ['logits'])



Fetch
`Resnet50 <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights>`__
CV classification model from torchvision:

.. code:: ipython3

    from torchvision.models import resnet50, ResNet50_Weights
    
    # create model object
    pytorch_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # switch model from training to inference mode
    pytorch_model.eval()




.. parsed-literal::

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )



Convert PyTorch model to ONNX format:

.. code:: ipython3

    import torch
    import warnings
    
    ONNX_CV_MODEL_PATH = MODEL_DIRECTORY_PATH / "resnet.onnx"
    
    if ONNX_CV_MODEL_PATH.exists():
        print(f"ONNX model {ONNX_CV_MODEL_PATH} already exists.")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            torch.onnx.export(model=pytorch_model, args=torch.randn(1, 3, 224, 224), f=ONNX_CV_MODEL_PATH)
        print(f"ONNX model exported to {ONNX_CV_MODEL_PATH}")


.. parsed-literal::

    ONNX model exported to model/resnet.onnx


Conversion
----------



To convert a model to OpenVINO IR, use the following API:

.. code:: ipython3

    import openvino as ov
    
    # ov.convert_model returns an openvino.runtime.Model object
    print(ONNX_NLP_MODEL_PATH)
    ov_model = ov.convert_model(ONNX_NLP_MODEL_PATH)
    
    # then model can be serialized to *.xml & *.bin files
    ov.save_model(ov_model, MODEL_DIRECTORY_PATH / "distilbert.xml")


.. parsed-literal::

    model/distilbert.onnx


.. code:: ipython3

    ! ovc model/distilbert.onnx --output_model model/distilbert.xml


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression by removing argument "compress_to_fp16" or set it to false "compress_to_fp16=False".
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ SUCCESS ] XML file: model/distilbert.xml
    [ SUCCESS ] BIN file: model/distilbert.bin


Setting Input Shapes
^^^^^^^^^^^^^^^^^^^^



Model conversion is supported for models with dynamic input shapes that
contain undefined dimensions. However, if the shape of data is not going
to change from one inference request to another, it is recommended to
set up static shapes (when all dimensions are fully defined) for the
inputs. Doing so at the model preparation stage, not at runtime, can be
beneficial in terms of performance and memory consumption.

For more information refer to `Setting Input
Shapes <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/setting-input-shapes.html>`__
documentation.

.. code:: ipython3

    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_NLP_MODEL_PATH, input=[("input_ids", [1, 128]), ("attention_mask", [1, 128])])

.. code:: ipython3

    ! ovc model/distilbert.onnx --input input_ids[1,128],attention_mask[1,128] --output_model model/distilbert.xml


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression by removing argument "compress_to_fp16" or set it to false "compress_to_fp16=False".
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ SUCCESS ] XML file: model/distilbert.xml
    [ SUCCESS ] BIN file: model/distilbert.bin


The ``input`` parameter allows overriding original input shapes if it is
supported by the model topology. Shapes with dynamic dimensions in the
original model can be replaced with static shapes for the converted
model, and vice versa. The dynamic dimension can be marked in model
conversion API parameter as ``-1`` or ``?`` when using ``ovc``:

.. code:: ipython3

    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_NLP_MODEL_PATH, input=[("input_ids", [1, -1]), ("attention_mask", [1, -1])])

.. code:: ipython3

    ! ovc model/distilbert.onnx --input "input_ids[1,?],attention_mask[1,?]" --output_model model/distilbert.xml


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression by removing argument "compress_to_fp16" or set it to false "compress_to_fp16=False".
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ SUCCESS ] XML file: model/distilbert.xml
    [ SUCCESS ] BIN file: model/distilbert.bin


To optimize memory consumption for models with undefined dimensions in
runtime, model conversion API provides the capability to define
boundaries of dimensions. The boundaries of undefined dimension can be
specified with ellipsis in the command line or with
``openvino.Dimension`` class in Python. For example, launch model
conversion for the ONNX Bert model and specify a boundary for the
sequence length dimension:

.. code:: ipython3

    import openvino as ov
    
    
    sequence_length_dim = ov.Dimension(10, 128)
    
    ov_model = ov.convert_model(
        ONNX_NLP_MODEL_PATH,
        input=[
            ("input_ids", [1, sequence_length_dim]),
            ("attention_mask", [1, sequence_length_dim]),
        ],
    )

.. code:: ipython3

    ! ovc model/distilbert.onnx --input input_ids[1,10..128],attention_mask[1,10..128] --output_model model/distilbert.xml


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression by removing argument "compress_to_fp16" or set it to false "compress_to_fp16=False".
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ SUCCESS ] XML file: model/distilbert.xml
    [ SUCCESS ] BIN file: model/distilbert.bin


Compressing a Model to FP16
^^^^^^^^^^^^^^^^^^^^^^^^^^^



By default model weights compressed to FP16 format when saving OpenVINO
model to IR. This saves up to 2x storage space for the model file and in
most cases doesn’t sacrifice model accuracy. Weight compression can be
disabled by setting ``compress_to_fp16`` flag to ``False``:

.. code:: ipython3

    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_NLP_MODEL_PATH)
    ov.save_model(ov_model, MODEL_DIRECTORY_PATH / "distilbert.xml", compress_to_fp16=False)

.. code:: ipython3

    ! ovc model/distilbert.onnx --output_model model/distilbert.xml --compress_to_fp16=False


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    [ SUCCESS ] XML file: model/distilbert.xml
    [ SUCCESS ] BIN file: model/distilbert.bin


Convert Models from memory
^^^^^^^^^^^^^^^^^^^^^^^^^^



Model conversion API supports passing original framework Python object
directly. More details can be found in
`PyTorch <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html>`__,
`TensorFlow <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html>`__,
`PaddlePaddle <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-paddle.html>`__
frameworks conversion guides.

.. code:: ipython3

    import openvino as ov
    import torch
    
    example_input = torch.rand(1, 3, 224, 224)
    
    ov_model = ov.convert_model(pytorch_model, example_input=example_input, input=example_input.shape)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. code:: ipython3

    import os
    
    import openvino as ov
    import tensorflow_hub as hub
    
    os.environ["TFHUB_CACHE_DIR"] = str(Path("./tfhub_modules").resolve())
    
    model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-lightning/versions/4")
    movenet = model.signatures["serving_default"]
    
    ov_model = ov.convert_model(movenet)


.. parsed-literal::

    2024-10-22 22:40:40.138322: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...


Migration from Legacy conversion API
------------------------------------



In the 2023.1 OpenVINO release OpenVINO Model Conversion API was
introduced with the corresponding Python API: ``openvino.convert_model``
method. ``ovc`` and ``openvino.convert_model`` represent a lightweight
alternative of ``mo`` and ``openvino.tools.mo.convert_model`` which are
considered legacy API now. ``mo.convert_model()`` provides a wide range
of preprocessing parameters. Most of these parameters have analogs in
OVC or can be replaced with functionality from ``ov.PrePostProcessor``
class. Refer to `Optimize Preprocessing
notebook <optimize-preprocessing-with-output.html>`__ for
more information about `Preprocessing
API <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html>`__.
Here is the migration guide from legacy model preprocessing to
Preprocessing API.

Specifying Layout
^^^^^^^^^^^^^^^^^



Layout defines the meaning of dimensions in a shape and can be specified
for both inputs and outputs. Some preprocessing requires to set input
layouts, for example, setting a batch, applying mean or scales, and
reversing input channels (BGR<->RGB). For the layout syntax, check the
`Layout API
overview <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing/layout-api-overview.html>`__.
To specify the layout, you can use the layout option followed by the
layout value.

The following example specifies the ``NCHW`` layout for a Pytorch
Resnet50 model that was exported to the ONNX format:

.. code:: ipython3

    # Converter API
    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_CV_MODEL_PATH)
    
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input("input.1").model().set_layout(ov.Layout("nchw"))
    ov_model = prep.build()

.. code:: python

   # Legacy Model Optimizer API
   from openvino.tools import mo

   ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, layout="nchw")

Changing Model Layout
^^^^^^^^^^^^^^^^^^^^^



Transposing of matrices/tensors is a typical operation in Deep Learning
- you may have a BMP image ``640x480``, which is an array of
``{480, 640, 3}`` elements, but Deep Learning model can require input
with shape ``{1, 3, 480, 640}``.

Conversion can be done implicitly, using the layout of a user’s tensor
and the layout of an original model:

.. code:: ipython3

    # Converter API
    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_CV_MODEL_PATH)
    
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input("input.1").tensor().set_layout(ov.Layout("nhwc"))
    prep.input("input.1").model().set_layout(ov.Layout("nchw"))
    ov_model = prep.build()

Legacy Model Optimizer API
==========================

.. code:: python

   from openvino.tools import mo

   ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, layout="nchw->nhwc")

   # alternatively use source_layout and target_layout parameters
   ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, source_layout="nchw", target_layout="nhwc")

Specifying Mean and Scale Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Using Preprocessing API ``mean`` and ``scale`` values can be set. Using
these API, model embeds the corresponding preprocessing block for
mean-value normalization of the input data and optimizes this block.
Refer to `Optimize Preprocessing
notebook <optimize-preprocessing-with-output.html>`__ for
more examples.

.. code:: ipython3

    # Converter API
    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_CV_MODEL_PATH)
    
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input("input.1").tensor().set_layout(ov.Layout("nchw"))
    prep.input("input.1").preprocess().mean([255 * x for x in [0.485, 0.456, 0.406]])
    prep.input("input.1").preprocess().scale([255 * x for x in [0.229, 0.224, 0.225]])
    
    ov_model = prep.build()

.. code:: python

   # Legacy Model Optimizer API

   from openvino.tools import mo


   ov_model = mo.convert_model(
       ONNX_CV_MODEL_PATH,
       mean_values=[255 * x for x in [0.485, 0.456, 0.406]],
       scale_values=[255 * x for x in [0.229, 0.224, 0.225]],
   )

Reversing Input Channels
^^^^^^^^^^^^^^^^^^^^^^^^



Sometimes, input images for your application can be of the ``RGB`` (or
``BGR``) format, and the model is trained on images of the ``BGR`` (or
``RGB``) format, which is in the opposite order of color channels. In
this case, it is important to preprocess the input images by reverting
the color channels before inference.

.. code:: ipython3

    # Converter API
    import openvino as ov
    
    ov_model = ov.convert_model(ONNX_CV_MODEL_PATH)
    
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input("input.1").tensor().set_layout(ov.Layout("nchw"))
    prep.input("input.1").preprocess().reverse_channels()
    ov_model = prep.build()

.. code:: python

   # Legacy Model Optimizer API
   from openvino.tools import mo

   ov_model = mo.convert_model(ONNX_CV_MODEL_PATH, reverse_input_channels=True)

Cutting Off Parts of a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Cutting model inputs and outputs from a model is no longer available in
the new conversion API. Instead, we recommend performing the cut in the
original framework. Examples of model cutting of TensorFlow protobuf,
TensorFlow SavedModel, and ONNX formats with tools provided by the
Tensorflow and ONNX frameworks can be found in `documentation
guide <https://docs.openvino.ai/2024/documentation/legacy-features/transition-legacy-conversion-api.html#cutting-off-parts-of-a-model>`__.
For PyTorch, TensorFlow 2 Keras, and PaddlePaddle, we recommend changing
the original model code to perform the model cut.
