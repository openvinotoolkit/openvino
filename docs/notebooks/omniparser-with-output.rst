Screen Parsing with OmniParser and OpenVINO
===========================================

Recent breakthrough in Visual Language Processing and Large Language
models made significant strides in understanding and interacting with
the world through text and images. However, accurately parsing and
understanding complex graphical user interfaces (GUIs) remains a
significant challenge. OmniParser is a comprehensive method for parsing
user interface screenshots into structured and easy-to-understand
elements. This enables more accurate and efficient interaction with
GUIs, empowering AI agents to perform tasks across various platforms and
applications.

|image0|

More details about model can be found in `Microsoft blog
post <https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/>`__,
`paper <https://arxiv.org/pdf/2408.00203>`__, `original
repo <https://github.com/microsoft/OmniParser>`__ and `model
card <https://huggingface.co/microsoft/OmniParser>`__. In this tutorial
we consider how to run OmniParser using OpenVINO.


**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Prepare models <#prepare-models>`__

   -  `Convert models to OpenVINO Intermediate representation
      format <#convert-models-to-openvino-intermediate-representation-format>`__

      -  `Icon Detector <#icon-detector>`__
      -  `Screen captioning model <#screen-captioning-model>`__

-  `Run OmniParser using OpenVINO <#run-omniparser-using-openvino>`__

   -  `Icon Detector <#icon-detector>`__

      -  `Select inference device for icon
         detector <#select-inference-device-for-icon-detector>`__

   -  `Screen regions captioning <#screen-regions-captioning>`__

      -  `Select device for screen region
         captioning <#select-device-for-screen-region-captioning>`__

   -  `Recognition text on the
      screen <#recognition-text-on-the-screen>`__

      -  `Select device for OCR <#select-device-for-ocr>`__

   -  `Run model inference <#run-model-inference>`__

-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://microsoft.github.io/OmniParser/static/images/flow_merged0.png

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "torch>=2.1" easyocr torchvision accelerate "supervision==0.18.0" accelerate timm "einops==0.8.0" "ultralytics==8.1.24" pillow opencv-python "gradio>=4.19" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2024.4.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    import requests

    notebook_utils_path = Path("notebook_utils.py")
    florence_helper_path = Path("ov_florence2_helper.py")

    if not notebook_utils_path.exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        notebook_utils_path.open("w").write(r.text)

    if not florence_helper_path.exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/florence2/ov_florence2_helper.py")
        florence_helper_path.open("w").write(r.text)

Prepare models
--------------



OmniParser leverages a two-step process: 1. Interactable Region
Detection: - Identifies clickable elements like buttons and icons within
a UI. - Employs a specialized model trained on a diverse dataset of web
pages. - Accurately detects interactive elements, even in complex UIs.

2. Semantic Captioning:

   -  Assigns meaningful descriptions to detected elements.
   -  Combines optical character recognition (OCR) and a captioning
      model.
   -  Provides context for accurate action generation.

Convert models to OpenVINO Intermediate representation format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For starting work with OpenVINO
we should convert models to OpenVINO Intermediate Representation format
first.

`OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original model instance and example input for tracing and returns
``ov.Model`` representing this model in OpenVINO framework. Converted
model can be used for saving on disk using ``ov.save_model`` function or
directly loading on device using ``core.complie_model``.

Let’s consider each pipeline part.

Icon Detector
^^^^^^^^^^^^^



Icon detector in OmniParser is represented by YOLO based model trained
on curated by model authors interactable icon detection dataset.

For conversion and model inference we will utilize Ultralytics provided
API. You can find more examples of this API usage in these
`tutorials <https://openvinotoolkit.github.io/openvino_notebooks/?libraries=Ultralytics>`__

.. code:: ipython3

    from ov_omniparser_helper import download_omniparser_icon_detector

    icon_detector_dir = download_omniparser_icon_detector()


.. parsed-literal::

    2024-11-22 01:51:07.385705: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-11-22 01:51:07.410345: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



.. parsed-literal::

    weights/icon_detect/best.pt:   0%|          | 0.00/11.7M [00:00<?, ?B/s]



.. parsed-literal::

    weights/icon_detect/model.yaml:   0%|          | 0.00/1.06k [00:00<?, ?B/s]


.. code:: ipython3

    from ultralytics import YOLO
    import gc

    ov_icon_detector_path = icon_detector_dir / "best_openvino_model/best.xml"

    if not ov_icon_detector_path.exists():
        icon_detector = YOLO(icon_detector_dir / "best.pt", task="detect")
        icon_detector.export(format="openvino", dynamic=True, half=True)
        del icon_detector
        gc.collect();


.. parsed-literal::

    Ultralytics YOLOv8.1.24 🚀 Python-3.8.10 torch-2.2.2+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)
    model summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs

    PyTorch: starting from 'weights/icon_detect/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (11.7 MB)

    OpenVINO: starting export with openvino 2024.4.0-16579-c3152d32c9c-releases/2024/4...
    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    OpenVINO: export success ✅ 1.5s, saved as 'weights/icon_detect/best_openvino_model/' (6.1 MB)

    Export complete (2.9s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/notebooks/omniparser/weights/icon_detect
    Predict:         yolo predict task=detect model=weights/icon_detect/best_openvino_model imgsz=640 half
    Validate:        yolo val task=detect model=weights/icon_detect/best_openvino_model imgsz=640 data=None half
    Visualize:       https://netron.app


Screen captioning model
^^^^^^^^^^^^^^^^^^^^^^^



The second part of OmniParser pipeline is creating detailed descriptions
of recognized clickable regions. For these purposes pipeline suggests to
use several visual language processing models like BLIP2, Florence2 or
Phi-3-Vision. In this tutorial we will focus on making screen region
captioning using Florence2. Previously we explained in details model
workflow and steps for running it using OpenVINO in this
`tutorial <https://openvinotoolkit.github.io/openvino_notebooks/?search=Florence-2%3A+Open+Source+Vision+Foundation+Model>`__.

.. code:: ipython3

    from ov_omniparser_helper import download_omniparser_florence_model

    florence_caption_dir = download_omniparser_florence_model()



.. parsed-literal::

    Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]



.. parsed-literal::

    LICENSE:   0%|          | 0.00/1.14k [00:00<?, ?B/s]



.. parsed-literal::

    config.json:   0%|          | 0.00/2.43k [00:00<?, ?B/s]



.. parsed-literal::

    SUPPORT.md:   0%|          | 0.00/1.24k [00:00<?, ?B/s]



.. parsed-literal::

    SECURITY.md:   0%|          | 0.00/2.66k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.56k [00:00<?, ?B/s]



.. parsed-literal::

    configuration_florence2.py:   0%|          | 0.00/15.1k [00:00<?, ?B/s]



.. parsed-literal::

    CODE_OF_CONDUCT.md:   0%|          | 0.00/444 [00:00<?, ?B/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/14.8k [00:00<?, ?B/s]



.. parsed-literal::

    vocab.json:   0%|          | 0.00/1.10M [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]



.. parsed-literal::

    processing_florence2.py:   0%|          | 0.00/46.4k [00:00<?, ?B/s]



.. parsed-literal::

    preprocessor_config.json:   0%|          | 0.00/806 [00:00<?, ?B/s]



.. parsed-literal::

    modeling_florence2.py:   0%|          | 0.00/127k [00:00<?, ?B/s]



.. parsed-literal::

    tokenizer_config.json:   0%|          | 0.00/34.0 [00:00<?, ?B/s]



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/464M [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors:   0%|          | 0.00/1.08G [00:00<?, ?B/s]



.. parsed-literal::

    icon_caption_florence/config.json:   0%|          | 0.00/5.66k [00:00<?, ?B/s]



.. parsed-literal::

    (…)_caption_florence/generation_config.json:   0%|          | 0.00/292 [00:00<?, ?B/s]


.. code:: ipython3

    from ov_florence2_helper import convert_florence2

    # Uncomment the line to see conversion code
    # ??convert_florence2

.. code:: ipython3

    ov_florence_path = Path("weights/icon_caption_florence_ov")
    convert_florence2(florence_caption_dir.name, ov_florence_path, florence_caption_dir)


.. parsed-literal::

    ⌛ icon_caption_florence conversion started. Be patient, it may takes some time.
    ⌛ Load Original model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
    Florence2LanguageForConditionalGeneration has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
      - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
      - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
      - If you are not the owner of the model architecture class, please contact the model code owner to update it.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.


.. parsed-literal::

    ✅ Original model successfully loaded
    ⌛ Image Embeddings conversion started
    ✅ Image Embeddings successfuly converted
    ⌛ Text Embedding conversion started
    ✅ Text Embedding conversion started
    ⌛ Encoder conversion started
    ✅ Encoder conversion finished
    ⌛ Decoder conversion started
    ✅ Decoder conversion finished
    ✅ icon_caption_florence already converted and can be found in weights/icon_caption_florence_ov


Run OmniParser using OpenVINO
-----------------------------



Now, it is time to configure and run OmniParser inference using
OpenVINO.

Icon Detector
~~~~~~~~~~~~~



Select inference device for icon detector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget("CPU", ["NPU"])

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    from ov_omniparser_helper import load_ov_icon_detector

    ov_icon_detector = load_ov_icon_detector(ov_icon_detector_path, device.value)


.. parsed-literal::

    Ultralytics YOLOv8.1.24 🚀 Python-3.8.10 torch-2.2.2+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)
    Loading weights/icon_detect/best_openvino_model for OpenVINO inference...


Screen regions captioning
~~~~~~~~~~~~~~~~~~~~~~~~~



``OVFlorence2Model`` class defined in ``ov_florence2_helper.py``
provides convenient way for running model. It accepts directory with
converted model and inference device as arguments. For running model we
will use ``generate`` method. Additionally, for model usage we also need
``Processor`` class, that distributed with original model and can be
loaded using ``AutoProcessor`` from ``transformers`` library. Processor
is responsible for input data preparation and decoding model output.

.. code:: ipython3

    from ov_florence2_helper import OVFlorence2Model
    from transformers import AutoProcessor

    # Uncomment the line to see model class code
    # ??OVFlorence2Model

Select device for screen region captioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    ov_icon_caption_gen = OVFlorence2Model(ov_florence_path, device.value)
    processor = AutoProcessor.from_pretrained(ov_florence_path, trust_remote_code=True)

Recognition text on the screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Alongside with captioning model, OmniParser also uses Optical Character
Recognition (OCR) for understanding text on the screen.
`EasyOCR <https://github.com/JaidedAI/EasyOCR>`__ is a python module for
extracting text from image. It is a general OCR that can read both
natural scene text and dense text in document and supports 80+
languages. EasyOCR utilizes AI for detection text regions and recognize
text inside of predicted regions. We will also utilize both text
detection and recognition models using OpenVINO.

Select device for OCR
^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    import ipywidgets as widgets

    device_detector = device_widget(exclude=["NPU"], description="Detector device:")
    device_recognizer = device_widget(exclude=["NPU"], description="Recognizer device:")

    device_box = widgets.VBox([device_detector, device_recognizer])
    device_box




.. parsed-literal::

    VBox(children=(Dropdown(description='Detector device:', index=1, options=('CPU', 'AUTO'), value='AUTO'), Dropd…



.. code:: ipython3

    from ov_omniparser_helper import easyocr_reader

    # Uncomment the line to see easyocr_reader helper code
    # ??easyocr_reader

.. code:: ipython3

    reader = easyocr_reader("weights/easyocr", device_detector.value, device_recognizer.value)


.. parsed-literal::

    Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.




.. code:: ipython3

    from PIL import Image

    test_image_path = Path("examples/windows_home.png")
    test_image_path.parent.mkdir(exist_ok=True, parents=True)

    if not test_image_path.exists():
        Image.open(requests.get("https://github.com/microsoft/OmniParser/blob/master/imgs/windows_home.png?raw=true", stream=True).raw).save(test_image_path)

Run model inference
~~~~~~~~~~~~~~~~~~~



``process_image`` function defined in ``ov_omniparser_helper.py``
provides easy-to-use interface for screen parsing process.

.. code:: ipython3

    from ov_omniparser_helper import process_image

    # Uncomment this line to see process_image code
    # ??process_image

.. code:: ipython3

    procesed_image, label_coordinates, icon_descriptions = process_image(
        test_image_path, ov_icon_detector, {"model": ov_icon_caption_gen, "processor": processor}, reader
    )


.. parsed-literal::


    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/823/archive/.workspace/scm/ov-notebook/notebooks/omniparser/examples/windows_home.png: 640x640 32 0s, 38.2ms
    Speed: 2.4ms preprocess, 38.2ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 640)
    finish processing


Function returns image with drawn detected boxes, boxes coordinates and
description for each region.

.. code:: ipython3

    display(procesed_image.resize((1200, 1200)))
    print(icon_descriptions)



.. image:: omniparser-with-output_files/omniparser-with-output_32_0.png


.. parsed-literal::

    Text Box ID 0: 3.46 PM
    Text Box ID 1: Search
    Text Box ID 2: Microsoft
    Text Box ID 3: 10/25/2024
    Icon Box ID 4: Microsoft Outlook.
    Icon Box ID 5: Image
    Icon Box ID 6: Microsoft OneNote.
    Icon Box ID 7: Microsoft Office.
    Icon Box ID 8: a folder for organizing files.
    Icon Box ID 9: Microsoft Office.
    Icon Box ID 10: Security shield.
    Icon Box ID 11: Microsoft 365.
    Icon Box ID 12: Microsoft Edge browser.
    Icon Box ID 13: Microsoft Edge browser.
    Icon Box ID 14: Decrease
    Icon Box ID 15: the Windows operating system.
    Icon Box ID 16: mountains and a beach.
    Icon Box ID 17: a search function.


Interactive demo
----------------

.. code:: ipython3

    from gradio_helper import make_demo


    def process_image_gradio(image, box_threshold, iou_threshold, imgsz):
        image_result, _, parsed_text = process_image(
            image,
            ov_icon_detector,
            {"model": ov_icon_caption_gen, "processor": processor},
            reader,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        return image_result, parsed_text


    demo = make_demo(process_image_gradio)

    try:
        demo.launch(debug=False, height=600)
    except Exception:
        demo.launch(debug=False, share=True, height=600)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







