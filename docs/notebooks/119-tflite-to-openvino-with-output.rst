Convert a Tensorflow Lite Model to OpenVINO™
============================================

`TensorFlow Lite <https://www.tensorflow.org/lite/guide>`__, often
referred to as TFLite, is an open source library developed for deploying
machine learning models to edge devices.

This short tutorial shows how to convert a TensorFlow Lite
`EfficientNet-Lite-B0 <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
image classification model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_IR_and_opsets.html>`__
(OpenVINO IR) format, using Model Converter. After creating the OpenVINO
IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and do inference with a sample image.

**Table of contents:**

- `Preparation <#preparation>`__

  - `Install requirements <#install-requirements>`__
  - `Imports <#imports>`__

- `Download TFLite model <#download-tflite-model>`__
- `Convert a Model to OpenVINO IR Format <#convert-a-model-to-openvino-ir-format>`__
- `Load model using OpenVINO TensorFlow Lite Frontend <#load-model-using-openvino-tensorflow-lite-frontend>`__
- `Run OpenVINO model inference <#run-openvino-model-inference>`__

  - `Select inference device <#select-inference-device>`__

- `Estimate Model Performance <#estimate-model-performance>`__

Preparation
###############################################################################################################################

Install requirements
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    !pip install -q "openvino==2023.1.0.dev20230811"
    !pip install -q opencv-python requests tqdm
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    );

Imports
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    from pathlib import Path
    import numpy as np
    from PIL import Image
    import openvino as ov
    
    from notebook_utils import download_file, load_image

Download TFLite model
###############################################################################################################################

.. code:: ipython3

    model_dir = Path("model")
    tflite_model_path = model_dir / "efficientnet_lite0_fp32_2.tflite"
    
    ov_model_path = tflite_model_path.with_suffix(".xml")
    model_url = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite"
    
    download_file(model_url, tflite_model_path.name, model_dir)



.. parsed-literal::

    model/efficientnet_lite0_fp32_2.tflite:   0%|          | 0.00/17.7M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/119-tflite-to-openvino/model/efficientnet_lite0_fp32_2.tflite')



Convert a Model to OpenVINO IR Format
###############################################################################################################################

To convert the TFLite model to OpenVINO IR, model conversion Python API
can be used. ``ov.convert_model`` function accepts the path to the
TFLite model and returns an OpenVINO Model class instance which
represents this model. The obtained model is ready to use and to be
loaded on a device using ``ov.compile_model`` or can be saved on a disk
using ``ov.save_model`` function, reducing loading time for next
running. By default, model weights are compressed to FP16 during
serialization by ``ov.save_model``. For more information about model
conversion, see this
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.
For TensorFlow Lite models support, refer to this
`tutorial <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite.html>`__.

.. code:: ipython3

    ov_model = ov.convert_model(tflite_model_path)
    ov.save_model(ov_model, ov_model_path)
    print(f"Model {tflite_model_path} successfully converted and saved to {ov_model_path}")


.. parsed-literal::

    Model model/efficientnet_lite0_fp32_2.tflite successfully converted and saved to model/efficientnet_lite0_fp32_2.xml


Load model using OpenVINO TensorFlow Lite Frontend
###############################################################################################################################

TensorFlow Lite models are supported via ``FrontEnd`` API. You may skip
conversion to IR and read models directly by OpenVINO runtime API. For
more examples supported formats reading via Frontend API, please look
this `tutorial <../002-openvino-api>`__.

.. code:: ipython3

    core = ov.Core()
    
    ov_model = core.read_model(tflite_model_path)

Run OpenVINO model inference
###############################################################################################################################

We can find information about model input preprocessing in its
`description <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
on `TensorFlow Hub <https://tfhub.dev/>`__.

.. code:: ipython3

    image = load_image("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png")
    # load_image reads the image in BGR format, [:,:,::-1] reshape transfroms it to RGB
    image = Image.fromarray(image[:,:,::-1])
    resized_image = image.resize((224, 224))
    input_tensor = np.expand_dims((np.array(resized_image).astype(np.float32) - 127) / 128, 0)

Select inference device
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Select device from dropdown list for running inference using OpenVINO:

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_model = core.compile_model(ov_model)
    predicted_scores = compiled_model(input_tensor)[0]

.. code:: ipython3

    imagenet_classes_file_path = download_file("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt")
    imagenet_classes = open(imagenet_classes_file_path).read().splitlines()
    
    top1_predicted_cls_id = np.argmax(predicted_scores)
    top1_predicted_score = predicted_scores[0][top1_predicted_cls_id]
    predicted_label = imagenet_classes[top1_predicted_cls_id]
    
    display(image.resize((640, 512)))
    print(f"Predicted label: {predicted_label} with probability {top1_predicted_score :2f}")



.. parsed-literal::

    imagenet_2012.txt:   0%|          | 0.00/30.9k [00:00<?, ?B/s]



.. image:: 119-tflite-to-openvino-with-output_files/119-tflite-to-openvino-with-output_16_1.png


.. parsed-literal::

    Predicted label: n02109047 Great Dane with probability 0.715318


Estimate Model Performance
###############################################################################################################################

`Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
is used to measure the inference performance of the model on CPU and
GPU.

.. note::

   For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    print("Benchmark model inference on CPU")
    !benchmark_app -m $ov_model_path -d CPU -t 15
    if "GPU" in core.available_devices:
        print("\n\nBenchmark model inference on GPU")
        !benchmark_app -m $ov_model_path -d GPU -t 15


.. parsed-literal::

    Benchmark model inference on CPU
    /bin/bash: benchmark_app: command not found

