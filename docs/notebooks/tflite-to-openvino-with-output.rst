Convert a Tensorflow Lite Model to OpenVINO™
============================================

`TensorFlow Lite <https://www.tensorflow.org/lite/guide>`__, often
referred to as TFLite, is an open source library developed for deploying
machine learning models to edge devices.

This short tutorial shows how to convert a TensorFlow Lite
`EfficientNet-Lite-B0 <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
image classification model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets.html>`__
(OpenVINO IR) format, using Model Converter. After creating the OpenVINO
IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/2024/openvino-workflow/running-inference.html>`__
and do inference with a sample image.


**Table of contents:**


-  `Preparation <#preparation>`__

   -  `Install requirements <#install-requirements>`__
   -  `Imports <#imports>`__

-  `Download TFLite model <#download-tflite-model>`__
-  `Convert a Model to OpenVINO IR
   Format <#convert-a-model-to-openvino-ir-format>`__
-  `Load model using OpenVINO TensorFlow Lite
   Frontend <#load-model-using-openvino-tensorflow-lite-frontend>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__

   -  `Select inference device <#select-inference-device>`__

-  `Estimate Model Performance <#estimate-model-performance>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Preparation
-----------



Install requirements
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q opencv-python requests tqdm kagglehub Pillow
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.




.. parsed-literal::

    24692



Imports
~~~~~~~



.. code:: ipython3

    from pathlib import Path
    import numpy as np
    from PIL import Image
    import openvino as ov
    
    from notebook_utils import download_file, load_image, device_widget

Download TFLite model
---------------------



.. code:: ipython3

    import kagglehub
    
    model_dir = kagglehub.model_download("tensorflow/efficientnet/tfLite/lite0-fp32")
    tflite_model_path = Path(model_dir) / "2.tflite"
    
    ov_model_path = tflite_model_path.with_suffix(".xml")


.. parsed-literal::

    Warning: Looks like you're using an outdated `kagglehub` version, please consider updating (latest version: 0.3.3)


Convert a Model to OpenVINO IR Format
-------------------------------------



To convert the TFLite model to OpenVINO IR, model conversion Python API
can be used. ``ov.convert_model`` function accepts the path to the
TFLite model and returns an OpenVINO Model class instance which
represents this model. The obtained model is ready to use and to be
loaded on a device using ``ov.compile_model`` or can be saved on a disk
using ``ov.save_model`` function, reducing loading time for next
running. By default, model weights are compressed to FP16 during
serialization by ``ov.save_model``. For more information about model
conversion, see this
`page <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.
For TensorFlow Lite models support, refer to this
`tutorial <https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow-lite.html>`__.

.. code:: ipython3

    ov_model = ov.convert_model(tflite_model_path)
    ov.save_model(ov_model, ov_model_path)
    print(f"Model {tflite_model_path} successfully converted and saved to {ov_model_path}")


.. parsed-literal::

    Model /opt/home/k8sworker/.cache/kagglehub/models/tensorflow/efficientnet/tfLite/lite0-fp32/2/2.tflite successfully converted and saved to /opt/home/k8sworker/.cache/kagglehub/models/tensorflow/efficientnet/tfLite/lite0-fp32/2/2.xml


Load model using OpenVINO TensorFlow Lite Frontend
--------------------------------------------------



TensorFlow Lite models are supported via ``FrontEnd`` API. You may skip
conversion to IR and read models directly by OpenVINO runtime API. For
more examples supported formats reading via Frontend API, please look
this `tutorial <openvino-api-with-output.html>`__.

.. code:: ipython3

    core = ov.Core()
    
    ov_model = core.read_model(tflite_model_path)

Run OpenVINO model inference
----------------------------



We can find information about model input preprocessing in its
`description <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
on `TensorFlow Hub <https://tfhub.dev/>`__.

.. code:: ipython3

    image = load_image("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png")
    # load_image reads the image in BGR format, [:,:,::-1] reshape transfroms it to RGB
    image = Image.fromarray(image[:, :, ::-1])
    resized_image = image.resize((224, 224))
    input_tensor = np.expand_dims((np.array(resized_image).astype(np.float32) - 127) / 128, 0)

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_model = core.compile_model(ov_model, device.value)
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



.. image:: tflite-to-openvino-with-output_files/tflite-to-openvino-with-output_16_1.png


.. parsed-literal::

    Predicted label: n02109047 Great Dane with probability 0.715318


Estimate Model Performance
--------------------------

`Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
is used to measure the inference performance of the model on CPU and
GPU.

   **NOTE**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    print(f"Benchmark model inference on {device.value}")
    !benchmark_app -m $ov_model_path -d $device.value -t 15


.. parsed-literal::

    Benchmark model inference on AUTO
    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 9.48 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,224,224,3]
    [ INFO ] Model outputs:
    [ INFO ]     Softmax (node: 61) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,H,W,C] / [1,224,224,3]
    [ INFO ] Model outputs:
    [ INFO ]     Softmax (node: 61) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 157.24 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Lite_Frontend_IR
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: TensorFlow_Lite_Frontend_IR
    [ INFO ]     NUM_STREAMS: 6
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.14 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            17436 iterations
    [ INFO ] Duration:         15008.41 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.99 ms
    [ INFO ]    Average:       5.03 ms
    [ INFO ]    Min:           2.84 ms
    [ INFO ]    Max:           24.60 ms
    [ INFO ] Throughput:   1161.75 FPS

