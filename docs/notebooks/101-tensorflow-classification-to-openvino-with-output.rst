Convert a TensorFlow Model to OpenVINO™
=======================================

This short tutorial shows how to convert a TensorFlow
`MobileNetV3 <https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
image classification model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_IR_and_opsets.html>`__
(OpenVINO IR) format, using `Model Conversion
API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.
After creating the OpenVINO IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and do inference with a sample image.

**Table of contents:**

-  `Imports <#imports>`__
-  `Settings <#settings>`__
-  `Download model <#download-model>`__
-  `Convert a Model to OpenVINO IR
   Format <#convert-a-model-to-openvino-ir-format>`__

   -  `Convert a TensorFlow Model to OpenVINO IR
      Format <#convert-a-tensorflow-model-to-openvino-ir-format>`__

-  `Test Inference on the Converted
   Model <#test-inference-on-the-converted-model>`__

   -  `Load the Model <#load-the-model>`__

-  `Select inference device <#select-inference-device>`__

   -  `Get Model Information <#get-model-information>`__
   -  `Load an Image <#load-an-image>`__
   -  `Do Inference <#do-inference>`__

-  `Timing <#timing>`__

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import time
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    import tensorflow as tf
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import download_file


.. parsed-literal::

    2023-11-14 22:30:46.626761: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-11-14 22:30:46.661288: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-11-14 22:30:47.171314: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Settings
--------



.. code:: ipython3

    # The paths of the source and converted models.
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = Path("model/v3-small_224_1.0_float")
    
    ir_path = Path("model/v3-small_224_1.0_float.xml")

Download model
--------------



Load model using `tf.keras.applications
api <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small>`__
and save it to the disk.

.. code:: ipython3

    model = tf.keras.applications.MobileNetV3Small()
    model.save(model_path)


.. parsed-literal::

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.


.. parsed-literal::

    2023-11-14 22:30:50.201471: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2023-11-14 22:30:50.201504: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2023-11-14 22:30:50.201508: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2023-11-14 22:30:50.201646: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2023-11-14 22:30:50.201662: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2023-11-14 22:30:50.201665: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration


.. parsed-literal::

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


.. parsed-literal::

    2023-11-14 22:30:54.370304: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,1,1,1024]
    	 [[{{node inputs}}]]
    2023-11-14 22:30:57.509389: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,1,1,1024]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 54). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/v3-small_224_1.0_float/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/v3-small_224_1.0_float/assets


Convert a Model to OpenVINO IR Format
-------------------------------------



Convert a TensorFlow Model to OpenVINO IR Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Use the model conversion Python API to convert the TensorFlow model to
OpenVINO IR. The ``ov.convert_model`` function accept path to saved
model directory and returns OpenVINO Model class instance which
represents this model. Obtained model is ready to use and to be loaded
on a device using ``ov.compile_model`` or can be saved on a disk using
the ``ov.save_model`` function. See the
`tutorial <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>`__
for more information about using model conversion API with TensorFlow
models.

.. code:: ipython3

    # Run model conversion API if the IR model file does not exist
    if not ir_path.exists():
        print("Exporting TensorFlow model to IR... This may take a few minutes.")
        ov_model = ov.convert_model(model_path, input=[[1, 224, 224, 3]])
        ov.save_model(ov_model, ir_path)
    else:
        print(f"IR model {ir_path} already exists.")


.. parsed-literal::

    Exporting TensorFlow model to IR... This may take a few minutes.


Test Inference on the Converted Model
-------------------------------------



Load the Model
~~~~~~~~~~~~~~



.. code:: ipython3

    core = ov.Core()
    model = core.read_model(ir_path)

Select inference device
-----------------------



select device from dropdown list for running inference using OpenVINO

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

    compiled_model = core.compile_model(model=model, device_name=device.value)

Get Model Information
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    network_input_shape = input_key.shape 

Load an Image
~~~~~~~~~~~~~



Load an image, resize it, and convert it to the input shape of the
network.

.. code:: ipython3

    # Download the image from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data"
    )
    
    # The MobileNet network expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)
    
    # Resize the image to the network input shape.
    resized_image = cv2.resize(src=image, dsize=(224, 224))
    
    # Transpose the image to the network input shape.
    input_image = np.expand_dims(resized_image, 0)
    
    plt.imshow(image);



.. parsed-literal::

    data/coco.jpg:   0%|          | 0.00/202k [00:00<?, ?B/s]



.. image:: 101-tensorflow-classification-to-openvino-with-output_files/101-tensorflow-classification-to-openvino-with-output_19_1.png


Do Inference
~~~~~~~~~~~~



.. code:: ipython3

    result = compiled_model(input_image)[output_key]
    
    result_index = np.argmax(result)

.. code:: ipython3

    # Download the datasets from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data"
    )
    
    # Convert the inference result to a class name.
    imagenet_classes = image_filename.read_text().splitlines()
    
    imagenet_classes[result_index]



.. parsed-literal::

    data/imagenet_2012.txt:   0%|          | 0.00/30.9k [00:00<?, ?B/s]




.. parsed-literal::

    'n02099267 flat-coated retriever'



Timing
------



Measure the time it takes to do inference on thousand images. This gives
an indication of performance. For more accurate benchmarking, use the
`Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
in OpenVINO. Note that many optimizations are possible to improve the
performance.

.. code:: ipython3

    num_images = 1000
    
    start = time.perf_counter()
    
    for _ in range(num_images):
        compiled_model([input_image])
    
    end = time.perf_counter()
    time_ir = end - start
    
    print(
        f"IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )


.. parsed-literal::

    IR model in OpenVINO Runtime/CPU: 0.0010 seconds per image, FPS: 962.52

