OpenVINO™ Runtime API Tutorial
==============================

This notebook explains the basics of the OpenVINO Runtime API. It
covers:

-  `Loading OpenVINO Runtime and Showing
   Info <#loading-openvino-runtime-and-showing-info>`__
-  `Loading a Model <#loading-a-model>`__

   -  `OpenVINO IR Model <#openvino-ir-model>`__
   -  `ONNX Model <#onnx-model>`__
   -  `PaddlePaddle Model <#paddlepaddle-model>`__
   -  `TensorFlow Model <#tensorflow-model>`__
   -  `TensorFlow Lite Model <#tensorflow-lite-model>`__

-  `Getting Information about a
   Model <#getting-information-about-a-model>`__

   -  `Model Inputs <#model-inputs>`__
   -  `Model Outputs <#model-outputs>`__

-  `Doing Inference on a Model <#doing-inference-on-a-model>`__
-  `Reshaping and Resizing <#reshaping-and-resizing>`__

   -  `Change Image Size <#change-image-size>`__
   -  `Change Batch Size <#change-batch-size>`__

-  `Caching a Model <#caching-a-model>`__

The notebook is divided into sections with headers. The next cell
contains global requirements installation and imports. Each section is
standalone and does not depend on any previous sections. A segmentation
and classification OpenVINO IR model and a segmentation ONNX model are
provided as examples. These model files can be replaced with your own
models. The exact outputs will be different, but the process is the
same.

.. code:: ipython3

    # Required imports. Please execute this cell first.
    %pip install -q "openvino>=2023.1.0"
    %pip install requests tqdm
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import download_file


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: requests in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2.31.0)
    Requirement already satisfied: tqdm in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.66.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests) (3.3.1)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests) (2023.7.22)
    Note: you may need to restart the kernel to use updated packages.


Loading OpenVINO Runtime and Showing Info
-----------------------------------------

Initialize OpenVINO Runtime with Core()

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()

OpenVINO Runtime can load a network on a device. A device in this
context means a CPU, an Intel GPU, a Neural Compute Stick 2, etc. The
``available_devices`` property shows the available devices in your
system. The “FULL_DEVICE_NAME” option to ``core.get_property()`` shows
the name of the device.

In this notebook, the CPU device is used. To use an integrated GPU, use
``device_name="GPU"`` instead. Be aware that loading a network on GPU
will be slower than loading a network on CPU, but inference will likely
be faster.

.. code:: ipython3

    devices = core.available_devices
    
    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


Loading a Model
---------------

After initializing OpenVINO Runtime, first read the model file with
``read_model()``, then compile it to the specified device with the
``compile_model()`` method.

`OpenVINO™ supports several model
formats <https://docs.openvino.ai/2023.0/Supported_Model_Formats.html#doxid-supported-model-formats>`__
and enables developers to convert them to its own OpenVINO IR format
using a tool dedicated to this task.

OpenVINO IR Model
~~~~~~~~~~~~~~~~~

An OpenVINO IR (Intermediate Representation) model consists of an
``.xml`` file, containing information about network topology, and a
``.bin`` file, containing the weights and biases binary data. Models in
OpenVINO IR format are obtained by using model conversion API. The
``read_model()`` function expects the ``.bin`` weights file to have the
same filename and be located in the same directory as the ``.xml`` file:
``model_weights_file == Path(model_xml).with_suffix(".bin")``. If this
is the case, specifying the weights file is optional. If the weights
file has a different filename, it can be specified using the ``weights``
parameter in ``read_model()``.

The OpenVINO `Model Conversion
API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
tool is used to convert models to OpenVINO IR format. Model conversion
API reads the original model and creates an OpenVINO IR model (``.xml``
and ``.bin`` files) so inference can be performed without delays due to
format conversion. Optionally, model conversion API can adjust the model
to be more suitable for inference, for example, by alternating input
shapes, embedding preprocessing and cutting training parts off. For
information on how to convert your existing TensorFlow, PyTorch or ONNX
model to OpenVINO IR format with model conversion API, refer to the
`tensorflow-to-openvino <101-tensorflow-classification-to-openvino-with-output.html>`__
and
`pytorch-onnx-to-openvino <102-pytorch-onnx-to-openvino-with-output.html>`__
notebooks.

.. code:: ipython3

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'
    
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory='model')
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory='model')



.. parsed-literal::

    model/classification.xml:   0%|          | 0.00/179k [00:00<?, ?B/s]



.. parsed-literal::

    model/classification.bin:   0%|          | 0.00/4.84M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.bin')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    classification_model_xml = "model/classification.xml"
    
    model = core.read_model(model=classification_model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU")

ONNX Model
~~~~~~~~~~

`ONNX <https://onnx.ai/>`__ is an open format built to represent machine
learning models. ONNX defines a common set of operators - the building
blocks of machine learning and deep learning models - and a common file
format to enable AI developers to use models with a variety of
frameworks, tools, runtimes, and compilers. OpenVINO supports reading
models in ONNX format directly,that means they can be used with OpenVINO
Runtime without any prior conversion.

Reading and loading an ONNX model, which is a single ``.onnx`` file,
works the same way as with an OpenVINO IR model. The ``model`` argument
points to the filename of an ONNX model.

.. code:: ipython3

    onnx_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/segmentation.onnx'
    onnx_model_name = 'segmentation.onnx'
    
    download_file(onnx_model_url, filename=onnx_model_name, directory='model')



.. parsed-literal::

    model/segmentation.onnx:   0%|          | 0.00/4.41M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/segmentation.onnx')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    onnx_model_path = "model/segmentation.onnx"
    
    model_onnx = core.read_model(model=onnx_model_path)
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name="CPU")

The ONNX model can be exported to OpenVINO IR with ``save_model()``:

.. code:: ipython3

    ov.save_model(model_onnx, output_model="model/exported_onnx_model.xml")

PaddlePaddle Model
~~~~~~~~~~~~~~~~~~

`PaddlePaddle <https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html>`__
models saved for inference can also be passed to OpenVINO Runtime
without any conversion step. Pass the filename with extension to
``read_model`` and exported an OpenVINO IR with ``save_model``

.. code:: ipython3

    paddle_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    paddle_model_name = 'inference.pdmodel'
    paddle_params_name = 'inference.pdiparams'
    
    download_file(paddle_model_url + paddle_model_name, filename=paddle_model_name, directory='model')
    download_file(paddle_model_url + paddle_params_name, filename=paddle_params_name, directory='model')



.. parsed-literal::

    model/inference.pdmodel:   0%|          | 0.00/1.03M [00:00<?, ?B/s]



.. parsed-literal::

    model/inference.pdiparams:   0%|          | 0.00/21.0M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/inference.pdiparams')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    paddle_model_path = 'model/inference.pdmodel'
    
    model_paddle = core.read_model(model=paddle_model_path)
    compiled_model_paddle = core.compile_model(model=model_paddle, device_name="CPU")

.. code:: ipython3

    ov.save_model(model_paddle, output_model="model/exported_paddle_model.xml")

TensorFlow Model
~~~~~~~~~~~~~~~~

TensorFlow models saved in frozen graph format can also be passed to
``read_model`` starting in OpenVINO 2022.3.

   **NOTE**: Directly loading TensorFlow models is available as a
   preview feature in the OpenVINO 2022.3 release. Fully functional
   support will be provided in the upcoming 2023 releases. Currently
   support is limited to only frozen graph inference format. Other
   TensorFlow model formats must be converted to OpenVINO IR using
   `model conversion
   API <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>`__.

.. code:: ipython3

    pb_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/classification.pb'
    pb_model_name = 'classification.pb'
    
    download_file(pb_model_url, filename=pb_model_name, directory='model')



.. parsed-literal::

    model/classification.pb:   0%|          | 0.00/9.88M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.pb')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    tf_model_path = "model/classification.pb"
    
    model_tf = core.read_model(model=tf_model_path)
    compiled_model_tf = core.compile_model(model=model_tf, device_name="CPU")

.. code:: ipython3

    ov.save_model(model_tf, output_model="model/exported_tf_model.xml")

TensorFlow Lite Model
~~~~~~~~~~~~~~~~~~~~~

`TFLite <https://www.tensorflow.org/lite>`__ models saved for inference
can also be passed to OpenVINO Runtime. Pass the filename with extension
``.tflite`` to ``read_model`` and exported an OpenVINO IR with
``save_model``.

This tutorial uses the image classification model
`inception_v4_quant <https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/default/1>`__.
It is pre-trained model optimized to work with TensorFlow Lite.

.. code:: ipython3

    from pathlib import Path
    
    tflite_model_url = 'https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/default/1?lite-format=tflite'
    tflite_model_path = Path('model/classification.tflite')
    
    download_file(tflite_model_url, filename=tflite_model_path.name, directory=tflite_model_path.parent)



.. parsed-literal::

    model/classification.tflite:   0%|          | 0.00/40.9M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.tflite')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    
    model_tflite = core.read_model(tflite_model_path)
    compiled_model_tflite = core.compile_model(model=model_tflite, device_name="CPU")

.. code:: ipython3

    ov.save_model(model_tflite, output_model="model/exported_tflite_model.xml")

Getting Information about a Model
---------------------------------

The OpenVINO Model instance stores information about the model.
Information about the inputs and outputs of the model are in
``model.inputs`` and ``model.outputs``. These are also properties of the
``CompiledModel`` instance. While using ``model.inputs`` and
``model.outputs`` in the cells below, you can also use
``compiled_model.inputs`` and ``compiled_model.outputs``.

.. code:: ipython3

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'
    
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory='model')
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory='model')


.. parsed-literal::

    'model/classification.xml' already exists.
    'model/classification.bin' already exists.




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.bin')



Model Inputs
~~~~~~~~~~~~

Information about all input layers is stored in the ``inputs``
dictionary.

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    classification_model_xml = "model/classification.xml"
    model = core.read_model(model=classification_model_xml)
    model.inputs




.. parsed-literal::

    [<Output: names[input, input:0] shape[1,3,224,224] type: f32>]



The cell above shows that the loaded model expects one input with the
name *input*. If you loaded a different model, you may see a different
input layer name, and you may see more inputs. You may also obtain info
about each input layer using ``model.input(index)``, where index is a
numeric index of the input layers in the model. If a model has only one
input, index can be omitted.

.. code:: ipython3

    input_layer = model.input(0)

It is often useful to have a reference to the name of the first input
layer. For a model with one input, ``model.input(0).any_name`` gets this
name.

.. code:: ipython3

    input_layer.any_name




.. parsed-literal::

    'input'



The next cell prints the input layout, precision and shape.

.. code:: ipython3

    print(f"input precision: {input_layer.element_type}")
    print(f"input shape: {input_layer.shape}")


.. parsed-literal::

    input precision: <Type: 'float32'>
    input shape: [1,3,224,224]


This cell shows that the model expects inputs with a shape of
[1,3,224,224], and that this is in the ``NCHW`` layout. This means that
the model expects input data with the batch size of 1 (``N``), 3
channels (``C``) , and images with a height (``H``) and width (``W``)
equal to 224. The input data is expected to be of ``FP32`` (floating
point) precision.

Model Outputs
~~~~~~~~~~~~~

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    classification_model_xml = "model/classification.xml"
    model = core.read_model(model=classification_model_xml)
    model.outputs




.. parsed-literal::

    [<Output: names[MobilenetV3/Predictions/Softmax] shape[1,1001] type: f32>]



Model output info is stored in ``model.outputs``. The cell above shows
that the model returns one output, with the
``MobilenetV3/Predictions/Softmax`` name. Loading a different model will
result in different output layer name, and more outputs might be
returned. Similar to input, you may also obtain information about each
output separately using ``model.output(index)``

Since this model has one output, follow the same method as for the input
layer to get its name.

.. code:: ipython3

    output_layer = model.output(0)
    output_layer.any_name




.. parsed-literal::

    'MobilenetV3/Predictions/Softmax'



Getting the output precision and shape is similar to getting the input
precision and shape.

.. code:: ipython3

    print(f"output precision: {output_layer.element_type}")
    print(f"output shape: {output_layer.shape}")


.. parsed-literal::

    output precision: <Type: 'float32'>
    output shape: [1,1001]


This cell shows that the model returns outputs with a shape of [1,
1001], where 1 is the batch size (``N``) and 1001 is the number of
classes (``C``). The output is returned as 32-bit floating point.

Doing Inference on a Model
--------------------------

   **NOTE** this notebook demonstrates only the basic synchronous
   inference API. For an async inference example, please refer to `Async
   API notebook <115-async-api-with-output.html>`__

The diagram below shows a typical inference pipeline with OpenVINO

.. figure:: https://docs.openvino.ai/2023.0/_images/IMPLEMENT_PIPELINE_with_API_C.svg
   :alt: image.png

   image.png

Creating OpenVINO Core and model compilation is covered in the previous
steps. The next step is preparing an inference request. To do inference
on a model, first create an inference request by calling the
``create_infer_request()`` method of ``CompiledModel``,
``compiled_model`` that was loaded with ``compile_model()``. Then, call
the ``infer()`` method of ``InferRequest``. It expects one argument:
``inputs``. This is a dictionary that maps input layer names to input
data or list of input data in ``np.ndarray`` format, where the position
of the input tensor corresponds to input index. If a model has a single
input, wrapping to a dictionary or list can be omitted.

.. code:: ipython3

    # Install opencv package for image handling
    !pip install -q opencv-python

**Load the network**

.. code:: ipython3

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'
    
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory='model')
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory='model')


.. parsed-literal::

    'model/classification.xml' already exists.
    'model/classification.bin' already exists.




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.bin')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    classification_model_xml = "model/classification.xml"
    model = core.read_model(model=classification_model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

**Load an image and convert to the input shape**

To propagate an image through the network, it needs to be loaded into an
array, resized to the shape that the network expects, and converted to
the input layout of the network.

.. code:: ipython3

    import cv2
    
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg",
        directory="data"
    )
    image = cv2.imread(str(image_filename))
    image.shape



.. parsed-literal::

    data/coco_hollywood.jpg:   0%|          | 0.00/485k [00:00<?, ?B/s]




.. parsed-literal::

    (663, 994, 3)



The image has a shape of (663,994,3). It is 663 pixels in height, 994
pixels in width, and has 3 color channels. A reference to the height and
width expected by the network is obtained and the image is resized to
these dimensions.

.. code:: ipython3

    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer.shape
    # OpenCV resize expects the destination size as (width, height).
    resized_image = cv2.resize(src=image, dsize=(W, H))
    resized_image.shape




.. parsed-literal::

    (224, 224, 3)



Now, the image has the width and height that the network expects. This
is still in ``HWC`` format and must be changed to ``NCHW`` format.
First, call the ``np.transpose()`` method to change to ``CHW`` and then
add the ``N`` dimension (where ``N``\ = 1) by calling the
``np.expand_dims()`` method. Next, convert the data to ``FP32`` with
``np.astype()`` method.

.. code:: ipython3

    import numpy as np
    
    input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
    input_data.shape




.. parsed-literal::

    (1, 3, 224, 224)



**Do inference**

Now that the input data is in the right shape, run inference. The
``CompiledModel`` inference result is a dictionary where keys are the
Output class instances (the same keys in ``compiled_model.outputs`` that
can also be obtained with ``compiled_model.output(index)``) and values -
predicted result in ``np.array`` format.

.. code:: ipython3

    # for single input models only
    result = compiled_model(input_data)[output_layer]
    
    # for multiple inputs in a list
    result = compiled_model([input_data])[output_layer]
    
    # or using a dictionary, where the key is input tensor name or index
    result = compiled_model({input_layer.any_name: input_data})[output_layer]

You can also create ``InferRequest`` and run ``infer`` method on
request.

.. code:: ipython3

    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: input_data})
    result = request.get_output_tensor(output_layer.index).data

The ``.infer()`` function sets output tensor, that can be reached, using
``get_output_tensor()``. Since this network returns one output, and the
reference to the output layer is in the ``output_layer.index``
parameter, you can get the data with
``request.get_output_tensor(output_layer.index)``. To get a numpy array
from the output, use the ``.data`` parameter.

.. code:: ipython3

    result.shape




.. parsed-literal::

    (1, 1001)



The output shape is (1,1001), which is the expected output shape. This
shape indicates that the network returns probabilities for 1001 classes.
To learn more about this notion, refer to the `hello world
notebook <001-hello-world-with-output.html>`__.

Reshaping and Resizing
----------------------

Change Image Size
~~~~~~~~~~~~~~~~~

Instead of reshaping the image to fit the model, it is also possible to
reshape the model to fit the image. Be aware that not all models support
reshaping, and models that do, may not support all input shapes. The
model accuracy may also suffer if you reshape the model input shape.

First check the input shape of the model, then reshape it to the new
input shape.

.. code:: ipython3

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'segmentation.xml'
    ir_model_name_bin = 'segmentation.bin'
    
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory='model')
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory='model')



.. parsed-literal::

    model/segmentation.xml:   0%|          | 0.00/1.38M [00:00<?, ?B/s]



.. parsed-literal::

    model/segmentation.bin:   0%|          | 0.00/1.09M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/segmentation.bin')



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    segmentation_model_xml = "model/segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)
    
    print("~~~~ ORIGINAL MODEL ~~~~")
    print(f"input shape: {segmentation_input_layer.shape}")
    print(f"output shape: {segmentation_output_layer.shape}")
    
    new_shape = ov.PartialShape([1, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name="CPU")
    # help(segmentation_compiled_model)
    print("~~~~ RESHAPED MODEL ~~~~")
    print(f"model input shape: {segmentation_input_layer.shape}")
    print(
        f"compiled_model input shape: "
        f"{segmentation_compiled_model.input(index=0).shape}"
    )
    print(f"compiled_model output shape: {segmentation_output_layer.shape}")


.. parsed-literal::

    ~~~~ ORIGINAL MODEL ~~~~
    input shape: [1,3,512,512]
    output shape: [1,1,512,512]
    ~~~~ RESHAPED MODEL ~~~~
    model input shape: [1,3,544,544]
    compiled_model input shape: [1,3,544,544]
    compiled_model output shape: [1,1,544,544]


The input shape for the segmentation network is [1,3,512,512], with the
``NCHW`` layout: the network expects 3-channel images with a width and
height of 512 and a batch size of 1. Reshape the network with the
``.reshape()`` method of ``IENetwork`` to make it accept input images
with a width and height of 544. This segmentation network always returns
arrays with the input width and height of equal value. Therefore,
setting the input dimensions to 544x544 also modifies the output
dimensions. After reshaping, compile the network once again.

Change Batch Size
~~~~~~~~~~~~~~~~~

Use the ``.reshape()`` method to set the batch size, by increasing the
first element of ``new_shape``. For example, to set a batch size of two,
set ``new_shape = (2,3,544,544)`` in the cell above.

.. code:: ipython3

    import openvino as ov
    
    segmentation_model_xml = "model/segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)
    new_shape = ov.PartialShape([2, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name="CPU")
    
    print(f"input shape: {segmentation_input_layer.shape}")
    print(f"output shape: {segmentation_output_layer.shape}")


.. parsed-literal::

    input shape: [2,3,544,544]
    output shape: [2,1,544,544]


The output shows that by setting the batch size to 2, the first element
(``N``) of the input and output shape has a value of 2. Propagate the
input image through the network to see the result:

.. code:: ipython3

    import numpy as np
    import openvino as ov
    
    core = ov.Core()
    segmentation_model_xml = "model/segmentation.xml"
    segmentation_model = core.read_model(model=segmentation_model_xml)
    segmentation_input_layer = segmentation_model.input(0)
    segmentation_output_layer = segmentation_model.output(0)
    new_shape = ov.PartialShape([2, 3, 544, 544])
    segmentation_model.reshape({segmentation_input_layer.any_name: new_shape})
    segmentation_compiled_model = core.compile_model(model=segmentation_model, device_name="CPU")
    input_data = np.random.rand(2, 3, 544, 544)
    
    output = segmentation_compiled_model([input_data])
    
    print(f"input data shape: {input_data.shape}")
    print(f"result data data shape: {segmentation_output_layer.shape}")


.. parsed-literal::

    input data shape: (2, 3, 544, 544)
    result data data shape: [2,1,544,544]


Caching a Model
---------------

For some devices, like GPU, loading a model can take some time. Model
Caching solves this issue by caching the model in a cache directory. If
``core.compile_model(model=net, device_name=device_name, config=config_dict)``
is set, caching will be used. This option checks if a model exists in
the cache. If so, it loads it from the cache. If not, it loads the model
regularly, and stores it in the cache, so that the next time the model
is loaded when this option is set, the model will be loaded from the
cache.

In the cell below, we create a *model_cache* directory as a subdirectory
of *model*, where the model will be cached for the specified device. The
model will be loaded to the GPU. After running this cell once, the model
will be cached, so subsequent runs of this cell will load the model from
the cache.

*Note: Model Caching is also available on CPU devices*

.. code:: ipython3

    ir_model_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/'
    ir_model_name_xml = 'classification.xml'
    ir_model_name_bin = 'classification.bin'
    
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory='model')
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory='model')


.. parsed-literal::

    'model/classification.xml' already exists.
    'model/classification.bin' already exists.




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/002-openvino-api/model/classification.bin')



.. code:: ipython3

    import time
    from pathlib import Path
    
    import openvino as ov
    
    core = ov.Core()
    
    device_name = "GPU" 
    
    if device_name in core.available_devices:
        cache_path = Path("model/model_cache")
        cache_path.mkdir(exist_ok=True)
        # Enable caching for OpenVINO Runtime. To disable caching set enable_caching = False
        enable_caching = True
        config_dict = {"CACHE_DIR": str(cache_path)} if enable_caching else {}
    
        classification_model_xml = "model/classification.xml"
        model = core.read_model(model=classification_model_xml)
    
        start_time = time.perf_counter()
        compiled_model = core.compile_model(model=model, device_name=device_name, config=config_dict)
        end_time = time.perf_counter()
        print(f"Loading the network to the {device_name} device took {end_time-start_time:.2f} seconds.")

After running the previous cell, we know the model exists in the cache
directory. Then, we delete the compiled model and load it again. Now, we
measure the time it takes now.

.. code:: ipython3

    if device_name in core.available_devices:
        del compiled_model
        start_time = time.perf_counter()
        compiled_model = core.compile_model(model=model, device_name=device_name, config=config_dict)
        end_time = time.perf_counter()
        print(f"Loading the network to the {device_name} device took {end_time-start_time:.2f} seconds.")
