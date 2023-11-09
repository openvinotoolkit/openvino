Convert a PyTorch Model to ONNX and OpenVINO™ IR
================================================

This tutorial demonstrates step-by-step instructions on how to do
inference on a PyTorch semantic segmentation model, using OpenVINO
Runtime.

First, the PyTorch model is exported in `ONNX <https://onnx.ai/>`__
format and then converted to OpenVINO IR. Then the respective ONNX and
OpenVINO IR models are loaded into OpenVINO Runtime to show model
predictions. In this tutorial, we will use LR-ASPP model with
MobileNetV3 backbone.

According to the paper, `Searching for
MobileNetV3 <https://arxiv.org/pdf/1905.02244.pdf>`__, LR-ASPP or Lite
Reduced Atrous Spatial Pyramid Pooling has a lightweight and efficient
segmentation decoder architecture. The diagram below illustrates the
model architecture:

.. figure:: https://user-images.githubusercontent.com/29454499/207099169-48dca3dc-a8eb-4e11-be92-40cebeec7a88.png
   :alt: image

   image

The model is pre-trained on the `MS
COCO <https://cocodataset.org/#home>`__ dataset. Instead of training on
all 80 classes, the segmentation model has been trained on 20 classes
from the `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__
dataset: **background, aeroplane, bicycle, bird, boat, bottle, bus, car,
cat, chair, cow, dining table, dog, horse, motorbike, person, potted
plant, sheep, sofa, train, tv monitor**

More information about the model is available in the `torchvision
documentation <https://pytorch.org/vision/main/models/lraspp.html>`__

**Table of contents:**


-  `Preparation <#preparation>`__

   -  `Imports <#imports>`__
   -  `Settings <#settings>`__
   -  `Load Model <#load-model>`__

-  `ONNX Model Conversion <#onnx-model-conversion>`__

   -  `Convert PyTorch model to
      ONNX <#convert-pytorch-model-to-onnx>`__
   -  `Convert ONNX Model to OpenVINO IR
      Format <#convert-onnx-model-to-openvino-ir-format>`__

-  `Show Results <#show-results>`__

   -  `Load and Preprocess an Input
      Image <#load-and-preprocess-an-input-image>`__
   -  `Load the OpenVINO IR Network and Run Inference on the ONNX
      model <#load-the-openvino-ir-network-and-run-inference-on-the-onnx-model>`__

      -  `1. ONNX Model in OpenVINO
         Runtime <#-onnx-model-in-openvino-runtime>`__
      -  `Select inference device <#select-inference-device>`__
      -  `2. OpenVINO IR Model in OpenVINO
         Runtime <#-openvino-ir-model-in-openvino-runtime>`__
      -  `Select inference device <#select-inference-device>`__

-  `PyTorch Comparison <#pytorch-comparison>`__
-  `Performance Comparison <#performance-comparison>`__
-  `References <#references>`__

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino>=2023.1.0" onnx


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Preparation 
-----------------------------------------------------

Imports 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import time
    import warnings
    from pathlib import Path
    
    import cv2
    import numpy as np
    import openvino as ov
    import torch
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import segmentation_map_to_image, viz_result_image, SegmentationMap, Label, download_file

Settings 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set a name for the model, then define width and height of the image that
will be used by the network during inference. According to the input
transforms function, the model is pre-trained on images with a height of
520 and width of 780.

.. code:: ipython3

    IMAGE_WIDTH = 780
    IMAGE_HEIGHT = 520
    DIRECTORY_NAME = "model"
    BASE_MODEL_NAME = DIRECTORY_NAME + "/lraspp_mobilenet_v3_large"
    weights_path = Path(BASE_MODEL_NAME + ".pt")
    
    # Paths where ONNX and OpenVINO IR models will be stored.
    onnx_path = weights_path.with_suffix('.onnx')
    if not onnx_path.parent.exists():
        onnx_path.parent.mkdir()
    ir_path = onnx_path.with_suffix(".xml")

Load Model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, PyTorch models represent an instance of ``torch.nn.Module``
class, initialized by a state dictionary with model weights. Typical
steps for getting a pre-trained model:

1. Create instance of model class
2. Load checkpoint state dict, which contains pre-trained model weights
3. Turn model to evaluation for switching some operations to inference
mode

The ``torchvision`` module provides a ready to use set of functions for
model class initialization. We will use
``torchvision.models.segmentation.lraspp_mobilenet_v3_large``. You can
directly pass pre-trained model weights to the model initialization
function using weights enum
``LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1``. However,
for demonstration purposes, we will create it separately. Download the
pre-trained weights and load the model. This may take some time if you
have not downloaded the model before.

.. code:: ipython3

    print("Downloading the LRASPP MobileNetV3 model (if it has not been downloaded already)...") 
    download_file(LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.url, filename=weights_path.name, directory=weights_path.parent)
    # create model object
    model = lraspp_mobilenet_v3_large()
    # read state dict, use map_location argument to avoid a situation where weights are saved in cuda (which may not be unavailable on the system)
    state_dict = torch.load(weights_path, map_location='cpu')
    # load state dict to model
    model.load_state_dict(state_dict)
    # switch model from training to inference mode
    model.eval()
    print("Loaded PyTorch LRASPP MobileNetV3 model")


.. parsed-literal::

    Downloading the LRASPP MobileNetV3 model (if it has not been downloaded already)...



.. parsed-literal::

    model/lraspp_mobilenet_v3_large.pt:   0%|          | 0.00/12.5M [00:00<?, ?B/s]


.. parsed-literal::

    Loaded PyTorch LRASPP MobileNetV3 model


ONNX Model Conversion 
---------------------------------------------------------------

Convert PyTorch model to ONNX 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenVINO supports PyTorch models that are exported in ONNX format. We
will use the ``torch.onnx.export`` function to obtain the ONNX model,
you can learn more about this feature in the `PyTorch
documentation <https://pytorch.org/docs/stable/onnx.html>`__. We need to
provide a model object, example input for model tracing and path where
the model will be saved. When providing example input, it is not
necessary to use real data, dummy input data with specified shape is
sufficient. Optionally, we can provide a target onnx opset for
conversion and/or other parameters specified in documentation
(e.g. input and output names or dynamic shapes).

Sometimes a warning will be shown, but in most cases it is harmless, so
let us just filter it out. When the conversion is successful, the last
line of the output will read:
``ONNX model exported to model/lraspp_mobilenet_v3_large.onnx.``

.. code:: ipython3

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if not onnx_path.exists():
            dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
            )
            print(f"ONNX model exported to {onnx_path}.")
        else:
            print(f"ONNX model {onnx_path} already exists.")


.. parsed-literal::

    ONNX model exported to model/lraspp_mobilenet_v3_large.onnx.


Convert ONNX Model to OpenVINO IR Format 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert the ONNX model to OpenVINO IR with ``FP16`` precision, use
model conversion API. The models are saved inside the current directory.
For more information on how to convert models, see this
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.

.. code:: ipython3

    if not ir_path.exists():
        print("Exporting ONNX model to IR... This may take a few minutes.")
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, ir_path)
    else:
        print(f"IR model {ir_path} already exists.")


.. parsed-literal::

    Exporting ONNX model to IR... This may take a few minutes.


Show Results 
------------------------------------------------------

Confirm that the segmentation results look as expected by comparing
model predictions on the ONNX, OpenVINO IR and PyTorch models.

Load and Preprocess an Input Image 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Images need to be normalized before propagating through the network.

.. code:: ipython3

    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the given mean and standard deviation
        for CityScapes models.
        """
        image = image.astype(np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image /= 255.0
        image -= mean
        image /= std
        return image

.. code:: ipython3

    # Download the image from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data"
    )
    
    image = cv2.cvtColor(cv2.imread(str(image_filename)), cv2.COLOR_BGR2RGB)
    
    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized_image = normalize(resized_image)
    
    # Convert the resized images to network input shape.
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)



.. parsed-literal::

    data/coco.jpg:   0%|          | 0.00/202k [00:00<?, ?B/s]


Load the OpenVINO IR Network and Run Inference on the ONNX model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenVINO Runtime can load ONNX models directly. First, load the ONNX
model, do inference and show the results. Then, load the model that was
converted to OpenVINO Intermediate Representation (OpenVINO IR) with
OpenVINO Converter and do inference on that model, and show the results
on an image.

1. ONNX Model in OpenVINO Runtime 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # Instantiate OpenVINO Core
    core = ov.Core()
    
    # Read model to OpenVINO Runtime
    model_onnx = core.read_model(model=onnx_path)

Select inference device 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    # Load model on device
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name=device.value)
    
    # Run inference on the input image
    res_onnx = compiled_model_onnx([normalized_input_image])[0]

Model predicts probabilities for how well each pixel corresponds to a
specific label. To get the label with highest probability for each
pixel, operation argmax should be applied. After that, color coding can
be applied to each label for more convenient visualization.

.. code:: ipython3

    voc_labels = [
        Label(index=0, color=(0, 0, 0), name="background"),
        Label(index=1, color=(128, 0, 0), name="aeroplane"),
        Label(index=2, color=(0, 128, 0), name="bicycle"),
        Label(index=3, color=(128, 128, 0), name="bird"),
        Label(index=4, color=(0, 0, 128), name="boat"),
        Label(index=5, color=(128, 0, 128), name="bottle"),
        Label(index=6, color=(0, 128, 128), name="bus"),
        Label(index=7, color=(128, 128, 128), name="car"),
        Label(index=8, color=(64, 0, 0), name="cat"),
        Label(index=9, color=(192, 0, 0), name="chair"),
        Label(index=10, color=(64, 128, 0), name="cow"),
        Label(index=11, color=(192, 128, 0), name="dining table"),
        Label(index=12, color=(64, 0, 128), name="dog"),
        Label(index=13, color=(192, 0, 128), name="horse"),
        Label(index=14, color=(64, 128, 128), name="motorbike"),
        Label(index=15, color=(192, 128, 128), name="person"),
        Label(index=16, color=(0, 64, 0), name="potted plant"),
        Label(index=17, color=(128, 64, 0), name="sheep"),
        Label(index=18, color=(0, 192, 0), name="sofa"),
        Label(index=19, color=(128, 192, 0), name="train"),
        Label(index=20, color=(0, 64, 128), name="tv monitor")
    ]
    VOCLabels = SegmentationMap(voc_labels)
    
    # Convert the network result to a segmentation map and display the result.
    result_mask_onnx = np.squeeze(np.argmax(res_onnx, axis=1)).astype(np.uint8)
    viz_result_image(
        image,
        segmentation_map_to_image(result_mask_onnx, VOCLabels.get_colormap()),
        resize=True,
    )




.. image:: 102-pytorch-onnx-to-openvino-with-output_files/102-pytorch-onnx-to-openvino-with-output_22_0.png



2. OpenVINO IR Model in OpenVINO Runtime 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Select inference device 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Load the network in OpenVINO Runtime.
    core = ov.Core()
    model_ir = core.read_model(model=ir_path)
    compiled_model_ir = core.compile_model(model=model_ir, device_name=device.value)
    
    # Get input and output layers.
    output_layer_ir = compiled_model_ir.output(0)
    
    # Run inference on the input image.
    res_ir = compiled_model_ir([normalized_input_image])[output_layer_ir]

.. code:: ipython3

    result_mask_ir = np.squeeze(np.argmax(res_ir, axis=1)).astype(np.uint8)
    viz_result_image(
        image,
        segmentation_map_to_image(result=result_mask_ir, colormap=VOCLabels.get_colormap()),
        resize=True,
    )




.. image:: 102-pytorch-onnx-to-openvino-with-output_files/102-pytorch-onnx-to-openvino-with-output_27_0.png



PyTorch Comparison 
------------------------------------------------------------

Do inference on the PyTorch model to verify that the output visually
looks the same as the output on the ONNX/OpenVINO IR models.

.. code:: ipython3

    model.eval()
    with torch.no_grad():
        result_torch = model(torch.as_tensor(normalized_input_image).float())
    
    result_mask_torch = torch.argmax(result_torch['out'], dim=1).squeeze(0).numpy().astype(np.uint8)
    viz_result_image(
        image,
        segmentation_map_to_image(result=result_mask_torch, colormap=VOCLabels.get_colormap()),
        resize=True,
    )




.. image:: 102-pytorch-onnx-to-openvino-with-output_files/102-pytorch-onnx-to-openvino-with-output_29_0.png



Performance Comparison 
----------------------------------------------------------------

Measure the time it takes to do inference on twenty images. This gives
an indication of performance. For more accurate benchmarking, use the
`Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__.
Keep in mind that many optimizations are possible to improve the
performance.

.. code:: ipython3

    num_images = 100
    
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_images):
            model(torch.as_tensor(input_image).float())
        end = time.perf_counter()
        time_torch = end - start
    print(
        f"PyTorch model on CPU: {time_torch/num_images:.3f} seconds per image, "
        f"FPS: {num_images/time_torch:.2f}"
    )
    
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name="CPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_onnx([normalized_input_image])
    end = time.perf_counter()
    time_onnx = end - start
    print(
        f"ONNX model in OpenVINO Runtime/CPU: {time_onnx/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx:.2f}"
    )
    
    compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_ir([input_image])
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"OpenVINO IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )
    
    if "GPU" in core.available_devices:
        compiled_model_onnx_gpu = core.compile_model(model=model_onnx, device_name="GPU")
        start = time.perf_counter()
        for _ in range(num_images):
            compiled_model_onnx_gpu([input_image])
        end = time.perf_counter()
        time_onnx_gpu = end - start
        print(
            f"ONNX model in OpenVINO/GPU: {time_onnx_gpu/num_images:.3f} "
            f"seconds per image, FPS: {num_images/time_onnx_gpu:.2f}"
        )
    
        compiled_model_ir_gpu = core.compile_model(model=model_ir, device_name="GPU")
        start = time.perf_counter()
        for _ in range(num_images):
            compiled_model_ir_gpu([input_image])
        end = time.perf_counter()
        time_ir_gpu = end - start
        print(
            f"IR model in OpenVINO/GPU: {time_ir_gpu/num_images:.3f} "
            f"seconds per image, FPS: {num_images/time_ir_gpu:.2f}"
        )


.. parsed-literal::

    PyTorch model on CPU: 0.044 seconds per image, FPS: 22.94
    ONNX model in OpenVINO Runtime/CPU: 0.020 seconds per image, FPS: 49.24
    OpenVINO IR model in OpenVINO Runtime/CPU: 0.032 seconds per image, FPS: 30.92


**Show Device Information**

.. code:: ipython3

    devices = core.available_devices
    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


References 
----------------------------------------------------

-  `Torchvision <https://pytorch.org/vision/stable/index.html>`__
-  `Pytorch ONNX
   Documentation <https://pytorch.org/docs/stable/onnx.html>`__
-  `PIP install openvino-dev <https://pypi.org/project/openvino-dev/>`__
-  `OpenVINO ONNX
   support <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_ONNX_Support.html>`__
-  `Model Conversion API
   documentation <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
-  `Converting Pytorch
   model <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch.html>`__
