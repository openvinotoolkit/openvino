Classification with ConvNeXt and OpenVINO
=========================================

The
`torchvision.models <https://pytorch.org/vision/stable/models.html>`__
subpackage contains definitions of models for addressing different
tasks, including: image classification, pixelwise semantic segmentation,
object detection, instance segmentation, person keypoint detection,
video classification, and optical flow. Throughout this notebook we will
show how to use one of them.

The ConvNeXt model is based on the `A ConvNet for the
2020s <https://arxiv.org/abs/2201.03545>`__ paper. The outcome of this
exploration is a family of pure ConvNet models dubbed ConvNeXt.
Constructed entirely from standard ConvNet modules, ConvNeXts compete
favorably with Transformers in terms of accuracy and scalability,
achieving 87.8% ImageNet top-1 accuracy and outperforming Swin
Transformers on COCO detection and ADE20K segmentation, while
maintaining the simplicity and efficiency of standard ConvNets. The
``torchvision.models`` subpackage
`contains <https://pytorch.org/vision/main/models/convnext.html>`__
several pretrained ConvNeXt model. In this tutorial we will use ConvNeXt
Tiny model.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Get a test image <#get-a-test-image>`__
-  `Get a pretrained model <#get-a-pretrained-model>`__
-  `Define a preprocessing and prepare an input
   data <#define-a-preprocessing-and-prepare-an-input-data>`__
-  `Use the original model to run an
   inference <#use-the-original-model-to-run-an-inference>`__
-  `Convert the model to OpenVINO Intermediate representation
   format <#convert-the-model-to-openvino-intermediate-representation-format>`__
-  `Use the OpenVINO IR model to run an
   inference <#use-the-openvino-ir-model-to-run-an-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
    %pip install -q  "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Get a test image
----------------

First of all lets get a test
image from an open dataset.

.. code:: ipython3

    import requests
    
    from torchvision.io import read_image
    import torchvision.transforms as transforms
    
    
    img_path = "cats_image.jpeg"
    r = requests.get("https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg")
    
    with open(img_path, "wb") as f:
        f.write(r.content)
    image = read_image(img_path)
    display(transforms.ToPILImage()(image))



.. image:: convnext-classification-with-output_files/convnext-classification-with-output_4_0.png


Get a pretrained model
----------------------

Torchvision provides a
mechanism of `listing and retrieving available
models <https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models>`__.

.. code:: ipython3

    import torchvision.models as models
    
    # List available models
    all_models = models.list_models()
    # List of models by type. Classification models are in the parent module.
    classification_models = models.list_models(module=models)
    
    print(classification_models)


.. parsed-literal::

    ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']


We will use ``convnext_tiny``. To get a pretrained model just use
``models.get_model("convnext_tiny", weights='DEFAULT')`` or a specific
method of ``torchvision.models`` for this model using `default
weights <https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights>`__
that is equivalent to ``ConvNeXt_Tiny_Weights.IMAGENET1K_V1``. If you
don’t specify ``weight`` or specify ``weights=None`` it will be a random
initialization. To get all available weights for the model you can call
``weights_enum = models.get_model_weights("convnext_tiny")``, but there
is only one for this model. You can find more information how to
initialize pre-trained models
`here <https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models>`__.

.. code:: ipython3

    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

Define a preprocessing and prepare an input data
------------------------------------------------

You can use
``torchvision.transforms`` to make a preprocessing or
use\ `preprocessing transforms from the model
wight <https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models>`__.

.. code:: ipython3

    import torch
    
    
    preprocess = models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    
    input_data = preprocess(image)
    input_data = torch.stack([input_data], dim=0)

Use the original model to run an inference
------------------------------------------



.. code:: ipython3

    outputs = model(input_data)

And print results

.. code:: ipython3

    # download class number to class label mapping
    imagenet_classes_file_path = "imagenet_2012.txt"
    r = requests.get(
        url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
    )
    
    with open(imagenet_classes_file_path, "w") as f:
        f.write(r.text)
    
    imagenet_classes = open(imagenet_classes_file_path).read().splitlines()
    
    
    def print_results(outputs: torch.Tensor):
        _, predicted_class = outputs.max(1)
        predicted_probability = torch.softmax(outputs, dim=1)[0, predicted_class].item()
    
        print(f"Predicted Class: {predicted_class.item()}")
        print(f"Predicted Label: {imagenet_classes[predicted_class.item()]}")
        print(f"Predicted Probability: {predicted_probability}")

.. code:: ipython3

    print_results(outputs)


.. parsed-literal::

    Predicted Class: 281
    Predicted Label: n02123045 tabby, tabby cat
    Predicted Probability: 0.5351971983909607


Convert the model to OpenVINO Intermediate representation format
----------------------------------------------------------------



OpenVINO supports PyTorch through conversion to OpenVINO Intermediate
Representation (IR) format. To take the advantage of OpenVINO
optimization tools and features, the model should be converted using the
OpenVINO Converter tool (OVC). The ``openvino.convert_model`` function
provides Python API for OVC usage. The function returns the instance of
the OpenVINO Model class, which is ready for use in the Python
interface. However, it can also be saved on disk using
``openvino.save_model`` for future execution.

.. code:: ipython3

    from pathlib import Path
    
    import openvino as ov
    
    
    ov_model_xml_path = Path("models/ov_convnext_model.xml")
    
    if not ov_model_xml_path.exists():
        ov_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(model, example_input=torch.randn(1, 3, 224, 224))
        # add transform to OpenVINO preprocessing converting
        ov.save_model(converted_model, ov_model_xml_path)
    else:
        print(f"IR model {ov_model_xml_path} already exists.")

When the ``openvino.save_model`` function is used, an OpenVINO model is
serialized in the file system as two files with ``.xml`` and ``.bin``
extensions. This pair of files is called OpenVINO Intermediate
Representation format (OpenVINO IR, or just IR) and useful for efficient
model deployment. OpenVINO IR can be loaded into another application for
inference using the ``openvino.Core.read_model`` function.

Select device from dropdown list for running inference using OpenVINO

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



.. code:: ipython3

    core = ov.Core()
    
    compiled_model = core.compile_model(ov_model_xml_path, device_name=device.value)

Use the OpenVINO IR model to run an inference
---------------------------------------------



.. code:: ipython3

    outputs = compiled_model(input_data)[0]
    print_results(torch.from_numpy(outputs))


.. parsed-literal::

    Predicted Class: 281
    Predicted Label: n02123045 tabby, tabby cat
    Predicted Probability: 0.5664422512054443

