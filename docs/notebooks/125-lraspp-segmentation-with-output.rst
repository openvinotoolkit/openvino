Semantic segmentation with LRASPP MobileNet v3 and OpenVINO
===========================================================

The
`torchvision.models <https://pytorch.org/vision/stable/models.html>`__
subpackage contains definitions of models for addressing different
tasks, including: image classification, pixelwise semantic segmentation,
object detection, instance segmentation, person keypoint detection,
video classification, and optical flow. Throughout this notebook we will
show how to use one of them. The LRASPP model is based on the `Searching
for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ paper. According
to the paper, Searching for MobileNetV3, LR-ASPP or Lite Reduced Atrous
Spatial Pyramid Pooling has a lightweight and efficient segmentation
decoder architecture. he model is pre-trained on the `MS
COCO <https://cocodataset.org/#home>`__ dataset. Instead of training on
all 80 classes, the segmentation model has been trained on 20 classes
from the `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__
dataset: **background, aeroplane, bicycle, bird, boat, bottle, bus, car,
cat, chair, cow, dining table, dog, horse, motorbike, person, potted
plant, sheep, sofa, train, tv monitor**

More information about the model is available in the `torchvision
documentation <https://pytorch.org/vision/main/models/lraspp.html>`__

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites) <#prerequisites>`__
-  `Get a test image <#get-a-test-image>`__
-  `Download and prepare a model <#download-and-prepare-a-model>`__
-  `Define a preprocessing and prepare an input
   data <#define-a-preprocessing-and-prepare-an-input-data>`__
-  `Run an inference on the PyTorch
   model) <#run-an-inference-on-the-pytorch-model>`__
-  `Convert the original model to OpenVINO IR
   Format <#convert-the-original-model-to-openvino-ir-format>`__
-  `Run an inference on the OpenVINO
   model) <#run-an-inference-on-the-openvino-model>`__
-  `Show results <#show-results>`__
-  `Show results for the OpenVINO IR
   model) <#show-results-for-the-openvino-ir-model>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
    %pip install -q matplotlib
    %pip install -q "openvino>=2023.2.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    
    import openvino as ov
    import torch

Get a test image
----------------



First of all lets get a test image from an open dataset.

.. code:: ipython3

    import urllib.request
    
    from torchvision.io import read_image
    import torchvision.transforms as transforms
    
    
    img_path = 'cats_image.jpeg'
    urllib.request.urlretrieve(
        url='https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg',
        filename=img_path
    )
    image = read_image(img_path)
    display(transforms.ToPILImage()(image))



.. image:: 125-lraspp-segmentation-with-output_files/125-lraspp-segmentation-with-output_5_0.png


Download and prepare a model
----------------------------



Define width and height of the image that will be used by the network
during inference. According to the input transforms function, the model
is pre-trained on images with a height of 480 and width of 640.

.. code:: ipython3

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

Torchvision provides a mechanism of `listing and retrieving available
models <https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models>`__.

.. code:: ipython3

    import torchvision.models as models
    
    # List available models
    all_models = models.list_models()
    # List of models by type
    segmentation_models = models.list_models(module=models.segmentation)
    
    print(segmentation_models)


.. parsed-literal::

    ['deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'fcn_resnet101', 'fcn_resnet50', 'lraspp_mobilenet_v3_large']


We will use ``lraspp_mobilenet_v3_large``. You can get a model by name
using
``models.get_model("lraspp_mobilenet_v3_large", weights='DEFAULT')`` or
call a `corresponding
function <https://pytorch.org/vision/stable/models/lraspp.html>`__
directly. We will use
``torchvision.models.segmentation.lraspp_mobilenet_v3_large``. You can
directly pass pre-trained model weights to the model initialization
function using weights enum
LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1. It is a
default weights. To get all available weights for the model you can call
``weights_enum = models.get_model_weights("lraspp_mobilenet_v3_large")``,
but there is only one for this model.

.. code:: ipython3

    weights = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)

Define a preprocessing and prepare an input data
------------------------------------------------



You can use ``torchvision.transforms`` to make a preprocessing or
use\ `preprocessing transforms from the model
wight <https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models>`__.

.. code:: ipython3

    import numpy as np
    
    
    preprocess = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    preprocess.resize_size = (IMAGE_HEIGHT, IMAGE_WIDTH)  # change to an image size
    
    input_data = preprocess(image)
    input_data = np.expand_dims(input_data, axis=0)

Run an inference on the PyTorch model
-------------------------------------



.. code:: ipython3

    model.eval()
    with torch.no_grad():
        result_torch = model(torch.as_tensor(input_data).float())['out']

Convert the original model to OpenVINO IR Format
------------------------------------------------



To convert the original model to OpenVINO IR with ``FP16`` precision,
use model conversion API. The models are saved inside the current
directory. For more information on how to convert models, see this
`page <https://docs.openvino.ai/2023.3/openvino_docs_model_processing_introduction.html>`__.

.. code:: ipython3

    ov_model_xml_path = Path('models/ov_lraspp_model.xml')
    
    
    if not ov_model_xml_path.exists():
        ov_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        ov_model = ov.convert_model(model, example_input=dummy_input)
        ov.save_model(ov_model, ov_model_xml_path)
    else:
        print(f"IR model {ov_model_xml_path} already exists.")

Run an inference on the OpenVINO model
--------------------------------------



Select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Run an inference

.. code:: ipython3

    compiled_model = core.compile_model(ov_model_xml_path, device_name=device.value)

.. code:: ipython3

    res_ir = compiled_model(input_data)[0]

Show results
------------



Confirm that the segmentation results look as expected by comparing
model predictions on the OpenVINO IR and PyTorch models.

You can use `pytorch
tutorial <https://pytorch.org/vision/0.12/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py>`__
to visualize segmentation masks. Below is a simple example how to
visualize the image with a ``cat`` mask for the PyTorch model.

.. code:: ipython3

    import torch
    import matplotlib.pyplot as plt
    
    import torchvision.transforms.functional as F
    
    
    plt.rcParams["savefig.bbox"] = 'tight'
    
    
    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

Prepare and display a cat mask.

.. code:: ipython3

    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    
    normalized_mask = torch.nn.functional.softmax(result_torch, dim=1)
    
    cat_mask = normalized_mask[0, sem_class_to_idx['cat']]
    
    show(cat_mask)



.. image:: 125-lraspp-segmentation-with-output_files/125-lraspp-segmentation-with-output_28_0.png


The
`draw_segmentation_masks() <https://pytorch.org/vision/0.12/generated/torchvision.utils.draw_segmentation_masks.html#torchvision.utils.draw_segmentation_masks>`__\ function
can be used to plots those masks on top of the original image. This
function expects the masks to be boolean masks, but our masks above
contain probabilities in [0, 1]. To get boolean masks, we can do the
following:

.. code:: ipython3

    class_dim = 1
    boolean_cat_mask = (normalized_mask.argmax(class_dim) == sem_class_to_idx['cat'])

And now we can plot a boolean mask on top of the original image.

.. code:: ipython3

    from torchvision.utils import draw_segmentation_masks
    
    show(draw_segmentation_masks(image, masks=boolean_cat_mask, alpha=0.7, colors='yellow'))



.. image:: 125-lraspp-segmentation-with-output_files/125-lraspp-segmentation-with-output_32_0.png


Show results for the OpenVINO IR model
--------------------------------------



.. code:: ipython3

    normalized_mask = torch.nn.functional.softmax(torch.from_numpy(res_ir), dim=1)
    boolean_cat_mask = (normalized_mask.argmax(class_dim) == sem_class_to_idx['cat'])
    show(draw_segmentation_masks(image, masks=boolean_cat_mask, alpha=0.7, colors='yellow'))



.. image:: 125-lraspp-segmentation-with-output_files/125-lraspp-segmentation-with-output_34_0.png

