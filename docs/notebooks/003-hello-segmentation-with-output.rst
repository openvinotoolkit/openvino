Hello Image Segmentation
========================

A very basic introduction to using segmentation models with OpenVINO™.

In this tutorial, a pre-trained
`road-segmentation-adas-0001 <https://docs.openvino.ai/2023.0/omz_models_model_road_segmentation_adas_0001.html>`__
model from the `Open Model Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__ is used.
ADAS stands for Advanced Driver Assistance Services. The model
recognizes four classes: background, road, curb and mark.

**Table of contents:**

- `Imports <#imports>`__
- `Download model weights <#download-model-weights>`__
- `Select inference device <#select-inference-device>`__
- `Load the Model <#load-the-model>`__
- `Load an Image <#load-an-image>`__
- `Do Inference <#do-inference>`__
- `Prepare Data for Visualization <#prepare-data-for-visualization>`__
- `Visualize data <#visualize-data>`__

.. code:: ipython3

    # Install openvino package
    !pip install -q "openvino==2023.1.0.dev20230811"

Imports
#########################################

.. code:: ipython3

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    import sys
    
    sys.path.append("../utils")
    from notebook_utils import segmentation_map_to_image, download_file

Download model weights
#############################################################################################################################

.. code:: ipython3

    from pathlib import Path
    
    base_model_dir = Path("./model").expanduser()
    
    model_name = "road-segmentation-adas-0001"
    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'
    
    model_xml_path = base_model_dir / model_xml_name
    
    if not model_xml_path.exists():
        model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml"
        model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin"
    
        download_file(model_xml_url, model_xml_name, base_model_dir)
        download_file(model_bin_url, model_bin_name, base_model_dir)
    else:
        print(f'{model_name} already downloaded to {base_model_dir}')



.. parsed-literal::

    model/road-segmentation-adas-0001.xml:   0%|          | 0.00/389k [00:00<?, ?B/s]



.. parsed-literal::

    model/road-segmentation-adas-0001.bin:   0%|          | 0.00/720k [00:00<?, ?B/s]


Select inference device
#############################################################################################################################

Select device from dropdown list for running inference using OpenVINO:

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



Load the Model
#############################################################################################################################

.. code:: ipython3

    core = ov.Core()
    
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)

Load an Image
#############################################################################################################################

A sample image from the `Mapillary Vistas <https://www.mapillary.com/dataset/vistas>`__ dataset is
provided.

.. code:: ipython3

    # The segmentation network expects images in BGR format.
    image = cv2.imread("../data/image/empty_road_mapillary.jpg")
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = image.shape
    
    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer_ir.shape
    
    # OpenCV resize expects the destination size as (width, height).
    resized_image = cv2.resize(image, (W, H))
    
    # Reshape to the network input shape.
    input_image = np.expand_dims(
        resized_image.transpose(2, 0, 1), 0
    )  
    plt.imshow(rgb_image)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fe21c3c5970>




.. image:: 003-hello-segmentation-with-output_files/003-hello-segmentation-with-output_11_1.png


Do Inference
#############################################################################################################################

.. code:: ipython3

    # Run the inference.
    result = compiled_model([input_image])[output_layer_ir]
    
    # Prepare data for visualization.
    segmentation_mask = np.argmax(result, axis=1)
    plt.imshow(segmentation_mask.transpose(1, 2, 0))




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fe21c2a7940>




.. image:: 003-hello-segmentation-with-output_files/003-hello-segmentation-with-output_13_1.png


Prepare Data for Visualization
#############################################################################################################################

.. code:: ipython3

    # Define colormap, each color represents a class.
    colormap = np.array([[68, 1, 84], [48, 103, 141], [53, 183, 120], [199, 216, 52]])
    
    # Define the transparency of the segmentation mask on the photo.
    alpha = 0.3
    
    # Use function from notebook_utils.py to transform mask to an RGB image.
    mask = segmentation_map_to_image(segmentation_mask, colormap)
    resized_mask = cv2.resize(mask, (image_w, image_h))
    
    # Create an image with mask.
    image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)

Visualize data
#############################################################################################################################

.. code:: ipython3

    # Define titles with images.
    data = {"Base Photo": rgb_image, "Segmentation": mask, "Masked Photo": image_with_mask}
    
    # Create a subplot to visualize images.
    fig, axs = plt.subplots(1, len(data.items()), figsize=(15, 10))
    
    # Fill the subplot.
    for ax, (name, image) in zip(axs, data.items()):
        ax.axis('off')
        ax.set_title(name)
        ax.imshow(image)
    
    # Display an image.
    plt.show(fig)



.. image:: 003-hello-segmentation-with-output_files/003-hello-segmentation-with-output_17_0.png

