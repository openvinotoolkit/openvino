Hello Image Classification
==========================

This basic introduction to OpenVINO™ shows how to do inference with an
image classification model.

A pre-trained `MobileNetV3
model <https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__ is used in
this tutorial. For more information about how OpenVINO IR models are
created, refer to the `TensorFlow to
OpenVINO <101-tensorflow-classification-to-openvino-with-output.html>`__
tutorial.

**Table of contents:**


-  `Imports <#imports>`__
-  `Download the Model and data samples <#download-the-model-and-data-samples>`__
-  `Select inference device <#select-inference-device>`__
-  `Load the Model <#load-the-model>`__
-  `Load an Image <#load-an-image>`__
-  `Do Inference <#do-inference>`__

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino>=2023.1.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports 
-------------------------------------------------

.. code:: ipython3

    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import download_file

Download the Model and data samples 
-----------------------------------------------------------------------------

.. code:: ipython3

    base_artifacts_dir = Path('./artifacts').expanduser()
    
    model_name = "v3-small_224_1.0_float"
    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'
    
    model_xml_path = base_artifacts_dir / model_xml_name
    
    base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'
    
    if not model_xml_path.exists():
        download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
        download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
    else:
        print(f'{model_name} already downloaded to {base_artifacts_dir}')



.. parsed-literal::

    artifacts/v3-small_224_1.0_float.xml:   0%|          | 0.00/294k [00:00<?, ?B/s]



.. parsed-literal::

    artifacts/v3-small_224_1.0_float.bin:   0%|          | 0.00/4.84M [00:00<?, ?B/s]


Select inference device 
-----------------------------------------------------------------

select device from dropdown list for running inference using OpenVINO

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
--------------------------------------------------------

.. code:: ipython3

    core = ov.Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    
    output_layer = compiled_model.output(0)

Load an Image 
-------------------------------------------------------

.. code:: ipython3

    # Download the image from the openvino_notebooks storage
    image_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data"
    )
    
    # The MobileNet model expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)
    
    # Resize to MobileNet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))
    
    # Reshape to model input shape.
    input_image = np.expand_dims(input_image, 0)
    plt.imshow(image);



.. parsed-literal::

    data/coco.jpg:   0%|          | 0.00/202k [00:00<?, ?B/s]



.. image:: 001-hello-world-with-output_files/001-hello-world-with-output_11_1.png


Do Inference 
------------------------------------------------------

.. code:: ipython3

    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)

.. code:: ipython3

    imagenet_filename = download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data"
    )
    
    imagenet_classes = imagenet_filename.read_text().splitlines()



.. parsed-literal::

    data/imagenet_2012.txt:   0%|          | 0.00/30.9k [00:00<?, ?B/s]


.. code:: ipython3

    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes
    
    imagenet_classes[result_index]




.. parsed-literal::

    'n02099267 flat-coated retriever'


