Object segmentations with FastSAM and OpenVINO
==============================================

`The Fast Segment Anything Model
(FastSAM) <https://docs.ultralytics.com/models/fast-sam/>`__ is a
real-time CNN-based model that can segment any object within an image
based on various user prompts. ``Segment Anything`` task is designed to
make vision tasks easier by providing an efficient way to identify
objects in an image. FastSAM significantly reduces computational demands
while maintaining competitive performance, making it a practical choice
for a variety of vision tasks.

FastSAM is a model that aims to overcome the limitations of the `Segment
Anything Model (SAM) <https://docs.ultralytics.com/models/sam/>`__,
which is a Transformer model that requires significant computational
resources. FastSAM tackles the segment anything task by dividing it into
two consecutive stages: all-instance segmentation and prompt-guided
selection.

In the first stage,
```YOLOv8-seg`` <https://docs.ultralytics.com/tasks/segment/>`__ is used
to produce segmentation masks for all instances in the image. In the
second stage, FastSAM outputs the region-of-interest corresponding to
the prompt.

.. figure:: https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg
   :alt: pipeline

   pipeline

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#Prerequisites>`__

   -  `Install requirements <#Install-requirements>`__
   -  `Imports <#Imports>`__

-  `FastSAM in Ultralytics <#FastSAM-in-Ultralytics>`__
-  `Convert the model to OpenVINO Intermediate representation (IR)
   format <#Convert-the-model-to-OpenVINO-Intermediate-representation-(IR)-format>`__
-  `Embedding the converted models into the original
   pipeline <#Embedding-the-converted-models-into-the-original-pipeline>`__

   -  `Select inference device <#Select-inference-device>`__
   -  `Adapt OpenVINO models to the original
      pipeline <#Adapt-OpenVINO-models-to-the-original-pipeline>`__

-  `Optimize the model using NNCF Post-training Quantization
   API <#Optimize-the-model-using-NNCF-Post-training-Quantization-API>`__

   -  `Compare the performance of the Original and Quantized
      Models <#Compare-the-performance-of-the-Original-and-Quantized-Models>`__

-  `Try out the converted pipeline <#Try-out-the-converted-pipeline>`__

Prerequisites
-------------

`back to top ⬆️ <#Table-of-contents:>`__

Install requirements
~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    %pip install -q "ultralytics==8.0.200" onnx --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino-dev>=2023.1.0"
    %pip install -q "nncf>=2.6.0"
    %pip install -q "gradio>=4.13"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    import ipywidgets as widgets
    from pathlib import Path
    
    import openvino as ov
    import torch
    from PIL import Image, ImageDraw
    from ultralytics import FastSAM
    
    import urllib.request
    # Fetch skip_kernel_extension module
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/skip_kernel_extension.py',
        filename='skip_kernel_extension.py'
    )
    # Fetch `notebook_utils` module
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file
    %load_ext skip_kernel_extension

FastSAM in Ultralytics
----------------------

`back to top ⬆️ <#Table-of-contents:>`__

To work with `Fast Segment Anything
Model <https://github.com/CASIA-IVA-Lab/FastSAM>`__ by
``CASIA-IVA-Lab``, we will use the `Ultralytics
package <https://docs.ultralytics.com/>`__. Ultralytics package exposes
the ``FastSAM`` class, simplifying the model instantiation and weights
loading. The code below demonstrates how to initialize a ``FastSAM``
model and generate a segmentation map.

.. code:: ipython3

    model_name = "FastSAM-x"
    model = FastSAM(model_name)
    
    # Run inference on an image
    image_uri = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
    image_uri = download_file(image_uri)
    results = model(image_uri, device="cpu", retina_masks=True, imgsz=1024, conf=0.6, iou=0.9)


.. parsed-literal::

    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/FastSAM-x.pt to 'FastSAM-x.pt'...


.. parsed-literal::

      0%|          | 0.00/138M [00:00<?, ?B/s]

.. parsed-literal::

      0%|          | 176k/138M [00:00<01:20, 1.79MB/s]

.. parsed-literal::

      0%|          | 512k/138M [00:00<00:53, 2.72MB/s]

.. parsed-literal::

      1%|          | 896k/138M [00:00<00:44, 3.22MB/s]

.. parsed-literal::

      1%|          | 1.24M/138M [00:00<00:41, 3.42MB/s]

.. parsed-literal::

      1%|          | 1.62M/138M [00:00<00:40, 3.56MB/s]

.. parsed-literal::

      1%|▏         | 1.98M/138M [00:00<00:39, 3.61MB/s]

.. parsed-literal::

      2%|▏         | 2.36M/138M [00:00<00:38, 3.66MB/s]

.. parsed-literal::

      2%|▏         | 2.73M/138M [00:00<00:38, 3.70MB/s]

.. parsed-literal::

      2%|▏         | 3.11M/138M [00:00<00:38, 3.71MB/s]

.. parsed-literal::

      3%|▎         | 3.49M/138M [00:01<00:37, 3.75MB/s]

.. parsed-literal::

      3%|▎         | 3.86M/138M [00:01<00:37, 3.76MB/s]

.. parsed-literal::

      3%|▎         | 4.23M/138M [00:01<00:37, 3.75MB/s]

.. parsed-literal::

      3%|▎         | 4.61M/138M [00:01<00:37, 3.76MB/s]

.. parsed-literal::

      4%|▎         | 4.98M/138M [00:01<00:37, 3.76MB/s]

.. parsed-literal::

      4%|▍         | 5.36M/138M [00:01<00:37, 3.75MB/s]

.. parsed-literal::

      4%|▍         | 5.73M/138M [00:01<00:36, 3.77MB/s]

.. parsed-literal::

      4%|▍         | 6.10M/138M [00:01<00:36, 3.75MB/s]

.. parsed-literal::

      5%|▍         | 6.48M/138M [00:01<00:36, 3.77MB/s]

.. parsed-literal::

      5%|▍         | 6.85M/138M [00:01<00:36, 3.78MB/s]

.. parsed-literal::

      5%|▌         | 7.23M/138M [00:02<00:36, 3.77MB/s]

.. parsed-literal::

      5%|▌         | 7.60M/138M [00:02<00:36, 3.80MB/s]

.. parsed-literal::

      6%|▌         | 7.97M/138M [00:02<00:36, 3.76MB/s]

.. parsed-literal::

      6%|▌         | 8.34M/138M [00:02<00:36, 3.76MB/s]

.. parsed-literal::

      6%|▋         | 8.72M/138M [00:02<00:36, 3.75MB/s]

.. parsed-literal::

      7%|▋         | 9.09M/138M [00:02<00:35, 3.77MB/s]

.. parsed-literal::

      7%|▋         | 9.47M/138M [00:02<00:35, 3.76MB/s]

.. parsed-literal::

      7%|▋         | 9.84M/138M [00:02<00:35, 3.77MB/s]

.. parsed-literal::

      7%|▋         | 10.2M/138M [00:02<00:35, 3.76MB/s]

.. parsed-literal::

      8%|▊         | 10.6M/138M [00:03<00:35, 3.76MB/s]

.. parsed-literal::

      8%|▊         | 11.0M/138M [00:03<00:35, 3.77MB/s]

.. parsed-literal::

      8%|▊         | 11.3M/138M [00:03<00:35, 3.77MB/s]

.. parsed-literal::

      8%|▊         | 11.7M/138M [00:03<00:35, 3.76MB/s]

.. parsed-literal::

      9%|▊         | 12.1M/138M [00:03<00:35, 3.76MB/s]

.. parsed-literal::

      9%|▉         | 12.5M/138M [00:03<00:35, 3.75MB/s]

.. parsed-literal::

      9%|▉         | 12.8M/138M [00:03<00:34, 3.76MB/s]

.. parsed-literal::

     10%|▉         | 13.2M/138M [00:03<00:34, 3.75MB/s]

.. parsed-literal::

     10%|▉         | 13.6M/138M [00:03<00:34, 3.75MB/s]

.. parsed-literal::

     10%|█         | 14.0M/138M [00:03<00:34, 3.75MB/s]

.. parsed-literal::

     10%|█         | 14.3M/138M [00:04<00:34, 3.75MB/s]

.. parsed-literal::

     11%|█         | 14.7M/138M [00:04<00:34, 3.74MB/s]

.. parsed-literal::

     11%|█         | 15.1M/138M [00:04<00:34, 3.76MB/s]

.. parsed-literal::

     11%|█         | 15.5M/138M [00:04<00:34, 3.75MB/s]

.. parsed-literal::

     11%|█▏        | 15.8M/138M [00:04<00:34, 3.77MB/s]

.. parsed-literal::

     12%|█▏        | 16.2M/138M [00:04<00:34, 3.76MB/s]

.. parsed-literal::

     12%|█▏        | 16.6M/138M [00:04<00:33, 3.76MB/s]

.. parsed-literal::

     12%|█▏        | 17.0M/138M [00:04<00:33, 3.75MB/s]

.. parsed-literal::

     13%|█▎        | 17.3M/138M [00:04<00:33, 3.75MB/s]

.. parsed-literal::

     13%|█▎        | 17.7M/138M [00:04<00:33, 3.75MB/s]

.. parsed-literal::

     13%|█▎        | 18.1M/138M [00:05<00:33, 3.75MB/s]

.. parsed-literal::

     13%|█▎        | 18.5M/138M [00:05<00:33, 3.75MB/s]

.. parsed-literal::

     14%|█▎        | 18.8M/138M [00:05<00:33, 3.76MB/s]

.. parsed-literal::

     14%|█▍        | 19.2M/138M [00:05<00:33, 3.76MB/s]

.. parsed-literal::

     14%|█▍        | 19.6M/138M [00:05<00:33, 3.75MB/s]

.. parsed-literal::

     14%|█▍        | 20.0M/138M [00:05<00:33, 3.75MB/s]

.. parsed-literal::

     15%|█▍        | 20.3M/138M [00:05<00:32, 3.75MB/s]

.. parsed-literal::

     15%|█▍        | 20.7M/138M [00:05<00:32, 3.76MB/s]

.. parsed-literal::

     15%|█▌        | 21.1M/138M [00:05<00:32, 3.75MB/s]

.. parsed-literal::

     16%|█▌        | 21.5M/138M [00:06<00:32, 3.77MB/s]

.. parsed-literal::

     16%|█▌        | 21.8M/138M [00:06<00:32, 3.76MB/s]

.. parsed-literal::

     16%|█▌        | 22.2M/138M [00:06<00:32, 3.77MB/s]

.. parsed-literal::

     16%|█▋        | 22.6M/138M [00:06<00:31, 3.79MB/s]

.. parsed-literal::

     17%|█▋        | 22.9M/138M [00:06<00:32, 3.77MB/s]

.. parsed-literal::

     17%|█▋        | 23.3M/138M [00:06<00:32, 3.76MB/s]

.. parsed-literal::

     17%|█▋        | 23.7M/138M [00:06<00:31, 3.78MB/s]

.. parsed-literal::

     17%|█▋        | 24.1M/138M [00:06<00:31, 3.76MB/s]

.. parsed-literal::

     18%|█▊        | 24.4M/138M [00:06<00:31, 3.75MB/s]

.. parsed-literal::

     18%|█▊        | 24.8M/138M [00:06<00:31, 3.77MB/s]

.. parsed-literal::

     18%|█▊        | 25.2M/138M [00:07<00:31, 3.76MB/s]

.. parsed-literal::

     18%|█▊        | 25.6M/138M [00:07<00:31, 3.78MB/s]

.. parsed-literal::

     19%|█▉        | 25.9M/138M [00:07<00:31, 3.76MB/s]

.. parsed-literal::

     19%|█▉        | 26.3M/138M [00:07<00:31, 3.75MB/s]

.. parsed-literal::

     19%|█▉        | 26.7M/138M [00:07<00:31, 3.75MB/s]

.. parsed-literal::

     20%|█▉        | 27.1M/138M [00:07<00:31, 3.75MB/s]

.. parsed-literal::

     20%|█▉        | 27.4M/138M [00:07<00:30, 3.77MB/s]

.. parsed-literal::

     20%|██        | 27.8M/138M [00:07<00:30, 3.75MB/s]

.. parsed-literal::

     20%|██        | 28.2M/138M [00:07<00:30, 3.75MB/s]

.. parsed-literal::

     21%|██        | 28.6M/138M [00:08<00:30, 3.76MB/s]

.. parsed-literal::

     21%|██        | 28.9M/138M [00:08<00:30, 3.78MB/s]

.. parsed-literal::

     21%|██        | 29.3M/138M [00:08<00:30, 3.76MB/s]

.. parsed-literal::

     21%|██▏       | 29.7M/138M [00:08<00:30, 3.76MB/s]

.. parsed-literal::

     22%|██▏       | 30.1M/138M [00:08<00:30, 3.76MB/s]

.. parsed-literal::

     22%|██▏       | 30.4M/138M [00:08<00:29, 3.77MB/s]

.. parsed-literal::

     22%|██▏       | 30.8M/138M [00:08<00:29, 3.77MB/s]

.. parsed-literal::

     23%|██▎       | 31.2M/138M [00:08<00:29, 3.76MB/s]

.. parsed-literal::

     23%|██▎       | 31.6M/138M [00:08<00:29, 3.76MB/s]

.. parsed-literal::

     23%|██▎       | 31.9M/138M [00:08<00:29, 3.79MB/s]

.. parsed-literal::

     23%|██▎       | 32.3M/138M [00:09<00:29, 3.76MB/s]

.. parsed-literal::

     24%|██▎       | 32.7M/138M [00:09<00:29, 3.78MB/s]

.. parsed-literal::

     24%|██▍       | 33.1M/138M [00:09<00:29, 3.77MB/s]

.. parsed-literal::

     24%|██▍       | 33.4M/138M [00:09<00:29, 3.76MB/s]

.. parsed-literal::

     24%|██▍       | 33.8M/138M [00:09<00:29, 3.77MB/s]

.. parsed-literal::

     25%|██▍       | 34.2M/138M [00:09<00:29, 3.76MB/s]

.. parsed-literal::

     25%|██▍       | 34.6M/138M [00:09<00:28, 3.76MB/s]

.. parsed-literal::

     25%|██▌       | 34.9M/138M [00:09<00:28, 3.77MB/s]

.. parsed-literal::

     26%|██▌       | 35.3M/138M [00:09<00:28, 3.74MB/s]

.. parsed-literal::

     26%|██▌       | 35.7M/138M [00:10<00:28, 3.74MB/s]

.. parsed-literal::

     26%|██▌       | 36.0M/138M [00:10<00:28, 3.74MB/s]

.. parsed-literal::

     26%|██▋       | 36.4M/138M [00:10<00:28, 3.76MB/s]

.. parsed-literal::

     27%|██▋       | 36.8M/138M [00:10<00:28, 3.76MB/s]

.. parsed-literal::

     27%|██▋       | 37.2M/138M [00:10<00:28, 3.77MB/s]

.. parsed-literal::

     27%|██▋       | 37.5M/138M [00:10<00:28, 3.76MB/s]

.. parsed-literal::

     27%|██▋       | 37.9M/138M [00:10<00:27, 3.77MB/s]

.. parsed-literal::

     28%|██▊       | 38.3M/138M [00:10<00:27, 3.78MB/s]

.. parsed-literal::

     28%|██▊       | 38.7M/138M [00:10<00:27, 3.77MB/s]

.. parsed-literal::

     28%|██▊       | 39.0M/138M [00:10<00:27, 3.78MB/s]

.. parsed-literal::

     29%|██▊       | 39.4M/138M [00:11<00:27, 3.77MB/s]

.. parsed-literal::

     29%|██▉       | 39.8M/138M [00:11<00:27, 3.76MB/s]

.. parsed-literal::

     29%|██▉       | 40.2M/138M [00:11<00:27, 3.77MB/s]

.. parsed-literal::

     29%|██▉       | 40.5M/138M [00:11<00:27, 3.76MB/s]

.. parsed-literal::

     30%|██▉       | 40.9M/138M [00:11<00:27, 3.78MB/s]

.. parsed-literal::

     30%|██▉       | 41.3M/138M [00:11<00:26, 3.77MB/s]

.. parsed-literal::

     30%|███       | 41.7M/138M [00:11<00:26, 3.78MB/s]

.. parsed-literal::

     30%|███       | 42.0M/138M [00:11<00:26, 3.77MB/s]

.. parsed-literal::

     31%|███       | 42.4M/138M [00:11<00:26, 3.76MB/s]

.. parsed-literal::

     31%|███       | 42.8M/138M [00:11<00:26, 3.77MB/s]

.. parsed-literal::

     31%|███       | 43.2M/138M [00:12<00:26, 3.76MB/s]

.. parsed-literal::

     31%|███▏      | 43.5M/138M [00:12<00:26, 3.76MB/s]

.. parsed-literal::

     32%|███▏      | 43.9M/138M [00:12<00:26, 3.77MB/s]

.. parsed-literal::

     32%|███▏      | 44.3M/138M [00:12<00:26, 3.76MB/s]

.. parsed-literal::

     32%|███▏      | 44.6M/138M [00:12<00:26, 3.76MB/s]

.. parsed-literal::

     33%|███▎      | 45.0M/138M [00:12<00:25, 3.77MB/s]

.. parsed-literal::

     33%|███▎      | 45.4M/138M [00:12<00:25, 3.77MB/s]

.. parsed-literal::

     33%|███▎      | 45.8M/138M [00:12<00:25, 3.76MB/s]

.. parsed-literal::

     33%|███▎      | 46.1M/138M [00:12<00:25, 3.78MB/s]

.. parsed-literal::

     34%|███▎      | 46.5M/138M [00:13<00:25, 3.77MB/s]

.. parsed-literal::

     34%|███▍      | 46.9M/138M [00:13<00:25, 3.76MB/s]

.. parsed-literal::

     34%|███▍      | 47.3M/138M [00:13<00:25, 3.77MB/s]

.. parsed-literal::

     34%|███▍      | 47.6M/138M [00:13<00:25, 3.78MB/s]

.. parsed-literal::

     35%|███▍      | 48.0M/138M [00:13<00:25, 3.77MB/s]

.. parsed-literal::

     35%|███▍      | 48.4M/138M [00:13<00:24, 3.78MB/s]

.. parsed-literal::

     35%|███▌      | 48.8M/138M [00:13<00:24, 3.77MB/s]

.. parsed-literal::

     36%|███▌      | 49.1M/138M [00:13<00:24, 3.78MB/s]

.. parsed-literal::

     36%|███▌      | 49.5M/138M [00:13<00:24, 3.77MB/s]

.. parsed-literal::

     36%|███▌      | 49.9M/138M [00:13<00:24, 3.76MB/s]

.. parsed-literal::

     36%|███▋      | 50.2M/138M [00:14<00:24, 3.76MB/s]

.. parsed-literal::

     37%|███▋      | 50.6M/138M [00:14<00:24, 3.77MB/s]

.. parsed-literal::

     37%|███▋      | 51.0M/138M [00:14<00:24, 3.78MB/s]

.. parsed-literal::

     37%|███▋      | 51.4M/138M [00:14<00:24, 3.77MB/s]

.. parsed-literal::

     37%|███▋      | 51.8M/138M [00:14<00:24, 3.76MB/s]

.. parsed-literal::

     38%|███▊      | 52.1M/138M [00:14<00:24, 3.73MB/s]

.. parsed-literal::

     38%|███▊      | 52.5M/138M [00:14<00:24, 3.73MB/s]

.. parsed-literal::

     38%|███▊      | 52.9M/138M [00:14<00:23, 3.75MB/s]

.. parsed-literal::

     39%|███▊      | 53.2M/138M [00:14<00:23, 3.75MB/s]

.. parsed-literal::

     39%|███▉      | 53.6M/138M [00:15<00:23, 3.76MB/s]

.. parsed-literal::

     39%|███▉      | 54.0M/138M [00:15<00:23, 3.76MB/s]

.. parsed-literal::

     39%|███▉      | 54.4M/138M [00:15<00:23, 3.77MB/s]

.. parsed-literal::

     40%|███▉      | 54.7M/138M [00:15<00:23, 3.78MB/s]

.. parsed-literal::

     40%|███▉      | 55.1M/138M [00:15<00:23, 3.77MB/s]

.. parsed-literal::

     40%|████      | 55.5M/138M [00:15<00:22, 3.78MB/s]

.. parsed-literal::

     40%|████      | 55.9M/138M [00:15<00:22, 3.77MB/s]

.. parsed-literal::

     41%|████      | 56.2M/138M [00:15<00:22, 3.76MB/s]

.. parsed-literal::

     41%|████      | 56.6M/138M [00:15<00:22, 3.77MB/s]

.. parsed-literal::

     41%|████      | 57.0M/138M [00:15<00:22, 3.76MB/s]

.. parsed-literal::

     41%|████▏     | 57.4M/138M [00:16<00:22, 3.75MB/s]

.. parsed-literal::

     42%|████▏     | 57.7M/138M [00:16<00:22, 3.77MB/s]

.. parsed-literal::

     42%|████▏     | 58.1M/138M [00:16<00:22, 3.77MB/s]

.. parsed-literal::

     42%|████▏     | 58.5M/138M [00:16<00:22, 3.76MB/s]

.. parsed-literal::

     43%|████▎     | 58.8M/138M [00:16<00:21, 3.79MB/s]

.. parsed-literal::

     43%|████▎     | 59.2M/138M [00:16<00:21, 3.77MB/s]

.. parsed-literal::

     43%|████▎     | 59.6M/138M [00:16<00:21, 3.78MB/s]

.. parsed-literal::

     43%|████▎     | 60.0M/138M [00:16<00:21, 3.74MB/s]

.. parsed-literal::

     44%|████▎     | 60.3M/138M [00:16<00:21, 3.76MB/s]

.. parsed-literal::

     44%|████▍     | 60.7M/138M [00:16<00:21, 3.76MB/s]

.. parsed-literal::

     44%|████▍     | 61.1M/138M [00:17<00:21, 3.75MB/s]

.. parsed-literal::

     44%|████▍     | 61.5M/138M [00:17<00:21, 3.75MB/s]

.. parsed-literal::

     45%|████▍     | 61.8M/138M [00:17<00:21, 3.76MB/s]

.. parsed-literal::

     45%|████▌     | 62.2M/138M [00:17<00:21, 3.78MB/s]

.. parsed-literal::

     45%|████▌     | 62.6M/138M [00:17<00:21, 3.77MB/s]

.. parsed-literal::

     46%|████▌     | 63.0M/138M [00:17<00:20, 3.76MB/s]

.. parsed-literal::

     46%|████▌     | 63.3M/138M [00:17<00:20, 3.77MB/s]

.. parsed-literal::

     46%|████▌     | 63.7M/138M [00:17<00:20, 3.76MB/s]

.. parsed-literal::

     46%|████▋     | 64.1M/138M [00:17<00:20, 3.76MB/s]

.. parsed-literal::

     47%|████▋     | 64.5M/138M [00:18<00:20, 3.76MB/s]

.. parsed-literal::

     47%|████▋     | 64.8M/138M [00:18<00:20, 3.77MB/s]

.. parsed-literal::

     47%|████▋     | 65.2M/138M [00:18<00:20, 3.76MB/s]

.. parsed-literal::

     47%|████▋     | 65.6M/138M [00:18<00:20, 3.77MB/s]

.. parsed-literal::

     48%|████▊     | 65.9M/138M [00:18<00:20, 3.76MB/s]

.. parsed-literal::

     48%|████▊     | 66.3M/138M [00:18<00:19, 3.77MB/s]

.. parsed-literal::

     48%|████▊     | 66.7M/138M [00:18<00:19, 3.78MB/s]

.. parsed-literal::

     49%|████▊     | 67.1M/138M [00:18<00:19, 3.75MB/s]

.. parsed-literal::

     49%|████▉     | 67.4M/138M [00:18<00:19, 3.78MB/s]

.. parsed-literal::

     49%|████▉     | 67.8M/138M [00:18<00:19, 3.77MB/s]

.. parsed-literal::

     49%|████▉     | 68.2M/138M [00:19<00:19, 3.76MB/s]

.. parsed-literal::

     50%|████▉     | 68.5M/138M [00:19<00:19, 3.75MB/s]

.. parsed-literal::

     50%|████▉     | 68.9M/138M [00:19<00:19, 3.79MB/s]

.. parsed-literal::

     50%|█████     | 69.3M/138M [00:19<00:19, 3.79MB/s]

.. parsed-literal::

     50%|█████     | 69.7M/138M [00:19<00:19, 3.77MB/s]

.. parsed-literal::

     51%|█████     | 70.0M/138M [00:19<00:18, 3.76MB/s]

.. parsed-literal::

     51%|█████     | 70.4M/138M [00:19<00:18, 3.77MB/s]

.. parsed-literal::

     51%|█████     | 70.8M/138M [00:19<00:18, 3.76MB/s]

.. parsed-literal::

     51%|█████▏    | 71.2M/138M [00:19<00:18, 3.77MB/s]

.. parsed-literal::

     52%|█████▏    | 71.5M/138M [00:19<00:18, 3.76MB/s]

.. parsed-literal::

     52%|█████▏    | 71.9M/138M [00:20<00:18, 3.76MB/s]

.. parsed-literal::

     52%|█████▏    | 72.3M/138M [00:20<00:18, 3.77MB/s]

.. parsed-literal::

     53%|█████▎    | 72.7M/138M [00:20<00:18, 3.76MB/s]

.. parsed-literal::

     53%|█████▎    | 73.0M/138M [00:20<00:18, 3.77MB/s]

.. parsed-literal::

     53%|█████▎    | 73.4M/138M [00:20<00:18, 3.76MB/s]

.. parsed-literal::

     53%|█████▎    | 73.8M/138M [00:20<00:17, 3.79MB/s]

.. parsed-literal::

     54%|█████▎    | 74.2M/138M [00:20<00:17, 3.77MB/s]

.. parsed-literal::

     54%|█████▍    | 74.5M/138M [00:20<00:17, 3.76MB/s]

.. parsed-literal::

     54%|█████▍    | 74.9M/138M [00:20<00:17, 3.76MB/s]

.. parsed-literal::

     54%|█████▍    | 75.3M/138M [00:21<00:17, 3.76MB/s]

.. parsed-literal::

     55%|█████▍    | 75.7M/138M [00:21<00:17, 3.77MB/s]

.. parsed-literal::

     55%|█████▌    | 76.0M/138M [00:21<00:17, 3.76MB/s]

.. parsed-literal::

     55%|█████▌    | 76.4M/138M [00:21<00:17, 3.75MB/s]

.. parsed-literal::

     56%|█████▌    | 76.8M/138M [00:21<00:17, 3.75MB/s]

.. parsed-literal::

     56%|█████▌    | 77.2M/138M [00:21<00:17, 3.75MB/s]

.. parsed-literal::

     56%|█████▌    | 77.5M/138M [00:21<00:16, 3.77MB/s]

.. parsed-literal::

     56%|█████▋    | 77.9M/138M [00:21<00:16, 3.76MB/s]

.. parsed-literal::

     57%|█████▋    | 78.3M/138M [00:21<00:16, 3.77MB/s]

.. parsed-literal::

     57%|█████▋    | 78.6M/138M [00:21<00:16, 3.79MB/s]

.. parsed-literal::

     57%|█████▋    | 79.0M/138M [00:22<00:16, 3.76MB/s]

.. parsed-literal::

     57%|█████▋    | 79.4M/138M [00:22<00:16, 3.73MB/s]

.. parsed-literal::

     58%|█████▊    | 79.8M/138M [00:22<00:16, 3.73MB/s]

.. parsed-literal::

     58%|█████▊    | 80.1M/138M [00:22<00:16, 3.74MB/s]

.. parsed-literal::

     58%|█████▊    | 80.5M/138M [00:22<00:16, 3.76MB/s]

.. parsed-literal::

     59%|█████▊    | 80.9M/138M [00:22<00:16, 3.75MB/s]

.. parsed-literal::

     59%|█████▉    | 81.3M/138M [00:22<00:15, 3.77MB/s]

.. parsed-literal::

     59%|█████▉    | 81.6M/138M [00:22<00:15, 3.76MB/s]

.. parsed-literal::

     59%|█████▉    | 82.0M/138M [00:22<00:15, 3.77MB/s]

.. parsed-literal::

     60%|█████▉    | 82.4M/138M [00:23<00:15, 3.78MB/s]

.. parsed-literal::

     60%|█████▉    | 82.8M/138M [00:23<00:15, 3.76MB/s]

.. parsed-literal::

     60%|██████    | 83.1M/138M [00:23<00:15, 3.75MB/s]

.. parsed-literal::

     60%|██████    | 83.5M/138M [00:23<00:15, 3.76MB/s]

.. parsed-literal::

     61%|██████    | 83.9M/138M [00:23<00:15, 3.76MB/s]

.. parsed-literal::

     61%|██████    | 84.3M/138M [00:23<00:15, 3.75MB/s]

.. parsed-literal::

     61%|██████    | 84.6M/138M [00:23<00:14, 3.76MB/s]

.. parsed-literal::

     61%|██████▏   | 85.0M/138M [00:23<00:14, 3.78MB/s]

.. parsed-literal::

     62%|██████▏   | 85.4M/138M [00:23<00:14, 3.76MB/s]

.. parsed-literal::

     62%|██████▏   | 85.8M/138M [00:23<00:14, 3.77MB/s]

.. parsed-literal::

     62%|██████▏   | 86.1M/138M [00:24<00:14, 3.78MB/s]

.. parsed-literal::

     63%|██████▎   | 86.5M/138M [00:24<00:14, 3.77MB/s]

.. parsed-literal::

     63%|██████▎   | 86.9M/138M [00:24<00:14, 3.78MB/s]

.. parsed-literal::

     63%|██████▎   | 87.3M/138M [00:24<00:14, 3.76MB/s]

.. parsed-literal::

     63%|██████▎   | 87.6M/138M [00:24<00:14, 3.71MB/s]

.. parsed-literal::

     64%|██████▎   | 88.0M/138M [00:24<00:14, 3.72MB/s]

.. parsed-literal::

     64%|██████▍   | 88.4M/138M [00:24<00:13, 3.74MB/s]

.. parsed-literal::

     64%|██████▍   | 88.7M/138M [00:24<00:13, 3.74MB/s]

.. parsed-literal::

     64%|██████▍   | 89.1M/138M [00:24<00:13, 3.74MB/s]

.. parsed-literal::

     65%|██████▍   | 89.5M/138M [00:25<00:13, 3.74MB/s]

.. parsed-literal::

     65%|██████▌   | 89.9M/138M [00:25<00:13, 3.74MB/s]

.. parsed-literal::

     65%|██████▌   | 90.2M/138M [00:25<00:13, 3.76MB/s]

.. parsed-literal::

     66%|██████▌   | 90.6M/138M [00:25<00:13, 3.75MB/s]

.. parsed-literal::

     66%|██████▌   | 91.0M/138M [00:25<00:13, 3.77MB/s]

.. parsed-literal::

     66%|██████▌   | 91.4M/138M [00:25<00:12, 3.80MB/s]

.. parsed-literal::

     66%|██████▋   | 91.7M/138M [00:25<00:12, 3.78MB/s]

.. parsed-literal::

     67%|██████▋   | 92.1M/138M [00:25<00:12, 3.77MB/s]

.. parsed-literal::

     67%|██████▋   | 92.5M/138M [00:25<00:12, 3.78MB/s]

.. parsed-literal::

     67%|██████▋   | 92.9M/138M [00:25<00:12, 3.77MB/s]

.. parsed-literal::

     67%|██████▋   | 93.2M/138M [00:26<00:12, 3.75MB/s]

.. parsed-literal::

     68%|██████▊   | 93.6M/138M [00:26<00:12, 3.75MB/s]

.. parsed-literal::

     68%|██████▊   | 94.0M/138M [00:26<00:12, 3.79MB/s]

.. parsed-literal::

     68%|██████▊   | 94.4M/138M [00:26<00:12, 3.77MB/s]

.. parsed-literal::

     69%|██████▊   | 94.7M/138M [00:26<00:12, 3.76MB/s]

.. parsed-literal::

     69%|██████▉   | 95.1M/138M [00:26<00:11, 3.78MB/s]

.. parsed-literal::

     69%|██████▉   | 95.5M/138M [00:26<00:11, 3.76MB/s]

.. parsed-literal::

     69%|██████▉   | 95.9M/138M [00:26<00:11, 3.75MB/s]

.. parsed-literal::

     70%|██████▉   | 96.2M/138M [00:26<00:11, 3.76MB/s]

.. parsed-literal::

     70%|██████▉   | 96.6M/138M [00:26<00:11, 3.77MB/s]

.. parsed-literal::

     70%|███████   | 97.0M/138M [00:27<00:11, 3.76MB/s]

.. parsed-literal::

     70%|███████   | 97.3M/138M [00:27<00:11, 3.77MB/s]

.. parsed-literal::

     71%|███████   | 97.7M/138M [00:27<00:11, 3.78MB/s]

.. parsed-literal::

     71%|███████   | 98.1M/138M [00:27<00:11, 3.79MB/s]

.. parsed-literal::

     71%|███████   | 98.5M/138M [00:27<00:11, 3.77MB/s]

.. parsed-literal::

     72%|███████▏  | 98.8M/138M [00:27<00:10, 3.76MB/s]

.. parsed-literal::

     72%|███████▏  | 99.2M/138M [00:27<00:10, 3.77MB/s]

.. parsed-literal::

     72%|███████▏  | 99.6M/138M [00:27<00:10, 3.77MB/s]

.. parsed-literal::

     72%|███████▏  | 100M/138M [00:27<00:10, 3.76MB/s] 

.. parsed-literal::

     73%|███████▎  | 100M/138M [00:28<00:10, 3.75MB/s]

.. parsed-literal::

     73%|███████▎  | 101M/138M [00:28<00:10, 3.75MB/s]

.. parsed-literal::

     73%|███████▎  | 101M/138M [00:28<00:10, 3.76MB/s]

.. parsed-literal::

     73%|███████▎  | 101M/138M [00:28<00:10, 3.75MB/s]

.. parsed-literal::

     74%|███████▎  | 102M/138M [00:28<00:10, 3.75MB/s]

.. parsed-literal::

     74%|███████▍  | 102M/138M [00:28<00:10, 3.76MB/s]

.. parsed-literal::

     74%|███████▍  | 103M/138M [00:28<00:09, 3.78MB/s]

.. parsed-literal::

     74%|███████▍  | 103M/138M [00:28<00:09, 3.78MB/s]

.. parsed-literal::

     75%|███████▍  | 103M/138M [00:28<00:09, 3.77MB/s]

.. parsed-literal::

     75%|███████▌  | 104M/138M [00:28<00:09, 3.75MB/s]

.. parsed-literal::

     75%|███████▌  | 104M/138M [00:29<00:09, 3.75MB/s]

.. parsed-literal::

     76%|███████▌  | 104M/138M [00:29<00:09, 3.77MB/s]

.. parsed-literal::

     76%|███████▌  | 105M/138M [00:29<00:09, 3.76MB/s]

.. parsed-literal::

     76%|███████▌  | 105M/138M [00:29<00:09, 3.75MB/s]

.. parsed-literal::

     76%|███████▋  | 106M/138M [00:29<00:09, 3.76MB/s]

.. parsed-literal::

     77%|███████▋  | 106M/138M [00:29<00:09, 3.76MB/s]

.. parsed-literal::

     77%|███████▋  | 106M/138M [00:29<00:08, 3.77MB/s]

.. parsed-literal::

     77%|███████▋  | 107M/138M [00:29<00:08, 3.75MB/s]

.. parsed-literal::

     77%|███████▋  | 107M/138M [00:29<00:08, 3.77MB/s]

.. parsed-literal::

     78%|███████▊  | 107M/138M [00:30<00:08, 3.78MB/s]

.. parsed-literal::

     78%|███████▊  | 108M/138M [00:30<00:08, 3.78MB/s]

.. parsed-literal::

     78%|███████▊  | 108M/138M [00:30<00:08, 3.77MB/s]

.. parsed-literal::

     79%|███████▊  | 109M/138M [00:30<00:08, 3.78MB/s]

.. parsed-literal::

     79%|███████▉  | 109M/138M [00:30<00:08, 3.78MB/s]

.. parsed-literal::

     79%|███████▉  | 109M/138M [00:30<00:08, 3.75MB/s]

.. parsed-literal::

     79%|███████▉  | 110M/138M [00:30<00:07, 3.75MB/s]

.. parsed-literal::

     80%|███████▉  | 110M/138M [00:30<00:07, 3.76MB/s]

.. parsed-literal::

     80%|███████▉  | 110M/138M [00:30<00:07, 3.76MB/s]

.. parsed-literal::

     80%|████████  | 111M/138M [00:30<00:07, 3.75MB/s]

.. parsed-literal::

     80%|████████  | 111M/138M [00:31<00:07, 3.75MB/s]

.. parsed-literal::

     81%|████████  | 112M/138M [00:31<00:07, 3.75MB/s]

.. parsed-literal::

     81%|████████  | 112M/138M [00:31<00:07, 3.75MB/s]

.. parsed-literal::

     81%|████████▏ | 112M/138M [00:31<00:07, 3.76MB/s]

.. parsed-literal::

     82%|████████▏ | 113M/138M [00:31<00:07, 3.77MB/s]

.. parsed-literal::

     82%|████████▏ | 113M/138M [00:31<00:07, 3.76MB/s]

.. parsed-literal::

     82%|████████▏ | 113M/138M [00:31<00:06, 3.76MB/s]

.. parsed-literal::

     82%|████████▏ | 114M/138M [00:31<00:06, 3.77MB/s]

.. parsed-literal::

     83%|████████▎ | 114M/138M [00:31<00:06, 3.77MB/s]

.. parsed-literal::

     83%|████████▎ | 115M/138M [00:31<00:06, 3.76MB/s]

.. parsed-literal::

     83%|████████▎ | 115M/138M [00:32<00:06, 3.75MB/s]

.. parsed-literal::

     83%|████████▎ | 115M/138M [00:32<00:06, 3.74MB/s]

.. parsed-literal::

     84%|████████▎ | 116M/138M [00:32<00:06, 3.76MB/s]

.. parsed-literal::

     84%|████████▍ | 116M/138M [00:32<00:06, 3.76MB/s]

.. parsed-literal::

     84%|████████▍ | 116M/138M [00:32<00:06, 3.77MB/s]

.. parsed-literal::

     85%|████████▍ | 117M/138M [00:32<00:05, 3.76MB/s]

.. parsed-literal::

     85%|████████▍ | 117M/138M [00:32<00:05, 3.77MB/s]

.. parsed-literal::

     85%|████████▌ | 118M/138M [00:32<00:05, 3.78MB/s]

.. parsed-literal::

     85%|████████▌ | 118M/138M [00:32<00:05, 3.77MB/s]

.. parsed-literal::

     86%|████████▌ | 118M/138M [00:33<00:05, 3.77MB/s]

.. parsed-literal::

     86%|████████▌ | 119M/138M [00:33<00:05, 3.74MB/s]

.. parsed-literal::

     86%|████████▌ | 119M/138M [00:33<00:05, 3.76MB/s]

.. parsed-literal::

     86%|████████▋ | 119M/138M [00:33<00:05, 3.77MB/s]

.. parsed-literal::

     87%|████████▋ | 120M/138M [00:33<00:05, 3.76MB/s]

.. parsed-literal::

     87%|████████▋ | 120M/138M [00:33<00:05, 3.77MB/s]

.. parsed-literal::

     87%|████████▋ | 121M/138M [00:33<00:04, 3.78MB/s]

.. parsed-literal::

     87%|████████▋ | 121M/138M [00:33<00:04, 3.78MB/s]

.. parsed-literal::

     88%|████████▊ | 121M/138M [00:33<00:04, 3.77MB/s]

.. parsed-literal::

     88%|████████▊ | 122M/138M [00:33<00:04, 3.76MB/s]

.. parsed-literal::

     88%|████████▊ | 122M/138M [00:34<00:04, 3.77MB/s]

.. parsed-literal::

     89%|████████▊ | 122M/138M [00:34<00:04, 3.78MB/s]

.. parsed-literal::

     89%|████████▉ | 123M/138M [00:34<00:04, 3.77MB/s]

.. parsed-literal::

     89%|████████▉ | 123M/138M [00:34<00:04, 3.78MB/s]

.. parsed-literal::

     89%|████████▉ | 124M/138M [00:34<00:04, 3.77MB/s]

.. parsed-literal::

     90%|████████▉ | 124M/138M [00:34<00:03, 3.77MB/s]

.. parsed-literal::

     90%|████████▉ | 124M/138M [00:34<00:03, 3.76MB/s]

.. parsed-literal::

     90%|█████████ | 125M/138M [00:34<00:03, 3.77MB/s]

.. parsed-literal::

     90%|█████████ | 125M/138M [00:34<00:03, 3.76MB/s]

.. parsed-literal::

     91%|█████████ | 125M/138M [00:35<00:03, 3.77MB/s]

.. parsed-literal::

     91%|█████████ | 126M/138M [00:35<00:03, 3.76MB/s]

.. parsed-literal::

     91%|█████████▏| 126M/138M [00:35<00:03, 3.77MB/s]

.. parsed-literal::

     92%|█████████▏| 127M/138M [00:35<00:03, 3.77MB/s]

.. parsed-literal::

     92%|█████████▏| 127M/138M [00:35<00:03, 3.78MB/s]

.. parsed-literal::

     92%|█████████▏| 127M/138M [00:35<00:03, 3.78MB/s]

.. parsed-literal::

     92%|█████████▏| 128M/138M [00:35<00:02, 3.80MB/s]

.. parsed-literal::

     93%|█████████▎| 128M/138M [00:35<00:02, 3.78MB/s]

.. parsed-literal::

     93%|█████████▎| 128M/138M [00:35<00:02, 3.77MB/s]

.. parsed-literal::

     93%|█████████▎| 129M/138M [00:35<00:02, 3.78MB/s]

.. parsed-literal::

     93%|█████████▎| 129M/138M [00:36<00:02, 3.77MB/s]

.. parsed-literal::

     94%|█████████▎| 130M/138M [00:36<00:02, 3.74MB/s]

.. parsed-literal::

     94%|█████████▍| 130M/138M [00:36<00:02, 3.74MB/s]

.. parsed-literal::

     94%|█████████▍| 130M/138M [00:36<00:02, 3.74MB/s]

.. parsed-literal::

     95%|█████████▍| 131M/138M [00:36<00:02, 3.76MB/s]

.. parsed-literal::

     95%|█████████▍| 131M/138M [00:36<00:02, 3.77MB/s]

.. parsed-literal::

     95%|█████████▌| 131M/138M [00:36<00:01, 3.76MB/s]

.. parsed-literal::

     95%|█████████▌| 132M/138M [00:36<00:01, 3.76MB/s]

.. parsed-literal::

     96%|█████████▌| 132M/138M [00:36<00:01, 3.77MB/s]

.. parsed-literal::

     96%|█████████▌| 133M/138M [00:36<00:01, 3.76MB/s]

.. parsed-literal::

     96%|█████████▌| 133M/138M [00:37<00:01, 3.78MB/s]

.. parsed-literal::

     96%|█████████▋| 133M/138M [00:37<00:01, 3.77MB/s]

.. parsed-literal::

     97%|█████████▋| 134M/138M [00:37<00:01, 3.76MB/s]

.. parsed-literal::

     97%|█████████▋| 134M/138M [00:37<00:01, 3.77MB/s]

.. parsed-literal::

     97%|█████████▋| 134M/138M [00:37<00:01, 3.78MB/s]

.. parsed-literal::

     98%|█████████▊| 135M/138M [00:37<00:00, 3.77MB/s]

.. parsed-literal::

     98%|█████████▊| 135M/138M [00:37<00:00, 3.78MB/s]

.. parsed-literal::

     98%|█████████▊| 136M/138M [00:37<00:00, 3.78MB/s]

.. parsed-literal::

     98%|█████████▊| 136M/138M [00:37<00:00, 3.80MB/s]

.. parsed-literal::

     99%|█████████▊| 136M/138M [00:38<00:00, 3.76MB/s]

.. parsed-literal::

     99%|█████████▉| 137M/138M [00:38<00:00, 3.75MB/s]

.. parsed-literal::

     99%|█████████▉| 137M/138M [00:38<00:00, 3.77MB/s]

.. parsed-literal::

     99%|█████████▉| 137M/138M [00:38<00:00, 3.76MB/s]

.. parsed-literal::

    100%|█████████▉| 138M/138M [00:38<00:00, 3.77MB/s]

.. parsed-literal::

    100%|█████████▉| 138M/138M [00:38<00:00, 3.76MB/s]

.. parsed-literal::

    100%|██████████| 138M/138M [00:38<00:00, 3.76MB/s]

.. parsed-literal::

    



.. parsed-literal::

    coco_bike.jpg:   0%|          | 0.00/182k [00:00<?, ?B/s]


.. parsed-literal::

    


.. parsed-literal::

    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/notebooks/261-fast-segment-anything/coco_bike.jpg: 768x1024 37 objects, 641.7ms


.. parsed-literal::

    Speed: 3.8ms preprocess, 641.7ms inference, 27.5ms postprocess per image at shape (1, 3, 768, 1024)


The model returns segmentation maps for all the objects on the image.
Observe the results below.

.. code:: ipython3

    Image.fromarray(results[0].plot()[..., ::-1])




.. image:: 261-fast-segment-anything-with-output_files/261-fast-segment-anything-with-output_9_0.png



Convert the model to OpenVINO Intermediate representation (IR) format
---------------------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The Ultralytics Model export API enables conversion of PyTorch models to
OpenVINO IR format. Under the hood it utilizes the
``openvino.convert_model`` method to acquire OpenVINO IR versions of the
models. The method requires a model object and example input for model
tracing. The FastSAM model itself is based on YOLOv8 model.

.. code:: ipython3

    # instance segmentation model
    ov_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    if not ov_model_path.exists():
        ov_model = model.export(format="openvino", dynamic=True, half=False)



.. parsed-literal::

    Ultralytics YOLOv8.0.200 🚀 Python-3.8.10 torch-2.1.2+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)


.. parsed-literal::

    
    PyTorch: starting from 'FastSAM-x.pt' with input shape (1, 3, 1024, 1024) BCHW and output shape(s) ((1, 37, 21504), (1, 32, 256, 256)) (138.2 MB)


.. parsed-literal::

    
    ONNX: starting export with onnx 1.15.0 opset 17...


.. parsed-literal::

    ONNX: export success ✅ 3.9s, saved as 'FastSAM-x.onnx' (275.5 MB)


.. parsed-literal::

    
    OpenVINO: starting export with openvino 2023.3.0-13775-ceeafaf64f3-releases/2023/3...


.. parsed-literal::

    OpenVINO: export success ✅ 1.1s, saved as 'FastSAM-x_openvino_model/' (275.9 MB)


.. parsed-literal::

    
    Export complete (8.0s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/notebooks/261-fast-segment-anything
    Predict:         yolo predict task=segment model=FastSAM-x_openvino_model imgsz=1024  
    Validate:        yolo val task=segment model=FastSAM-x_openvino_model imgsz=1024 data=ultralytics/datasets/sa.yaml  
    Visualize:       https://netron.app


Embedding the converted models into the original pipeline
---------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

OpenVINO™ Runtime Python API is used to compile the model in OpenVINO IR
format. The
`Core <https://docs.openvino.ai/2022.3/api/ie_python_api/_autosummary/openvino.runtime.Core.html>`__
class provides access to the OpenVINO Runtime API. The ``core`` object,
which is an instance of the ``Core`` class represents the API and it is
used to compile the model.

.. code:: ipython3

    core = ov.Core()

Select inference device
^^^^^^^^^^^^^^^^^^^^^^^

`back to top ⬆️ <#Table-of-contents:>`__

Select device that will be used to do models inference using OpenVINO
from the dropdown list:

.. code:: ipython3

    DEVICE = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )
    
    DEVICE




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Adapt OpenVINO models to the original pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Here we create wrapper classes for the OpenVINO model that we want to
embed in the original inference pipeline. Here are some of the things to
consider when adapting an OV model: - Make sure that parameters passed
by the original pipeline are forwarded to the compiled OV model
properly; sometimes the OV model uses only a portion of the input
arguments and some are ignored, sometimes you need to convert the
argument to another data type or unwrap some data structures such as
tuples or dictionaries. - Guarantee that the wrapper class returns
results to the pipeline in an expected format. In the example below you
can see how we pack OV model outputs into a tuple of ``torch`` tensors.
- Pay attention to the model method used in the original pipeline for
calling the model - it may be not the ``forward`` method! In this
example, the model is a part of a ``predictor`` object and called as and
object, so we need to redefine the magic ``__call__`` method.

.. code:: ipython3

    class OVWrapper:
        def __init__(self, ov_model, device="CPU", stride=32) -> None:
            self.model = core.compile_model(ov_model, device_name=device)
    
            self.stride = stride
            self.pt = True
            self.fp16 = False
            self.names = {0: "object"}
    
        def __call__(self, im, **_):
            result = self.model(im)
            return torch.from_numpy(result[0]), torch.from_numpy(result[1])

Now we initialize the wrapper objects and load them to the FastSAM
pipeline.

.. code:: ipython3

    wrapped_model = OVWrapper(ov_model_path, device=DEVICE.value, stride=model.predictor.model.stride)
    model.predictor.model = wrapped_model
    
    ov_results = model(image_uri, device=DEVICE.value, retina_masks=True, imgsz=640, conf=0.6, iou=0.9)


.. parsed-literal::

    


.. parsed-literal::

    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/notebooks/261-fast-segment-anything/coco_bike.jpg: 480x640 33 objects, 336.9ms


.. parsed-literal::

    Speed: 3.5ms preprocess, 336.9ms inference, 18.5ms postprocess per image at shape (1, 3, 480, 640)


One can observe the converted model outputs in the next cell, they is
the same as of the original model.

.. code:: ipython3

    Image.fromarray(ov_results[0].plot()[..., ::-1])




.. image:: 261-fast-segment-anything-with-output_files/261-fast-segment-anything-with-output_21_0.png



Optimize the model using NNCF Post-training Quantization API
------------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
FastSAM.

The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` to obtain a quantized model.
3. Save the INT8 model using ``openvino.save_model()`` function.

.. code:: ipython3

    do_quantize = widgets.Checkbox(
        value=True,
        description='Quantization',
        disabled=False,
    )
    
    do_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



The ``nncf.quantize`` function provides an interface for model
quantization. It requires an instance of the OpenVINO Model and
quantization dataset. Optionally, some additional parameters for the
configuration quantization process (number of samples for quantization,
preset, ignored scope, etc.) can be provided. YOLOv8 model backing
FastSAM contains non-ReLU activation functions, which require asymmetric
quantization of activations. To achieve a better result, we will use a
``mixed`` quantization preset. It provides symmetric quantization of
weights and asymmetric quantization of activations. For more accurate
results, we should keep the operation in the postprocessing subgraph in
floating point precision, using the ``ignored_scope`` parameter.

The quantization algorithm is based on `The YOLOv8 quantization
example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/post_training_quantization/openvino/yolov8>`__
in the NNCF repo, refer there for more details. Moreover, you can check
out other quantization tutorials in the `OV notebooks
repo <../230-yolov8-optimization/>`__.

   **Note**: Model post-training quantization is time-consuming process.
   Be patient, it can take several minutes depending on your hardware.

.. code:: ipython3

    %%skip not $do_quantize.value
    
    import pickle
    from contextlib import contextmanager
    from zipfile import ZipFile
    
    import cv2
    from tqdm.autonotebook import tqdm
    
    import nncf
    
    
    COLLECT_CALIBRATION_DATA = False
    calibration_data = []
    
    @contextmanager
    def calibration_data_collection():
        global COLLECT_CALIBRATION_DATA
        try:
            COLLECT_CALIBRATION_DATA = True
            yield
        finally:
            COLLECT_CALIBRATION_DATA = False
    
    
    class NNCFWrapper:
        def __init__(self, ov_model, stride=32) -> None:
            self.model = core.read_model(ov_model)
            self.compiled_model = core.compile_model(self.model, device_name="CPU")
    
            self.stride = stride
            self.pt = True
            self.fp16 = False
            self.names = {0: "object"}
    
        def __call__(self, im, **_):
            if COLLECT_CALIBRATION_DATA:
                calibration_data.append(im)
    
            result = self.compiled_model(im)
            return torch.from_numpy(result[0]), torch.from_numpy(result[1])
    
    # Fetch data from the web and descibe a dataloader
    DATA_URL = "https://ultralytics.com/assets/coco128.zip"
    OUT_DIR = Path('.')
    
    download_file(DATA_URL, directory=OUT_DIR, show_progress=True)
    
    if not (OUT_DIR / "coco128/images/train2017").exists():
        with ZipFile('coco128.zip', "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
    
    class COCOLoader(torch.utils.data.Dataset):
        def __init__(self, images_path):
            self.images = list(Path(images_path).iterdir())
    
        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.read_image(image_path) for image_path in self.images[index]]
            return self.read_image(self.images[index])
    
        def read_image(self, image_path):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    
        def __len__(self):
            return len(self.images)
    
    
    def collect_calibration_data_for_decoder(model, calibration_dataset_size: int,
                                             calibration_cache_path: Path):
        global calibration_data
    
    
        if not calibration_cache_path.exists():
            coco_dataset = COCOLoader(OUT_DIR / 'coco128/images/train2017')
            with calibration_data_collection():
                for image in tqdm(coco_dataset[:calibration_dataset_size], desc="Collecting calibration data"):
                    model(image, retina_masks=True, imgsz=640, conf=0.6, iou=0.9, verbose=False)
            calibration_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(calibration_cache_path, "wb") as f:
                pickle.dump(calibration_data, f)
        else:
            with open(calibration_cache_path, "rb") as f:
                calibration_data = pickle.load(f)
    
        return calibration_data
    
    
    def quantize(model, save_model_path: Path, calibration_cache_path: Path,
                 calibration_dataset_size: int, preset: nncf.QuantizationPreset):
        calibration_data = collect_calibration_data_for_decoder(
            model, calibration_dataset_size, calibration_cache_path)
        quantized_ov_decoder = nncf.quantize(
            model.predictor.model.model,
            calibration_dataset=nncf.Dataset(calibration_data),
            preset=preset,
            subset_size=len(calibration_data),
            fast_bias_correction=True,
            ignored_scope=nncf.IgnoredScope(
                types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
                names=[
                    "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
                    "/model.22/Add",
                    "/model.22/Add_1",
                    "/model.22/Add_2",
                    "/model.22/Add_3",
                    "/model.22/Add_4",
                    "/model.22/Add_5",
                    "/model.22/Add_6",
                    "/model.22/Add_7",
                    "/model.22/Add_8",
                    "/model.22/Add_9",
                    "/model.22/Add_10",
                ],
            )
        )
        ov.save_model(quantized_ov_decoder, save_model_path)
    
    wrapped_model = NNCFWrapper(ov_model_path, stride=model.predictor.model.stride)
    model.predictor.model = wrapped_model
    
    calibration_dataset_size = 128
    quantized_model_path = Path(f"{model_name}_quantized") / "FastSAM-x.xml"
    calibration_cache_path = Path(f"calibration_data/coco{calibration_dataset_size}.pkl")
    if not quantized_model_path.exists():
        quantize(model, quantized_model_path, calibration_cache_path,
                 calibration_dataset_size=calibration_dataset_size,
                 preset=nncf.QuantizationPreset.MIXED)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino



.. parsed-literal::

    coco128.zip:   0%|          | 0.00/6.66M [00:00<?, ?B/s]



.. parsed-literal::

    Collecting calibration data:   0%|          | 0/128 [00:00<?, ?it/s]


.. parsed-literal::

    INFO:nncf:12 ignored nodes were found by name in the NNCFGraph


.. parsed-literal::

    INFO:nncf:13 ignored nodes were found by types in the NNCFGraph


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 204 /model.22/Sigmoid


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 246 /model.22/dfl/conv/Conv


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 275 /model.22/Sub


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 276 /model.22/Add_10


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 299 /model.22/Sub_1


.. parsed-literal::

    INFO:nncf:Not adding activation input quantizer for operation: 341 /model.22/Mul_5



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-598/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:84: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Compare the performance of the Original and Quantized Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

Finally, we iterate both the OV model and the quantized model over the
calibration dataset to measure the performance.

.. code:: ipython3

    %%skip not $do_quantize.value
    
    import datetime
    
    coco_dataset = COCOLoader(OUT_DIR / 'coco128/images/train2017')
    calibration_dataset_size = 128
    
    wrapped_model = OVWrapper(ov_model_path, device=DEVICE.value, stride=model.predictor.model.stride)
    model.predictor.model = wrapped_model
    
    start_time = datetime.datetime.now()
    for image in tqdm(coco_dataset, desc="Measuring inference time"):
        model(image, retina_masks=True, imgsz=640, conf=0.6, iou=0.9, verbose=False)
    duration_base = (datetime.datetime.now() - start_time).seconds
    print("Segmented in", duration_base, "seconds.")
    print("Resulting in", round(calibration_dataset_size / duration_base, 2), "fps")



.. parsed-literal::

    Measuring inference time:   0%|          | 0/128 [00:00<?, ?it/s]


.. parsed-literal::

    Segmented in 21 seconds.
    Resulting in 6.1 fps


.. code:: ipython3

    %%skip not $do_quantize.value
    
    quantized_wrapped_model = OVWrapper(quantized_model_path, device=DEVICE.value, stride=model.predictor.model.stride)
    model.predictor.model = quantized_wrapped_model
    
    start_time = datetime.datetime.now()
    for image in tqdm(coco_dataset, desc="Measuring inference time"):
        model(image, retina_masks=True, imgsz=640, conf=0.6, iou=0.9, verbose=False)
    duration_quantized = (datetime.datetime.now() - start_time).seconds
    print("Segmented in", duration_quantized, "seconds")
    print("Resulting in", round(calibration_dataset_size / duration_quantized, 2), "fps")
    print("That is", round(duration_base / duration_quantized, 2), "times faster!")



.. parsed-literal::

    Measuring inference time:   0%|          | 0/128 [00:00<?, ?it/s]


.. parsed-literal::

    Segmented in 11 seconds
    Resulting in 11.64 fps
    That is 1.91 times faster!


Try out the converted pipeline
------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

The demo app below is created using `Gradio
package <https://www.gradio.app/docs/interface>`__.

The app allows you to alter the model output interactively. Using the
Pixel selector type switch you can place foreground/background points or
bounding boxes on input image.

.. code:: ipython3

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fast_process(
        annotations,
        image,
        scale,
        better_quality=False,
        mask_random_color=True,
        bbox=None,
        use_retina=True,
        with_contours=True,
    ):
    
        original_h = image.height
        original_w = image.width
    
        if better_quality:
            for i, mask in enumerate(annotations):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
    
        inner_mask = fast_show_mask(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            retinamask=use_retina,
            target_height=original_h,
            target_width=original_w,
        )
    
        if with_contours:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                annotation = mask.astype(np.uint8)
                if not use_retina:
                    annotation = cv2.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_all.append(contour)
            cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
    
        image = image.convert("RGBA")
        overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), "RGBA")
        image.paste(overlay_inner, (0, 0), overlay_inner)
    
        if with_contours:
            overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), "RGBA")
            image.paste(overlay_contour, (0, 0), overlay_contour)
    
        return image
    
    
    # CPU post process
    def fast_show_mask(
        annotation,
        ax,
        random_color=False,
        bbox=None,
        retinamask=True,
        target_height=960,
        target_width=960,
    ):
        mask_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        # 
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)[::1]
        annotation = annotation[sorted_indices]
    
        index = (annotation != 0).argmax(axis=0)
        if random_color:
            color = np.random.random((mask_sum, 1, 1, 3))
        else:
            color = np.ones((mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
        transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual
    
        mask = np.zeros((height, weight, 4))
    
        h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing="ij")
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    
        mask[h_indices, w_indices, :] = mask_image[indices]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1))
    
        if not retinamask:
            mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
        return mask

.. code:: ipython3

    import gradio as gr
    
    examples = [[image_uri], ["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/empty_road_mapillary.jpg"],
                ["https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"]]
    
    object_points = []
    background_points = []
    bbox_points = []
    last_image = examples[0][0]

This is the main callback function that is called to segment an image
based on user input.

.. code:: ipython3

    def segment(
            image,
            model_type,
            input_size=1024, 
            iou_threshold=0.75,
            conf_threshold=0.4,
            better_quality=True,
            with_contours=True,
            use_retina=True,
            mask_random_color=True,
    ):
        if do_quantize.value and model_type == 'Quantized model':
            model.predictor.model = quantized_wrapped_model
        else:
            model.predictor.model = wrapped_model
        
        input_size = int(input_size)
        w, h = image.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))
    
        results = model(image,
                        device=DEVICE.value,
                        retina_masks=use_retina,
                        iou=iou_threshold,
                        conf=conf_threshold,
                        imgsz=input_size,)
    
        masks = results[0].masks.data
        # Calculate annotations
        if not (object_points or bbox_points):
            annotations = masks.cpu().numpy()
        else:
            annotations = []
    
        if object_points:
            all_points = object_points + background_points
            labels = [1] * len(object_points) + [0] * len(background_points)
            scaled_points = [[int(x * scale) for x in point] for point in all_points]
            h, w = masks[0].shape[:2]
            assert max(h, w) == input_size
            onemask = np.zeros((h, w))
            for mask in sorted(masks, key=lambda x: x.sum(), reverse=True):
                mask_np = (mask == 1.0).cpu().numpy()
                for point, label in zip(scaled_points, labels):
                    if mask_np[point[1], point[0]] == 1 and label == 1:
                        onemask[mask_np] = 1
                    if mask_np[point[1], point[0]] == 1 and label == 0:
                        onemask[mask_np] = 0
            annotations.append(onemask >= 1)
        if len(bbox_points) >= 2:
            scaled_bbox_points = []
            for i, point in enumerate(bbox_points):
                x, y = int(point[0] * scale), int(point[1] * scale)
                x = max(min(x, new_w), 0)
                y = max(min(y, new_h), 0)
                scaled_bbox_points.append((x, y))
    
            for i in range(0, len(scaled_bbox_points) - 1, 2):
                x0, y0, x1, y1 = *scaled_bbox_points[i], *scaled_bbox_points[i + 1]
                
                intersection_area = torch.sum(masks[:, y0:y1, x0:x1], dim=(1, 2))
                masks_area = torch.sum(masks, dim=(1, 2))
                bbox_area = (y1 - y0) * (x1 - x0)
    
                union = bbox_area + masks_area - intersection_area
                iou = intersection_area / union
                max_iou_index = torch.argmax(iou)
    
                annotations.append(masks[max_iou_index].cpu().numpy())
    
        return fast_process(
            annotations=np.array(annotations),
            image=image,
            scale=(1024 // input_size),
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            with_contours=with_contours
        )

.. code:: ipython3

    def select_point(img: Image.Image, point_type: str, evt: gr.SelectData) -> Image.Image:
        """Gradio select callback."""
        img = img.convert("RGBA")
        x, y = evt.index[0], evt.index[1]
        point_radius = np.round(max(img.size) / 100)
        if point_type == "Object point":
            object_points.append((x, y))
            color = (30, 255, 30, 200)
        elif point_type == "Background point":
            background_points.append((x, y))
            color = (255, 30, 30, 200)
        elif point_type == "Bounding Box":
            bbox_points.append((x, y))
            color = (10, 10, 255, 255)
            if len(bbox_points) % 2 == 0:
                # Draw a rectangle if number of points is even
                new_img = Image.new("RGBA", img.size, (255, 255, 255, 0))
                _draw = ImageDraw.Draw(new_img)
                x0, y0, x1, y1 = *bbox_points[-2], *bbox_points[-1]
                x0, x1 = sorted([x0, x1])
                y0, y1 = sorted([y0, y1])
                # Save sorted order
                bbox_points[-2] = (x0, y0)
                bbox_points[-1] = (x1, y1)
                _draw.rectangle((x0, y0, x1, y1), fill=(*color[:-1], 90))
                img = Image.alpha_composite(img, new_img)
        # Draw a point
        ImageDraw.Draw(img).ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=color
        )
        return img
    
    def clear_points() -> (Image.Image, None):
        """Gradio clear points callback."""
        global object_points, background_points, bbox_points
        # global object_points; global background_points; global bbox_points
        object_points = []
        background_points = []
        bbox_points = []
        return last_image, None
    
    def save_last_picked_image(img: Image.Image) -> None:
        """Gradio callback saves the last used image."""
        global last_image
        last_image = img
        # If we change the input image
        # we should clear all the previous points
        clear_points()
        # Removes the segmentation map output
        return None
    
    with gr.Blocks(title="Fast SAM") as demo:
        with gr.Row(variant="panel"):
            original_img = gr.Image(label="Input", value=examples[0][0], type="pil")
            segmented_img = gr.Image(label="Segmentation Map", type="pil")
        with gr.Row():
            point_type = gr.Radio(
                ["Object point", "Background point", "Bounding Box"],
                value="Object point", label="Pixel selector type"
            )
            model_type = gr.Radio(
                ["FP32 model", "Quantized model"] if do_quantize.value else ["FP32 model"],
                value="FP32 model", label="Select model variant"
            )
        with gr.Row(variant="panel"):
            segment_button = gr.Button("Segment", variant="primary")
            clear_button = gr.Button("Clear points", variant="secondary")
        gr.Examples(examples, inputs=original_img,
                    fn=save_last_picked_image, run_on_click=True, outputs=segmented_img
        )
    
        # Callbacks
        original_img.select(select_point,
                            inputs=[original_img, point_type],
                            outputs=original_img)
        original_img.upload(save_last_picked_image, inputs=original_img, outputs=segmented_img)
        clear_button.click(clear_points, outputs=[original_img, segmented_img])
        segment_button.click(segment, inputs=[original_img, model_type], outputs=segmented_img)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name="your server name", server_port="server port in int")`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. raw:: html

    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

