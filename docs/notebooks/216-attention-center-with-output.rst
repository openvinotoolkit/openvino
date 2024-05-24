The attention center model with OpenVINO™
=========================================

This notebook demonstrates how to use the `attention center
model <https://github.com/google/attention-center/tree/main>`__ with
OpenVINO. This model is in the `TensorFlow Lite
format <https://www.tensorflow.org/lite>`__, which is supported in
OpenVINO now by TFLite frontend.

Eye tracking is commonly used in visual neuroscience and cognitive
science to answer related questions such as visual attention and
decision making. Computational models that predict where to look have
direct applications to a variety of computer vision tasks. The attention
center model takes an RGB image as input and return a 2D point as
output. This 2D point is the predicted center of human attention on the
image i.e. the most salient part of images, on which people pay
attention fist to. This allows find the most visually salient regions
and handle it as early as possible. For example, it could be used for
the latest generation image format (such as `JPEG
XL <https://github.com/libjxl/libjxl>`__), which supports encoding the
parts that you pay attention to fist. It can help to improve user
experience, image will appear to load faster.

Attention center model architecture is: > The attention center model is
a deep neural net, which takes an image as input, and uses a pre-trained
classification network, e.g, ResNet, MobileNet, etc., as the backbone.
Several intermediate layers that output from the backbone network are
used as input for the attention center prediction module. These
different intermediate layers contain different information e.g.,
shallow layers often contain low level information like
intensity/color/texture, while deeper layers usually contain higher and
more semantic information like shape/object. All are useful for the
attention prediction. The attention center prediction applies
convolution, deconvolution and/or resizing operator together with
aggregation and sigmoid function to generate a weighting map for the
attention center. And then an operator (the Einstein summation operator
in our case) can be applied to compute the (gravity) center from the
weighting map. An L2 norm between the predicted attention center and the
ground-truth attention center can be computed as the training loss.
Source: `Google AI blog
post <https://opensource.googleblog.com/2022/12/open-sourcing-attention-center-model.html>`__.

.. figure:: https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjxLCDJHzJNjB_von-vFlq8TJJFA41aB85T-QE3ZNxW8kshAf3HOEyIEJ4uggXjbJmZhsdj7j6i6mvvmXtyaxXJPm3JHuKILNRTPfX9KvICbFBRD8KNuDVmLABzYuhQci3BT2BqV-wM54IxaoAV1YDBbnpJC92UZfEBGvakLusiqND2AaPpWPr2gJV1/s1600/image4.png
   :alt: drawing

   drawing

The attention center model has been trained with images from the `COCO
dataset <https://cocodataset.org/#home>`__ annotated with saliency from
the `SALICON dataset <http://salicon.net/>`__.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Download the attention-center
   model <#download-the-attention-center-model>`__

   -  `Convert Tensorflow Lite model to OpenVINO IR
      format <#convert-tensorflow-lite-model-to-openvino-ir-format>`__

-  `Select inference device <#select-inference-device>`__
-  `Prepare image to use with attention-center
   model <#prepare-image-to-use-with-attention-center-model>`__
-  `Load input image <#load-input-image>`__
-  `Get result with OpenVINO IR
   model <#get-result-with-openvino-ir-model>`__

.. code:: ipython3

    %pip install "openvino>=2023.2.0"


.. parsed-literal::

    Requirement already satisfied: openvino>=2023.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2023.3.0)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.2.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.2.0) (2023.2.1)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import cv2
    
    import numpy as np
    import tensorflow as tf
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    import openvino as ov


.. parsed-literal::

    2024-02-09 23:49:23.781601: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-09 23:49:23.815218: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-09 23:49:24.361080: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download the attention-center model
-----------------------------------



Download the model as part of `attention-center
repo <https://github.com/google/attention-center/tree/main>`__. The repo
include model in folder ``./model``.

.. code:: ipython3

    if not Path('./attention-center').exists():
        ! git clone https://github.com/google/attention-center


.. parsed-literal::

    Cloning into 'attention-center'...


.. parsed-literal::

    remote: Enumerating objects: 168, done.[K
    remote: Counting objects:   0% (1/168)[K
remote: Counting objects:   1% (2/168)[K
remote: Counting objects:   2% (4/168)[K
remote: Counting objects:   3% (6/168)[K
remote: Counting objects:   4% (7/168)[K
remote: Counting objects:   5% (9/168)[K
remote: Counting objects:   6% (11/168)[K
remote: Counting objects:   7% (12/168)[K
remote: Counting objects:   8% (14/168)[K
remote: Counting objects:   9% (16/168)[K
remote: Counting objects:  10% (17/168)[K
remote: Counting objects:  11% (19/168)[K
remote: Counting objects:  12% (21/168)[K
remote: Counting objects:  13% (22/168)[K
remote: Counting objects:  14% (24/168)[K
remote: Counting objects:  15% (26/168)[K
remote: Counting objects:  16% (27/168)[K
remote: Counting objects:  17% (29/168)[K
remote: Counting objects:  18% (31/168)[K
remote: Counting objects:  19% (32/168)[K
remote: Counting objects:  20% (34/168)[K
remote: Counting objects:  21% (36/168)[K
remote: Counting objects:  22% (37/168)[K
remote: Counting objects:  23% (39/168)[K
remote: Counting objects:  24% (41/168)[K
remote: Counting objects:  25% (42/168)[K
remote: Counting objects:  26% (44/168)[K
remote: Counting objects:  27% (46/168)[K
remote: Counting objects:  28% (48/168)[K
remote: Counting objects:  29% (49/168)[K
remote: Counting objects:  30% (51/168)[K
remote: Counting objects:  31% (53/168)[K
remote: Counting objects:  32% (54/168)[K
remote: Counting objects:  33% (56/168)[K
remote: Counting objects:  34% (58/168)[K
remote: Counting objects:  35% (59/168)[K
remote: Counting objects:  36% (61/168)[K
remote: Counting objects:  37% (63/168)[K
remote: Counting objects:  38% (64/168)[K
remote: Counting objects:  39% (66/168)[K
remote: Counting objects:  40% (68/168)[K
remote: Counting objects:  41% (69/168)[K
remote: Counting objects:  42% (71/168)[K
remote: Counting objects:  43% (73/168)[K
remote: Counting objects:  44% (74/168)[K
remote: Counting objects:  45% (76/168)[K
remote: Counting objects:  46% (78/168)[K
remote: Counting objects:  47% (79/168)[K
remote: Counting objects:  48% (81/168)[K
remote: Counting objects:  49% (83/168)[K
remote: Counting objects:  50% (84/168)[K
remote: Counting objects:  51% (86/168)[K
remote: Counting objects:  52% (88/168)[K
remote: Counting objects:  53% (90/168)[K
remote: Counting objects:  54% (91/168)[K
remote: Counting objects:  55% (93/168)[K
remote: Counting objects:  56% (95/168)[K
remote: Counting objects:  57% (96/168)[K
remote: Counting objects:  58% (98/168)[K
remote: Counting objects:  59% (100/168)[K
remote: Counting objects:  60% (101/168)[K
remote: Counting objects:  61% (103/168)[K
remote: Counting objects:  62% (105/168)[K
remote: Counting objects:  63% (106/168)[K
remote: Counting objects:  64% (108/168)[K
remote: Counting objects:  65% (110/168)[K
remote: Counting objects:  66% (111/168)[K
remote: Counting objects:  67% (113/168)[K
remote: Counting objects:  68% (115/168)[K

.. parsed-literal::

    remote: Counting objects:  69% (116/168)[K
remote: Counting objects:  70% (118/168)[K
remote: Counting objects:  71% (120/168)[K
remote: Counting objects:  72% (121/168)[K
remote: Counting objects:  73% (123/168)[K
remote: Counting objects:  74% (125/168)[K
remote: Counting objects:  75% (126/168)[K
remote: Counting objects:  76% (128/168)[K
remote: Counting objects:  77% (130/168)[K
remote: Counting objects:  78% (132/168)[K
remote: Counting objects:  79% (133/168)[K
remote: Counting objects:  80% (135/168)[K
remote: Counting objects:  81% (137/168)[K
remote: Counting objects:  82% (138/168)[K
remote: Counting objects:  83% (140/168)[K
remote: Counting objects:  84% (142/168)[K
remote: Counting objects:  85% (143/168)[K
remote: Counting objects:  86% (145/168)[K
remote: Counting objects:  87% (147/168)[K
remote: Counting objects:  88% (148/168)[K
remote: Counting objects:  89% (150/168)[K
remote: Counting objects:  90% (152/168)[K
remote: Counting objects:  91% (153/168)[K
remote: Counting objects:  92% (155/168)[K
remote: Counting objects:  93% (157/168)[K
remote: Counting objects:  94% (158/168)[K
remote: Counting objects:  95% (160/168)[K
remote: Counting objects:  96% (162/168)[K
remote: Counting objects:  97% (163/168)[K
remote: Counting objects:  98% (165/168)[K
remote: Counting objects:  99% (167/168)[K
remote: Counting objects: 100% (168/168)[K
remote: Counting objects: 100% (168/168), done.[K
    remote: Compressing objects:   0% (1/132)[K
remote: Compressing objects:   1% (2/132)[K
remote: Compressing objects:   2% (3/132)[K
remote: Compressing objects:   3% (4/132)[K
remote: Compressing objects:   4% (6/132)[K
remote: Compressing objects:   5% (7/132)[K
remote: Compressing objects:   6% (8/132)[K
remote: Compressing objects:   7% (10/132)[K
remote: Compressing objects:   8% (11/132)[K

.. parsed-literal::

    remote: Compressing objects:   9% (12/132)[K
remote: Compressing objects:  10% (14/132)[K
remote: Compressing objects:  11% (15/132)[K
remote: Compressing objects:  12% (16/132)[K
remote: Compressing objects:  13% (18/132)[K

.. parsed-literal::

    remote: Compressing objects:  14% (19/132)[K
remote: Compressing objects:  15% (20/132)[K
remote: Compressing objects:  16% (22/132)[K

.. parsed-literal::

    remote: Compressing objects:  17% (23/132)[K
remote: Compressing objects:  18% (24/132)[K

.. parsed-literal::

    remote: Compressing objects:  19% (26/132)[K
remote: Compressing objects:  20% (27/132)[K
remote: Compressing objects:  21% (28/132)[K

.. parsed-literal::

    remote: Compressing objects:  22% (30/132)[K
remote: Compressing objects:  23% (31/132)[K
remote: Compressing objects:  24% (32/132)[K

.. parsed-literal::

    remote: Compressing objects:  25% (33/132)[K

.. parsed-literal::

    remote: Compressing objects:  26% (35/132)[K
remote: Compressing objects:  27% (36/132)[K
remote: Compressing objects:  28% (37/132)[K
remote: Compressing objects:  29% (39/132)[K
remote: Compressing objects:  30% (40/132)[K
remote: Compressing objects:  31% (41/132)[K
remote: Compressing objects:  32% (43/132)[K
remote: Compressing objects:  33% (44/132)[K
remote: Compressing objects:  34% (45/132)[K
remote: Compressing objects:  35% (47/132)[K
remote: Compressing objects:  36% (48/132)[K
remote: Compressing objects:  37% (49/132)[K
remote: Compressing objects:  38% (51/132)[K
remote: Compressing objects:  39% (52/132)[K
remote: Compressing objects:  40% (53/132)[K
remote: Compressing objects:  41% (55/132)[K
remote: Compressing objects:  42% (56/132)[K
remote: Compressing objects:  43% (57/132)[K
remote: Compressing objects:  44% (59/132)[K
remote: Compressing objects:  45% (60/132)[K
remote: Compressing objects:  46% (61/132)[K
remote: Compressing objects:  47% (63/132)[K
remote: Compressing objects:  48% (64/132)[K
remote: Compressing objects:  49% (65/132)[K
remote: Compressing objects:  50% (66/132)[K
remote: Compressing objects:  51% (68/132)[K
remote: Compressing objects:  52% (69/132)[K
remote: Compressing objects:  53% (70/132)[K
remote: Compressing objects:  54% (72/132)[K
remote: Compressing objects:  55% (73/132)[K
remote: Compressing objects:  56% (74/132)[K
remote: Compressing objects:  57% (76/132)[K
remote: Compressing objects:  58% (77/132)[K
remote: Compressing objects:  59% (78/132)[K
remote: Compressing objects:  60% (80/132)[K
remote: Compressing objects:  61% (81/132)[K
remote: Compressing objects:  62% (82/132)[K
remote: Compressing objects:  63% (84/132)[K
remote: Compressing objects:  64% (85/132)[K
remote: Compressing objects:  65% (86/132)[K
remote: Compressing objects:  66% (88/132)[K
remote: Compressing objects:  67% (89/132)[K
remote: Compressing objects:  68% (90/132)[K
remote: Compressing objects:  69% (92/132)[K
remote: Compressing objects:  70% (93/132)[K
remote: Compressing objects:  71% (94/132)[K
remote: Compressing objects:  72% (96/132)[K
remote: Compressing objects:  73% (97/132)[K
remote: Compressing objects:  74% (98/132)[K
remote: Compressing objects:  75% (99/132)[K
remote: Compressing objects:  76% (101/132)[K
remote: Compressing objects:  77% (102/132)[K
remote: Compressing objects:  78% (103/132)[K
remote: Compressing objects:  79% (105/132)[K
remote: Compressing objects:  80% (106/132)[K
remote: Compressing objects:  81% (107/132)[K
remote: Compressing objects:  82% (109/132)[K
remote: Compressing objects:  83% (110/132)[K
remote: Compressing objects:  84% (111/132)[K
remote: Compressing objects:  85% (113/132)[K
remote: Compressing objects:  86% (114/132)[K
remote: Compressing objects:  87% (115/132)[K
remote: Compressing objects:  88% (117/132)[K
remote: Compressing objects:  89% (118/132)[K
remote: Compressing objects:  90% (119/132)[K
remote: Compressing objects:  91% (121/132)[K
remote: Compressing objects:  92% (122/132)[K
remote: Compressing objects:  93% (123/132)[K
remote: Compressing objects:  94% (125/132)[K
remote: Compressing objects:  95% (126/132)[K
remote: Compressing objects:  96% (127/132)[K
remote: Compressing objects:  97% (129/132)[K
remote: Compressing objects:  98% (130/132)[K
remote: Compressing objects:  99% (131/132)[K
remote: Compressing objects: 100% (132/132)[K
remote: Compressing objects: 100% (132/132), done.[K
    Receiving objects:   0% (1/168)
Receiving objects:   1% (2/168)
Receiving objects:   2% (4/168)
Receiving objects:   3% (6/168)
Receiving objects:   4% (7/168)
Receiving objects:   5% (9/168)
Receiving objects:   6% (11/168)
Receiving objects:   7% (12/168)
Receiving objects:   8% (14/168)
Receiving objects:   9% (16/168)
Receiving objects:  10% (17/168)
Receiving objects:  11% (19/168)
Receiving objects:  12% (21/168)
Receiving objects:  13% (22/168)
Receiving objects:  14% (24/168)
Receiving objects:  15% (26/168)
Receiving objects:  16% (27/168)
Receiving objects:  17% (29/168)
Receiving objects:  18% (31/168)
Receiving objects:  19% (32/168)
Receiving objects:  20% (34/168)
Receiving objects:  21% (36/168)
Receiving objects:  22% (37/168)
Receiving objects:  23% (39/168)

.. parsed-literal::

    Receiving objects:  24% (41/168)
Receiving objects:  25% (42/168)
Receiving objects:  26% (44/168)
Receiving objects:  27% (46/168)
Receiving objects:  28% (48/168)
Receiving objects:  29% (49/168)
Receiving objects:  30% (51/168)
Receiving objects:  31% (53/168)
Receiving objects:  32% (54/168)

.. parsed-literal::

    Receiving objects:  33% (56/168), 1.46 MiB | 2.90 MiB/s

.. parsed-literal::

    Receiving objects:  34% (58/168), 1.46 MiB | 2.90 MiB/s
Receiving objects:  35% (59/168), 1.46 MiB | 2.90 MiB/s

.. parsed-literal::

    Receiving objects:  35% (59/168), 3.15 MiB | 3.13 MiB/s

.. parsed-literal::

    Receiving objects:  35% (60/168), 6.50 MiB | 3.23 MiB/s

.. parsed-literal::

    Receiving objects:  36% (61/168), 6.50 MiB | 3.23 MiB/s

.. parsed-literal::

    Receiving objects:  36% (62/168), 9.93 MiB | 3.27 MiB/s

.. parsed-literal::

    Receiving objects:  37% (63/168), 9.93 MiB | 3.27 MiB/s

.. parsed-literal::

    Receiving objects:  38% (64/168), 9.93 MiB | 3.27 MiB/s

.. parsed-literal::

    Receiving objects:  39% (66/168), 11.62 MiB | 3.29 MiB/s

.. parsed-literal::

    Receiving objects:  40% (68/168), 11.62 MiB | 3.29 MiB/s

.. parsed-literal::

    Receiving objects:  40% (68/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  41% (69/168), 13.29 MiB | 3.29 MiB/s

.. parsed-literal::

    Receiving objects:  42% (71/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  43% (73/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  44% (74/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  45% (76/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  46% (78/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  47% (79/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  48% (81/168), 13.29 MiB | 3.29 MiB/s

.. parsed-literal::

    Receiving objects:  49% (83/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  50% (84/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  51% (86/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  52% (88/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  53% (90/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  54% (91/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  55% (93/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  56% (95/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  57% (96/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  58% (98/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  59% (100/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  60% (101/168), 13.29 MiB | 3.29 MiB/s
Receiving objects:  61% (103/168), 13.29 MiB | 3.29 MiB/s

.. parsed-literal::

    Receiving objects:  61% (104/168), 16.65 MiB | 3.34 MiB/s

.. parsed-literal::

    Receiving objects:  61% (104/168), 20.00 MiB | 3.34 MiB/s

.. parsed-literal::

    Receiving objects:  61% (104/168), 23.32 MiB | 3.33 MiB/s

.. parsed-literal::

    Receiving objects:  62% (105/168), 23.32 MiB | 3.33 MiB/s

.. parsed-literal::

    Receiving objects:  63% (106/168), 24.84 MiB | 3.29 MiB/s

.. parsed-literal::

    remote: Total 168 (delta 73), reused 114 (delta 28), pack-reused 0[K
    Receiving objects:  64% (108/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  65% (110/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  66% (111/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  67% (113/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  68% (115/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  69% (116/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  70% (118/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  71% (120/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  72% (121/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  73% (123/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  74% (125/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  75% (126/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  76% (128/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  77% (130/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  78% (132/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  79% (133/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  80% (135/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  81% (137/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  82% (138/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  83% (140/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  84% (142/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  85% (143/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  86% (145/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  87% (147/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  88% (148/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  89% (150/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  90% (152/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  91% (153/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  92% (155/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  93% (157/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  94% (158/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  95% (160/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  96% (162/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  97% (163/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  98% (165/168), 24.84 MiB | 3.29 MiB/s
Receiving objects:  99% (167/168), 24.84 MiB | 3.29 MiB/s
Receiving objects: 100% (168/168), 24.84 MiB | 3.29 MiB/s
Receiving objects: 100% (168/168), 26.22 MiB | 3.29 MiB/s, done.
    Resolving deltas:   0% (0/73)
Resolving deltas:   8% (6/73)
Resolving deltas:  12% (9/73)
Resolving deltas:  19% (14/73)
Resolving deltas:  28% (21/73)
Resolving deltas:  38% (28/73)
Resolving deltas:  39% (29/73)
Resolving deltas:  49% (36/73)
Resolving deltas:  69% (51/73)
Resolving deltas:  73% (54/73)
Resolving deltas:  78% (57/73)
Resolving deltas:  84% (62/73)
Resolving deltas:  90% (66/73)
Resolving deltas:  97% (71/73)
Resolving deltas: 100% (73/73)
Resolving deltas: 100% (73/73), done.


Convert Tensorflow Lite model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The attention-center model is pre-trained model in TensorFlow Lite
format. In this Notebook the model will be converted to OpenVINO IR
format with model conversion API. For more information about model
conversion, see this
`page <https://docs.openvino.ai/2023.3/openvino_docs_model_processing_introduction.html>`__.
This step is also skipped if the model is already converted.

Also TFLite models format is supported in OpenVINO by TFLite frontend,
so the model can be passed directly to ``core.read_model()``. You can
find example in
`002-openvino-api <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/002-openvino-api>`__.

.. code:: ipython3

    tflite_model_path = Path("./attention-center/model/center.tflite")
    
    ir_model_path = Path("./model/ir_center_model.xml")
    
    core = ov.Core()
    
    if not ir_model_path.exists():
        model = ov.convert_model(tflite_model_path, input=[('image:0', [1,480,640,3], ov.Type.f32)])
        ov.save_model(model, ir_model_path)
        print("IR model saved to {}".format(ir_model_path))
    else:
        print("Read IR model from {}".format(ir_model_path))
        model = core.read_model(ir_model_path)


.. parsed-literal::

    IR model saved to model/ir_center_model.xml


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

    if "GPU" in device.value:
        core.set_property(device_name=device.value, properties={'INFERENCE_PRECISION_HINT': ov.Type.f32})
    compiled_model = core.compile_model(model=model, device_name=device.value)

Prepare image to use with attention-center model
------------------------------------------------



The attention-center model takes an RGB image with shape (480, 640) as
input.

.. code:: ipython3

    class Image():
        def __init__(self, model_input_image_shape, image_path=None, image=None):
            self.model_input_image_shape = model_input_image_shape
            self.image = None
            self.real_input_image_shape = None
    
            if image_path is not None:
                self.image = cv2.imread(str(image_path))
                self.real_input_image_shape = self.image.shape
            elif image is not None:
                self.image = image
                self.real_input_image_shape = self.image.shape
            else:
                raise Exception("Sorry, image can't be found, please, specify image_path or image")
    
        def prepare_image_tensor(self):
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(rgb_image, (self.model_input_image_shape[1], self.model_input_image_shape[0]))
    
            image_tensor = tf.constant(np.expand_dims(resized_image, axis=0),
                                       dtype=tf.float32)
            return image_tensor
    
        def scalt_center_to_real_image_shape(self, predicted_center):
            new_center_y = round(predicted_center[0] * self.real_input_image_shape[1] / self.model_input_image_shape[1])
            new_center_x = round(predicted_center[1] * self.real_input_image_shape[0] / self.model_input_image_shape[0])
            return (int(new_center_y), int(new_center_x))
    
        def draw_attention_center_point(self, predicted_center):
            image_with_circle = cv2.circle(self.image,
                                           predicted_center,
                                           radius=10,
                                           color=(3, 3, 255),
                                           thickness=-1)
            return image_with_circle
    
        def print_image(self, predicted_center=None):
            image_to_print = self.image
            if predicted_center is not None:
                image_to_print = self.draw_attention_center_point(predicted_center)
    
            plt.imshow(cv2.cvtColor(image_to_print, cv2.COLOR_BGR2RGB))

Load input image
----------------



Upload input image using file loading button

.. code:: ipython3

    import ipywidgets as widgets
    
    load_file_widget = widgets.FileUpload(
        accept="image/*", multiple=False, description="Image file",
    )
    
    load_file_widget




.. parsed-literal::

    FileUpload(value=(), accept='image/*', description='Image file')



.. code:: ipython3

    import io
    import PIL
    from urllib.request import urlretrieve
    
    img_path = Path("data/coco.jpg")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        img_path,
    )
    
    # read uploaded image
    image = PIL.Image.open(io.BytesIO(list(load_file_widget.value.values())[-1]['content'])) if load_file_widget.value else PIL.Image.open(img_path)
    image.convert("RGB")
    
    input_image = Image((480, 640), image=(np.ascontiguousarray(image)[:, :, ::-1]).astype(np.uint8))
    image_tensor = input_image.prepare_image_tensor()
    input_image.print_image()


.. parsed-literal::

    2024-02-09 23:49:38.816368: E tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: forward compatibility was attempted on non supported HW
    2024-02-09 23:49:38.816405: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:168] retrieving CUDA diagnostic information for host: iotg-dev-workstation-07
    2024-02-09 23:49:38.816409: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:175] hostname: iotg-dev-workstation-07
    2024-02-09 23:49:38.816551: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:199] libcuda reported version is: 470.223.2
    2024-02-09 23:49:38.816566: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:203] kernel reported version is: 470.182.3
    2024-02-09 23:49:38.816569: E tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:312] kernel version 470.182.3 does not match DSO version 470.223.2 -- cannot find working devices in this configuration



.. image:: 216-attention-center-with-output_files/216-attention-center-with-output_15_1.png


Get result with OpenVINO IR model
---------------------------------



.. code:: ipython3

    output_layer = compiled_model.output(0)
    
    # make inference, get result in input image resolution
    res = compiled_model([image_tensor])[output_layer]
    # scale point to original image resulution
    predicted_center = input_image.scalt_center_to_real_image_shape(res[0])
    print(f'Prediction attention center point {predicted_center}')
    input_image.print_image(predicted_center)


.. parsed-literal::

    Prediction attention center point (292, 277)



.. image:: 216-attention-center-with-output_files/216-attention-center-with-output_17_1.png

