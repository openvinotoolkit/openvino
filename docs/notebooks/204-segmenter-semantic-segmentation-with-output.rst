Semantic Segmentation with OpenVINO™ using Segmenter
====================================================

Semantic segmentation is a difficult computer vision problem with many
applications such as autonomous driving, robotics, augmented reality,
and many others. Its goal is to assign labels to each pixel according to
the object it belongs to, creating so-called segmentation masks. To
properly assign this label, the model needs to consider the local as
well as global context of the image. This is where transformers offer
their advantage as they work well in capturing global context.

Segmenter is based on Vision Transformer working as an encoder, and Mask
Transformer working as a decoder. With this configuration, it achieves
good results on different datasets such as ADE20K, Pascal Context, and
Cityscapes. It works as shown in the diagram below, by taking the image,
splitting it into patches, and then encoding these patches. Mask
transformer combines encoded patches with class masks and decodes them
into a segmentation map as the output, where each pixel has a label
assigned to it.

|Segmenter diagram| > Credits for this image go to `original authors of
Segmenter <https://github.com/rstrudel/segmenter>`__.

More about the model and its details can be found in the following
paper: `Segmenter: Transformer for Semantic
Segmentation <https://arxiv.org/abs/2105.05633>`__ or in the
`repository <https://github.com/rstrudel/segmenter>`__.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Get and prepare PyTorch model <#get-and-prepare-pytorch-model>`__

   -  `Prerequisites <#prerequisites>`__
   -  `Loading PyTorch model <#loading-pytorch-model>`__

-  `Preparing preprocessing and visualization
   functions <#preparing-preprocessing-and-visualization-functions>`__

   -  `Preprocessing <#preprocessing>`__
   -  `Visualization <#visualization>`__

-  `Validation of inference of original
   model <#validation-of-inference-of-original-model>`__
-  `Convert PyTorch model to OpenVINO Intermediate Representation
   (IR) <#convert-pytorch-model-to-openvino-intermediate-representation-ir>`__
-  `Verify converted model
   inference <#verify-converted-model-inference>`__

   -  `Select inference device <#select-inference-device>`__

-  `Benchmarking performance of converted
   model <#benchmarking-performance-of-converted-model>`__

.. |Segmenter diagram| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/93932510/f57979e7-fd3b-449f-bf01-afe0f965abbc

To demonstrate how to convert and use Segmenter in OpenVINO, this
notebook consists of the following steps:

-  Preparing PyTorch Segmenter model
-  Preparing preprocessing and visualization functions
-  Validating inference of original model
-  Converting PyTorch model to OpenVINO IR
-  Validating inference of the converted model
-  Benchmark performance of the converted model

Get and prepare PyTorch model
-----------------------------



The first thing we’ll need to do is clone
`repository <https://github.com/rstrudel/segmenter>`__ containing model
and helper functions. We will use Tiny model with mask transformer, that
is ``Seg-T-Mask/16``. There are also better, but much larger models
available in the linked repo. This model is pre-trained on
`ADE20K <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`__
dataset used for segmentation.

The code from the repository already contains functions that create
model and load weights, but we will need to download config and trained
weights (checkpoint) file and add some additional helper functions.

Prerequisites
~~~~~~~~~~~~~



.. code:: ipython3

    import sys
    from pathlib import Path
    
    # clone Segmenter repo
    if not Path("segmenter").exists():
        !git clone https://github.com/rstrudel/segmenter
    else:
        print("Segmenter repo already cloned")
    
    # include path to Segmenter repo to use its functions
    sys.path.append("./segmenter")


.. parsed-literal::

    Cloning into 'segmenter'...


.. parsed-literal::

    remote: Enumerating objects: 268, done.[K
    Receiving objects:   0% (1/268)
Receiving objects:   1% (3/268)
Receiving objects:   2% (6/268)
Receiving objects:   3% (9/268)
Receiving objects:   4% (11/268)
Receiving objects:   5% (14/268)
Receiving objects:   6% (17/268)
Receiving objects:   7% (19/268)
Receiving objects:   8% (22/268)
Receiving objects:   9% (25/268)
Receiving objects:  10% (27/268)
Receiving objects:  11% (30/268)
Receiving objects:  12% (33/268)
Receiving objects:  13% (35/268)
Receiving objects:  14% (38/268)
Receiving objects:  15% (41/268)
Receiving objects:  16% (43/268)
Receiving objects:  17% (46/268)
Receiving objects:  18% (49/268)
Receiving objects:  19% (51/268)
Receiving objects:  20% (54/268)
Receiving objects:  21% (57/268)
Receiving objects:  22% (59/268)

.. parsed-literal::

    Receiving objects:  23% (62/268)

.. parsed-literal::

    Receiving objects:  24% (65/268)

.. parsed-literal::

    Receiving objects:  24% (65/268), 3.68 MiB | 3.63 MiB/s

.. parsed-literal::

    Receiving objects:  24% (66/268), 7.47 MiB | 3.70 MiB/s

.. parsed-literal::

    Receiving objects:  25% (67/268), 7.47 MiB | 3.70 MiB/s

.. parsed-literal::

    Receiving objects:  25% (68/268), 11.26 MiB | 3.73 MiB/s

.. parsed-literal::

    Receiving objects:  26% (70/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  27% (73/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  28% (76/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  29% (78/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  30% (81/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  31% (84/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  32% (86/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  33% (89/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  34% (92/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  35% (94/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  36% (97/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  37% (100/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  38% (102/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  39% (105/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  40% (108/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  41% (110/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  42% (113/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  43% (116/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  44% (118/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  45% (121/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  46% (124/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  47% (126/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  48% (129/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  49% (132/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  50% (134/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  51% (137/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  52% (140/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  53% (143/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  54% (145/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  55% (148/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  56% (151/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  57% (153/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  58% (156/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  59% (159/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  60% (161/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  61% (164/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  62% (167/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  63% (169/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  64% (172/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  65% (175/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  66% (177/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  67% (180/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  68% (183/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  69% (185/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  70% (188/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  71% (191/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  72% (193/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  73% (196/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  74% (199/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  75% (201/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  76% (204/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  77% (207/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  78% (210/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  79% (212/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  80% (215/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  81% (218/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  82% (220/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  83% (223/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  84% (226/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  85% (228/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  86% (231/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  87% (234/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  88% (236/268), 11.26 MiB | 3.73 MiB/s
Receiving objects:  89% (239/268), 11.26 MiB | 3.73 MiB/s

.. parsed-literal::

    Receiving objects:  90% (242/268), 11.26 MiB | 3.73 MiB/s

.. parsed-literal::

    Receiving objects:  91% (244/268), 11.26 MiB | 3.73 MiB/s

.. parsed-literal::

    Receiving objects:  92% (247/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  93% (250/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  94% (252/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  95% (255/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  96% (258/268), 13.11 MiB | 3.72 MiB/s

.. parsed-literal::

    Receiving objects:  97% (260/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  98% (263/268), 13.11 MiB | 3.72 MiB/s
Receiving objects:  99% (266/268), 13.11 MiB | 3.72 MiB/s

.. parsed-literal::

    Receiving objects:  99% (267/268), 15.03 MiB | 3.73 MiB/s

.. parsed-literal::

    remote: Total 268 (delta 0), reused 0 (delta 0), pack-reused 268[K
    Receiving objects: 100% (268/268), 15.03 MiB | 3.73 MiB/s
Receiving objects: 100% (268/268), 15.34 MiB | 3.73 MiB/s, done.
    Resolving deltas:   0% (0/117)
Resolving deltas:   1% (2/117)
Resolving deltas:   2% (3/117)
Resolving deltas:   3% (4/117)
Resolving deltas:   5% (6/117)
Resolving deltas:   7% (9/117)
Resolving deltas:   8% (10/117)
Resolving deltas:   9% (11/117)
Resolving deltas:  10% (12/117)
Resolving deltas:  11% (14/117)
Resolving deltas:  13% (16/117)
Resolving deltas:  15% (18/117)
Resolving deltas:  26% (31/117)
Resolving deltas:  33% (39/117)
Resolving deltas:  54% (64/117)
Resolving deltas:  56% (66/117)
Resolving deltas:  76% (90/117)
Resolving deltas:  80% (94/117)
Resolving deltas:  81% (95/117)
Resolving deltas:  82% (96/117)
Resolving deltas: 100% (117/117)
Resolving deltas: 100% (117/117), done.


.. code:: ipython3

    # Installing requirements
    %pip install -q "openvino>=2023.1.0"
    %pip install -r segmenter/requirements.txt


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Requirement already satisfied: torch in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 1)) (2.1.0+cpu)
    Requirement already satisfied: click in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 2)) (8.1.7)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 3)) (1.23.5)


.. parsed-literal::

    Collecting einops (from -r segmenter/requirements.txt (line 4))
      Using cached einops-0.7.0-py3-none-any.whl.metadata (13 kB)


.. parsed-literal::

    Collecting python-hostlist (from -r segmenter/requirements.txt (line 5))
      Using cached python_hostlist-1.23.0-py3-none-any.whl
    Requirement already satisfied: tqdm in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 6)) (4.66.1)
    Requirement already satisfied: requests in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 7)) (2.31.0)
    Requirement already satisfied: pyyaml in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from -r segmenter/requirements.txt (line 8)) (6.0.1)


.. parsed-literal::

    Collecting timm==0.4.12 (from -r segmenter/requirements.txt (line 9))
      Using cached timm-0.4.12-py3-none-any.whl (376 kB)


.. parsed-literal::

    Collecting mmcv==1.3.8 (from -r segmenter/requirements.txt (line 10))
      Using cached mmcv-1.3.8-py2.py3-none-any.whl


.. parsed-literal::

    Collecting mmsegmentation==0.14.1 (from -r segmenter/requirements.txt (line 11))


.. parsed-literal::

      Using cached mmsegmentation-0.14.1-py3-none-any.whl (201 kB)
    Requirement already satisfied: torchvision in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from timm==0.4.12->-r segmenter/requirements.txt (line 9)) (0.16.0+cpu)
    Requirement already satisfied: addict in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from mmcv==1.3.8->-r segmenter/requirements.txt (line 10)) (2.4.0)
    Requirement already satisfied: Pillow in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from mmcv==1.3.8->-r segmenter/requirements.txt (line 10)) (10.2.0)


.. parsed-literal::

    Collecting yapf (from mmcv==1.3.8->-r segmenter/requirements.txt (line 10))


.. parsed-literal::

      Using cached yapf-0.40.2-py3-none-any.whl.metadata (45 kB)
    Requirement already satisfied: matplotlib in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (3.7.4)
    Requirement already satisfied: prettytable in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (3.9.0)
    Requirement already satisfied: filelock in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (3.13.1)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (4.9.0)
    Requirement already satisfied: sympy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (1.12)
    Requirement already satisfied: networkx in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (3.1)
    Requirement already satisfied: jinja2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (3.1.3)
    Requirement already satisfied: fsspec in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch->-r segmenter/requirements.txt (line 1)) (2023.10.0)


.. parsed-literal::

    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->-r segmenter/requirements.txt (line 7)) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->-r segmenter/requirements.txt (line 7)) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->-r segmenter/requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests->-r segmenter/requirements.txt (line 7)) (2024.2.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jinja2->torch->-r segmenter/requirements.txt (line 1)) (2.1.5)


.. parsed-literal::

    Requirement already satisfied: contourpy>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (1.1.1)
    Requirement already satisfied: cycler>=0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (4.48.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (23.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (2.8.2)
    Requirement already satisfied: importlib-resources>=3.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (6.1.1)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prettytable->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (0.2.13)
    Requirement already satisfied: mpmath>=0.19 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from sympy->torch->-r segmenter/requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: importlib-metadata>=6.6.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from yapf->mmcv==1.3.8->-r segmenter/requirements.txt (line 10)) (7.0.1)
    Requirement already satisfied: platformdirs>=3.5.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from yapf->mmcv==1.3.8->-r segmenter/requirements.txt (line 10)) (4.2.0)


.. parsed-literal::

    Collecting tomli>=2.0.1 (from yapf->mmcv==1.3.8->-r segmenter/requirements.txt (line 10))
      Using cached tomli-2.0.1-py3-none-any.whl (12 kB)


.. parsed-literal::

    Requirement already satisfied: zipp>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from importlib-metadata>=6.6.0->yapf->mmcv==1.3.8->-r segmenter/requirements.txt (line 10)) (3.17.0)


.. parsed-literal::

    Requirement already satisfied: six>=1.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib->mmsegmentation==0.14.1->-r segmenter/requirements.txt (line 11)) (1.16.0)


.. parsed-literal::

    Using cached einops-0.7.0-py3-none-any.whl (44 kB)
    Using cached yapf-0.40.2-py3-none-any.whl (254 kB)


.. parsed-literal::

    Installing collected packages: python-hostlist, tomli, einops, yapf, mmsegmentation, mmcv, timm


.. parsed-literal::

      Attempting uninstall: tomli
        Found existing installation: tomli 1.2.3
        Uninstalling tomli-1.2.3:
          Successfully uninstalled tomli-1.2.3


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    black 21.7b0 requires tomli<2.0.0,>=0.2.6, but you have tomli 2.0.1 which is incompatible.
    Successfully installed einops-0.7.0 mmcv-1.3.8 mmsegmentation-0.14.1 python-hostlist-1.23.0 timm-0.4.12 tomli-2.0.1 yapf-0.40.2


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import numpy as np
    import yaml
    
    # Fetch the notebook utils script from the openvino_notebooks repo
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file, load_image

We’ll need ``timm``, ``mmsegmentation``, ``einops`` and ``mmcv``, to use
functions from segmenter repo

First, we will clone the Segmenter repo and then download weights and
config for our model.

.. code:: ipython3

    # download config and pretrained model weights
    # here we use tiny model, there are also better but larger models available in repository
    WEIGHTS_LINK = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/segmenter/checkpoints/ade20k/seg_tiny_mask/checkpoint.pth"
    CONFIG_LINK = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/segmenter/checkpoints/ade20k/seg_tiny_mask/variant.yml"
    
    MODEL_DIR = Path("model/")
    MODEL_DIR.mkdir(exist_ok=True)
    
    download_file(WEIGHTS_LINK, directory=MODEL_DIR, show_progress=True)
    download_file(CONFIG_LINK, directory=MODEL_DIR, show_progress=True)
    
    WEIGHT_PATH = MODEL_DIR / "checkpoint.pth"
    CONFIG_PATH = MODEL_DIR / "variant.yaml"



.. parsed-literal::

    model/checkpoint.pth:   0%|          | 0.00/26.4M [00:00<?, ?B/s]



.. parsed-literal::

    model/variant.yml:   0%|          | 0.00/940 [00:00<?, ?B/s]


Loading PyTorch model
~~~~~~~~~~~~~~~~~~~~~



PyTorch models are usually an instance of
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary containing model weights.
Typical steps to get the model are therefore:

1. Create an instance of the model class
2. Load checkpoint state dict, which contains pre-trained model weights
3. Turn the model to evaluation mode, to switch some operations to
   inference mode

We will now use already provided helper functions from repository to
initialize the model.

.. code:: ipython3

    from segmenter.segm.model.factory import load_model
    
    pytorch_model, config = load_model(WEIGHT_PATH)
    # put model into eval mode, to set it for inference
    pytorch_model.eval()
    print("PyTorch model loaded and ready for inference.")


.. parsed-literal::

    PyTorch model loaded and ready for inference.


Load normalization settings from config file.

.. code:: ipython3

    from segmenter.segm.data.utils import STATS
    # load normalization name, in our case "vit" since we are using transformer
    normalization_name = config["dataset_kwargs"]["normalization"]
    # load normalization params, mean and std from STATS
    normalization = STATS[normalization_name]


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


Preparing preprocessing and visualization functions
---------------------------------------------------



Now we will define utility functions for preprocessing and visualizing
the results.

Preprocessing
~~~~~~~~~~~~~



Inference input is tensor with shape ``[1, 3, H, W]`` in ``B, C, H, W``
format, where:

-  ``B`` - batch size (in our case 1, as we are just adding 1 with
   unsqueeze)
-  ``C`` - image channels (in our case RGB - 3)
-  ``H`` - image height
-  ``W`` - image width

Resizing to the correct scale and splitting to batches is done inside
inference, so we don’t need to resize or split the image in
preprocessing.

Model expects images in RGB channels format, scaled to [0, 1] range and
normalized with given mean and standard deviation provided in
``config.yml``.

.. code:: ipython3

    from PIL import Image
    import torch
    import torchvision.transforms.functional as F
    
    
    def preprocess(im: Image, normalization: dict) -> torch.Tensor:
        """
        Preprocess image: scale, normalize and unsqueeze
    
        :param im: input image
        :param normalization: dictionary containing normalization data from config file
        :return:
                im: processed (scaled and normalized) image
        """
        # change PIL image to tensor and scale to [0, 1]
        im = F.pil_to_tensor(im).float() / 255
        # normalize by given mean and standard deviation
        im = F.normalize(im, normalization["mean"], normalization["std"])
        # change dim from [C, H, W] to [1, C, H, W]
        im = im.unsqueeze(0)
    
        return im

Visualization
~~~~~~~~~~~~~



Inference output contains labels assigned to each pixel, so the output
in our case is ``[150, H, W]`` in ``CL, H, W`` format where:

-  ``CL`` - number of classes for labels (in our case 150)
-  ``H`` - image height
-  ``W`` - image width

Since we want to visualize this output, we reduce dimensions to
``[1, H, W]`` where we keep only class with the highest value as that is
the predicted label. We then combine original image with colors
corresponding to the inferred labels.

.. code:: ipython3

    from segmenter.segm.data.utils import dataset_cat_description, seg_to_rgb
    from segmenter.segm.data.ade20k import ADE20K_CATS_PATH
    
    
    def apply_segmentation_mask(pil_im: Image, results: torch.Tensor) -> Image:
        """
        Combine segmentation masks with the image
    
        :param pil_im: original input image
        :param results: tensor containing segmentation masks for each pixel
        :return:
                pil_blend: image with colored segmentation masks overlay
        """
        cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
    
        # 3D array, where each pixel has values for all classes, take index of max as label
        seg_map = results.argmax(0, keepdim=True)
        # transform label id to colors
        seg_rgb = seg_to_rgb(seg_map, cat_colors)
        seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        pil_seg = Image.fromarray(seg_rgb[0])
    
        # overlay segmentation mask over original image
        pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
    
        return pil_blend

Validation of inference of original model
-----------------------------------------



Now that we have everything ready, we can perform segmentation on
example image ``coco_hollywood.jpg``.

.. code:: ipython3

    from segmenter.segm.model.utils import inference
    
    # load image with PIL
    image = load_image("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg")
    # load_image reads the image in BGR format, [:,:,::-1] reshape transfroms it to RGB
    pil_image = Image.fromarray(image[:,:,::-1])
    
    # preprocess image with normalization params loaded in previous steps
    image = preprocess(pil_image, normalization)
    
    # inference function needs some meta parameters, where we specify that we don't flip images in inference mode
    im_meta = dict(flip=False)
    # perform inference with function from repository
    original_results = inference(model=pytorch_model,
                                 ims=[image],
                                 ims_metas=[im_meta],
                                 ori_shape=image.shape[2:4],
                                 window_size=config["inference_kwargs"]["window_size"],
                                 window_stride=config["inference_kwargs"]["window_stride"],
                                 batch_size=2)

After inference is complete, we need to transform output to segmentation
mask where each class has specified color, using helper functions from
previous steps.

.. code:: ipython3

    # combine segmentation mask with image
    blended_image = apply_segmentation_mask(pil_image, original_results)
    
    # show image with segmentation mask overlay
    blended_image




.. image:: 204-segmenter-semantic-segmentation-with-output_files/204-segmenter-semantic-segmentation-with-output_21_0.png



We can see that model segments the image into meaningful parts. Since we
are using tiny variant of model, the result is not as good as it is with
larger models, but it already shows nice segmentation performance.

Convert PyTorch model to OpenVINO Intermediate Representation (IR)
------------------------------------------------------------------



Now that we’ve verified that the inference of PyTorch model works, we
will convert it to OpenVINO IR format.

To do this, we first get input dimensions from the model configuration
file and create torch dummy input. Input dimensions are in our case
``[2, 3, 512, 512]`` in ``B, C, H, W]`` format, where:

-  ``B`` - batch size
-  ``C`` - image channels (in our case RGB - 3)
-  ``H`` - model input image height
-  ``W`` - model input image width

..

   Note that H and W are here fixed to 512, as this is required by the
   model. Resizing is done inside the inference function from the
   original repository.

After that, we use ``ov.convert_model`` function from PyTorch to convert
the model to OpenVINO model, which is ready to use in Python interface
but can also be serialized to OpenVINO IR format for future execution
using ``ov.save_model``. The process can generate some warnings, but
they are not a problem.

.. code:: ipython3

    import openvino as ov
    
    # get input sizes from config file
    batch_size = 2
    channels = 3
    image_size = config["dataset_kwargs"]["image_size"]
    
    # make dummy input with correct shapes obtained from config file
    dummy_input = torch.randn(batch_size, channels, image_size, image_size)
    
    model = ov.convert_model(pytorch_model, example_input=dummy_input, input=([batch_size, channels, image_size, image_size], ))
    # serialize model for saving IR
    ov.save_model(model, MODEL_DIR / "segmenter.xml")


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/utils.py:69: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if H % patch_size > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/utils.py:71: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if W % patch_size > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/vit.py:122: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if x.shape[1] != pos_embed.shape[1]:


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/decoder.py:100: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/utils.py:85: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if extra_h > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/204-segmenter-semantic-segmentation/./segmenter/segm/model/utils.py:87: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if extra_w > 0:


Verify converted model inference
--------------------------------



To test that model was successfully converted, we can use same inference
function from original repository, but we need to make custom class.

``SegmenterOV`` class contains OpenVINO model, with all attributes and
methods required by inference function. This way we don’t need to write
any additional custom code required to process input.

.. code:: ipython3

    class SegmenterOV:
        """
        Class containing OpenVINO model with all attributes required to work with inference function.
    
        :param model: compiled OpenVINO model
        :type model: CompiledModel
        :param output_blob: output blob used in inference
        :type output_blob: ConstOutput
        :param config: config file containing data about model and its requirements
        :type config: dict
        :param n_cls: number of classes to be predicted
        :type n_cls: int
        :param normalization:
        :type normalization: dict
    
        """
    
        def __init__(self, model_path: Path, device:str = "CPU"):
            """
            Constructor method.
            Initializes OpenVINO model and sets all required attributes
    
            :param model_path: path to model's .xml file, also containing variant.yml
            :param device: device string for selecting inference device
            """
            # init OpenVino core
            core = ov.Core()
            # read model
            model_xml = core.read_model(model_path)
            self.model = core.compile_model(model_xml, device)
            self.output_blob = self.model.output(0)
    
            # load model configs
            variant_path = Path(model_path).parent / "variant.yml"
            with open(variant_path, "r") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
    
            # load normalization specs from config
            normalization_name = self.config["dataset_kwargs"]["normalization"]
            self.normalization = STATS[normalization_name]
    
            # load number of classes from config
            self.n_cls = self.config["net_kwargs"]["n_cls"]
    
        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Perform inference on data and return the result in Tensor format
    
            :param data: input data to model
            :return: data inferred by model
            """
            return torch.from_numpy(self.model(data)[self.output_blob])

Now that we have created ``SegmenterOV`` helper class, we can use it in
inference function.

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



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



.. code:: ipython3

    # load model into SegmenterOV class
    model = SegmenterOV(MODEL_DIR / "segmenter.xml", device.value)

.. code:: ipython3

    # perform inference with same function as in case of PyTorch model from repository
    results = inference(model=model,
                        ims=[image],
                        ims_metas=[im_meta],
                        ori_shape=image.shape[2:4],
                        window_size=model.config["inference_kwargs"]["window_size"],
                        window_stride=model.config["inference_kwargs"]["window_stride"],
                        batch_size=2)

.. code:: ipython3

    # combine segmentation mask with image
    converted_blend = apply_segmentation_mask(pil_image, results)
    
    # show image with segmentation mask overlay
    converted_blend




.. image:: 204-segmenter-semantic-segmentation-with-output_files/204-segmenter-semantic-segmentation-with-output_32_0.png



As we can see, we get the same results as with original model.

Benchmarking performance of converted model
-------------------------------------------



Finally, use the OpenVINO `Benchmark
Tool <https://docs.openvino.ai/2023.3/openvino_sample_benchmark_tool.html>`__
to measure the inference performance of the model.

   NOTE: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

..

   Keep in mind that the authors of original paper used V100 GPU, which
   is significantly more powerful than the CPU used to obtain the
   following throughput. Therefore, FPS can’t be compared directly.

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Inference FP32 model (OpenVINO IR)
    !benchmark_app -m ./model/segmenter.xml -d $device.value -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2023.3.0-13775-ceeafaf64f3-releases/2023/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 23.09 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     im (node: im) : f32 / [...] / [2,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     y (node: aten::upsample_bilinear2d/Interpolate) : f32 / [...] / [2,150,512,512]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 2
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     im (node: im) : u8 / [N,C,H,W] / [2,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     y (node: aten::upsample_bilinear2d/Interpolate) : f32 / [...] / [2,150,512,512]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 385.39 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 6
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'im'!. This input will be filled with random values!
    [ INFO ] Fill input 'im' with random values 


.. parsed-literal::

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 210.45 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            1686 iterations
    [ INFO ] Duration:         120531.12 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        429.25 ms
    [ INFO ]    Average:       428.34 ms
    [ INFO ]    Min:           354.96 ms
    [ INFO ]    Max:           506.55 ms
    [ INFO ] Throughput:   27.98 FPS

