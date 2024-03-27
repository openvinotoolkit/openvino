Object detection and masking from prompts with GroundedSAM (GroundingDINO + SAM) and OpenVINO
=============================================================================================

In this notebook, we provide the OpenVINO™ optimization for the
combination of GroundingDINO + SAM =
`GroundedSAM <https://github.com/IDEA-Research/Grounded-Segment-Anything>`__
on Intel® platforms.

GroundedSAM aims to detect and segment anything with text inputs.
GroundingDINO is a language-guided query selection module to enhance
object detection using input text. It selects relevant features from
image and text inputs and returns predicted boxes with detections. The
Segment Anything Model (SAM) produces high quality object masks from
input prompts such as points or boxes, and it can be used to generate
masks for all objects in an image. We use box predictions from
GroundingDINO to mask the original image.

More details about the model can be found in the
`paper <https://arxiv.org/abs/2401.14159>`__, and the official
`repository <https://github.com/IDEA-Research/Grounded-Segment-Anything>`__.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/5703039/3c19063a-c60a-4d5d-b534-e1305a854180
   :alt: image

   image

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Clone repository and install
   requirements <#clone-repository-and-install-requirements>`__
-  `Download checkpoints and load PyTorch
   model <#download-checkpoints-and-load-pytorch-model>`__
-  `Convert GroundingDINO to OpenVINO IR
   format <#convert-groundingdino-to-openvino-ir-format>`__
-  `Run OpenVINO optimized
   GroundingDINO <#run-openvino-optimized-groundingdino>`__
-  `Convert SAM to OpenVINO IR <#convert-sam-to-openvino-ir>`__
-  `Combine GroundingDINO + SAM
   (GroundedSAM) <#combine-groundingdino--sam-groundedsam>`__
-  `Interactive GroundedSAM <#interactive-groundedsam>`__
-  `Cleanup <#cleanup>`__

Clone repositories and install requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q timm --extra-index-url https://download.pytorch.org/whl/cpu  # is needed for torch
    %pip install -q "openvino>=2024.0" opencv-python supervision transformers yapf pycocotools addict gradio


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    paddleclas 2.5.1 requires faiss-cpu==1.7.1.post2, but you have faiss-cpu 1.8.0 which is incompatible.
    paddleclas 2.5.1 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    ppgan 2.1.0 requires imageio==2.9.0, but you have imageio 2.34.0 which is incompatible.
    ppgan 2.1.0 requires librosa==0.8.1, but you have librosa 0.9.2 which is incompatible.
    ppgan 2.1.0 requires opencv-python<=4.6.0.66, but you have opencv-python 4.9.0.80 which is incompatible.
    pyannote-audio 2.0.1 requires torchaudio<1.0,>=0.10, but you have torchaudio 2.2.1+cpu which is incompatible.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


For faster computation and to limit RAM by default we use
``EfficientSAM`` for segmentation, but if you wish more accurate
segmentation you can select vanilla ``SAM``.

.. code:: ipython3

    import ipywidgets

    sam_type_widget = ipywidgets.Dropdown(
        options=['EfficientSAM', 'SAM'],
        value='EfficientSAM',
        description='Segment Anything type:',
    )
    sam_type_widget




.. parsed-literal::

    Dropdown(description='Segment Anything type:', options=('EfficientSAM', 'SAM'), value='EfficientSAM')



.. code:: ipython3

    use_efficient_sam = sam_type_widget.value == 'EfficientSAM'

.. code:: ipython3

    from pathlib import Path
    import sys
    import os

    repo_dir = Path("Grounded-Segment-Anything")
    ground_dino_dir = Path('GroundingDINO')
    efficient_sam_dir = Path('EfficientSAM')

    # we use grounding dino from a fork which contains modifications that allow conversion to OpenVINO IR format
    if not ground_dino_dir.exists():
        !git clone https://github.com/wenyi5608/GroundingDINO/
    if use_efficient_sam and not efficient_sam_dir.exists():
        !git clone https://github.com/yformer/EfficientSAM
    if not use_efficient_sam and not repo_dir.exists():
        !git clone https://github.com/IDEA-Research/Grounded-Segment-Anything

    # append to sys.path so that modules from the repo could be imported
    sys.path.append(str(ground_dino_dir))
    sys.path.append(str('EfficientSAM' if use_efficient_sam else repo_dir / 'segment_anything'))


.. parsed-literal::

    Cloning into 'GroundingDINO'...


.. parsed-literal::

    remote: Enumerating objects: 379, done.[K
    remote: Counting objects:   0% (1/177)[K
remote: Counting objects:   1% (2/177)[K
remote: Counting objects:   2% (4/177)[K
remote: Counting objects:   3% (6/177)[K
remote: Counting objects:   4% (8/177)[K
remote: Counting objects:   5% (9/177)[K
remote: Counting objects:   6% (11/177)[K
remote: Counting objects:   7% (13/177)[K
remote: Counting objects:   8% (15/177)[K
remote: Counting objects:   9% (16/177)[K
remote: Counting objects:  10% (18/177)[K
remote: Counting objects:  11% (20/177)[K
remote: Counting objects:  12% (22/177)[K
remote: Counting objects:  13% (24/177)[K
remote: Counting objects:  14% (25/177)[K
remote: Counting objects:  15% (27/177)[K
remote: Counting objects:  16% (29/177)[K
remote: Counting objects:  17% (31/177)[K
remote: Counting objects:  18% (32/177)[K
remote: Counting objects:  19% (34/177)[K
remote: Counting objects:  20% (36/177)[K
remote: Counting objects:  21% (38/177)[K
remote: Counting objects:  22% (39/177)[K
remote: Counting objects:  23% (41/177)[K
remote: Counting objects:  24% (43/177)[K
remote: Counting objects:  25% (45/177)[K
remote: Counting objects:  26% (47/177)[K
remote: Counting objects:  27% (48/177)[K
remote: Counting objects:  28% (50/177)[K
remote: Counting objects:  29% (52/177)[K
remote: Counting objects:  30% (54/177)[K
remote: Counting objects:  31% (55/177)[K
remote: Counting objects:  32% (57/177)[K
remote: Counting objects:  33% (59/177)[K
remote: Counting objects:  34% (61/177)[K
remote: Counting objects:  35% (62/177)[K
remote: Counting objects:  36% (64/177)[K
remote: Counting objects:  37% (66/177)[K
remote: Counting objects:  38% (68/177)[K
remote: Counting objects:  39% (70/177)[K
remote: Counting objects:  40% (71/177)[K
remote: Counting objects:  41% (73/177)[K
remote: Counting objects:  42% (75/177)[K
remote: Counting objects:  43% (77/177)[K
remote: Counting objects:  44% (78/177)[K
remote: Counting objects:  45% (80/177)[K
remote: Counting objects:  46% (82/177)[K
remote: Counting objects:  47% (84/177)[K
remote: Counting objects:  48% (85/177)[K
remote: Counting objects:  49% (87/177)[K
remote: Counting objects:  50% (89/177)[K
remote: Counting objects:  51% (91/177)[K
remote: Counting objects:  52% (93/177)[K
remote: Counting objects:  53% (94/177)[K
remote: Counting objects:  54% (96/177)[K
remote: Counting objects:  55% (98/177)[K
remote: Counting objects:  56% (100/177)[K
remote: Counting objects:  57% (101/177)[K
remote: Counting objects:  58% (103/177)[K
remote: Counting objects:  59% (105/177)[K
remote: Counting objects:  60% (107/177)[K
remote: Counting objects:  61% (108/177)[K
remote: Counting objects:  62% (110/177)[K
remote: Counting objects:  63% (112/177)[K
remote: Counting objects:  64% (114/177)[K
remote: Counting objects:  65% (116/177)[K
remote: Counting objects:  66% (117/177)[K
remote: Counting objects:  67% (119/177)[K
remote: Counting objects:  68% (121/177)[K
remote: Counting objects:  69% (123/177)[K
remote: Counting objects:  70% (124/177)[K
remote: Counting objects:  71% (126/177)[K
remote: Counting objects:  72% (128/177)[K
remote: Counting objects:  73% (130/177)[K
remote: Counting objects:  74% (131/177)[K
remote: Counting objects:  75% (133/177)[K
remote: Counting objects:  76% (135/177)[K
remote: Counting objects:  77% (137/177)[K
remote: Counting objects:  78% (139/177)[K
remote: Counting objects:  79% (140/177)[K
remote: Counting objects:  80% (142/177)[K
remote: Counting objects:  81% (144/177)[K
remote: Counting objects:  82% (146/177)[K
remote: Counting objects:  83% (147/177)[K
remote: Counting objects:  84% (149/177)[K
remote: Counting objects:  85% (151/177)[K
remote: Counting objects:  86% (153/177)[K
remote: Counting objects:  87% (154/177)[K
remote: Counting objects:  88% (156/177)[K
remote: Counting objects:  89% (158/177)[K
remote: Counting objects:  90% (160/177)[K
remote: Counting objects:  91% (162/177)[K
remote: Counting objects:  92% (163/177)[K
remote: Counting objects:  93% (165/177)[K
remote: Counting objects:  94% (167/177)[K
remote: Counting objects:  95% (169/177)[K
remote: Counting objects:  96% (170/177)[K
remote: Counting objects:  97% (172/177)[K
remote: Counting objects:  98% (174/177)[K
remote: Counting objects:  99% (176/177)[K
remote: Counting objects: 100% (177/177)[K
remote: Counting objects: 100% (177/177), done.[K
    remote: Compressing objects:   1% (1/64)[K
remote: Compressing objects:   3% (2/64)[K
remote: Compressing objects:   4% (3/64)[K
remote: Compressing objects:   6% (4/64)[K
remote: Compressing objects:   7% (5/64)[K
remote: Compressing objects:   9% (6/64)[K
remote: Compressing objects:  10% (7/64)[K
remote: Compressing objects:  12% (8/64)[K
remote: Compressing objects:  14% (9/64)[K
remote: Compressing objects:  15% (10/64)[K
remote: Compressing objects:  17% (11/64)[K
remote: Compressing objects:  18% (12/64)[K
remote: Compressing objects:  20% (13/64)[K
remote: Compressing objects:  21% (14/64)[K
remote: Compressing objects:  23% (15/64)[K
remote: Compressing objects:  25% (16/64)[K
remote: Compressing objects:  26% (17/64)[K
remote: Compressing objects:  28% (18/64)[K
remote: Compressing objects:  29% (19/64)[K
remote: Compressing objects:  31% (20/64)[K
remote: Compressing objects:  32% (21/64)[K
remote: Compressing objects:  34% (22/64)[K
remote: Compressing objects:  35% (23/64)[K
remote: Compressing objects:  37% (24/64)[K
remote: Compressing objects:  39% (25/64)[K
remote: Compressing objects:  40% (26/64)[K
remote: Compressing objects:  42% (27/64)[K
remote: Compressing objects:  43% (28/64)[K
remote: Compressing objects:  45% (29/64)[K
remote: Compressing objects:  46% (30/64)[K
remote: Compressing objects:  48% (31/64)[K
remote: Compressing objects:  50% (32/64)[K
remote: Compressing objects:  51% (33/64)[K
remote: Compressing objects:  53% (34/64)[K
remote: Compressing objects:  54% (35/64)[K
remote: Compressing objects:  56% (36/64)[K
remote: Compressing objects:  57% (37/64)[K
remote: Compressing objects:  59% (38/64)[K
remote: Compressing objects:  60% (39/64)[K
remote: Compressing objects:  62% (40/64)[K
remote: Compressing objects:  64% (41/64)[K
remote: Compressing objects:  65% (42/64)[K
remote: Compressing objects:  67% (43/64)[K
remote: Compressing objects:  68% (44/64)[K
remote: Compressing objects:  70% (45/64)[K
remote: Compressing objects:  71% (46/64)[K
remote: Compressing objects:  73% (47/64)[K
remote: Compressing objects:  75% (48/64)[K
remote: Compressing objects:  76% (49/64)[K
remote: Compressing objects:  78% (50/64)[K
remote: Compressing objects:  79% (51/64)[K
remote: Compressing objects:  81% (52/64)[K
remote: Compressing objects:  82% (53/64)[K
remote: Compressing objects:  84% (54/64)[K
remote: Compressing objects:  85% (55/64)[K
remote: Compressing objects:  87% (56/64)[K
remote: Compressing objects:  89% (57/64)[K
remote: Compressing objects:  90% (58/64)[K
remote: Compressing objects:  92% (59/64)[K
remote: Compressing objects:  93% (60/64)[K
remote: Compressing objects:  95% (61/64)[K
remote: Compressing objects:  96% (62/64)[K
remote: Compressing objects:  98% (63/64)[K
remote: Compressing objects: 100% (64/64)[K
remote: Compressing objects: 100% (64/64), done.[K
    Receiving objects:   0% (1/379)

.. parsed-literal::

    Receiving objects:   1% (4/379)

.. parsed-literal::

    Receiving objects:   2% (8/379)
Receiving objects:   3% (12/379)
Receiving objects:   4% (16/379)
Receiving objects:   5% (19/379)
Receiving objects:   6% (23/379)
Receiving objects:   7% (27/379)
Receiving objects:   8% (31/379)
Receiving objects:   9% (35/379)

.. parsed-literal::

    Receiving objects:  10% (38/379)
Receiving objects:  11% (42/379)
Receiving objects:  12% (46/379)
Receiving objects:  13% (50/379)
Receiving objects:  14% (54/379)
Receiving objects:  15% (57/379)
Receiving objects:  16% (61/379)
Receiving objects:  17% (65/379)
Receiving objects:  18% (69/379)
Receiving objects:  19% (73/379)
Receiving objects:  20% (76/379)
Receiving objects:  21% (80/379)
Receiving objects:  22% (84/379)
Receiving objects:  23% (88/379)
Receiving objects:  24% (91/379)
Receiving objects:  25% (95/379)

.. parsed-literal::

    Receiving objects:  26% (99/379)

.. parsed-literal::

    Receiving objects:  27% (103/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  28% (107/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  29% (110/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  30% (114/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  31% (118/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  32% (122/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  33% (126/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  34% (129/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  35% (133/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  36% (137/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  37% (141/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  38% (145/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  39% (148/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  40% (152/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  41% (156/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  42% (160/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  43% (163/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  44% (167/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  45% (171/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  46% (175/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  47% (179/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  48% (182/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  49% (186/379), 11.64 MiB | 21.59 MiB/s

.. parsed-literal::

    Receiving objects:  50% (190/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  51% (194/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  52% (198/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  53% (201/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  54% (205/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  55% (209/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  56% (213/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  57% (217/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  58% (220/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  59% (224/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  60% (228/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  61% (232/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  62% (235/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  63% (239/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  64% (243/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  65% (247/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  66% (251/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  67% (254/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  68% (258/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  69% (262/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  70% (266/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  71% (270/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  72% (273/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  73% (277/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  74% (281/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  75% (285/379), 11.64 MiB | 21.59 MiB/s
remote: Total 379 (delta 137), reused 113 (delta 113), pack-reused 202[K
    Receiving objects:  76% (289/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  77% (292/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  78% (296/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  79% (300/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  80% (304/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  81% (307/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  82% (311/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  83% (315/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  84% (319/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  85% (323/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  86% (326/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  87% (330/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  88% (334/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  89% (338/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  90% (342/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  91% (345/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  92% (349/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  93% (353/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  94% (357/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  95% (361/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  96% (364/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  97% (368/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  98% (372/379), 11.64 MiB | 21.59 MiB/s
Receiving objects:  99% (376/379), 11.64 MiB | 21.59 MiB/s
Receiving objects: 100% (379/379), 11.64 MiB | 21.59 MiB/s
Receiving objects: 100% (379/379), 14.03 MiB | 21.38 MiB/s, done.
    Resolving deltas:   0% (0/195)
Resolving deltas:   1% (2/195)
Resolving deltas:   3% (7/195)
Resolving deltas:   7% (14/195)
Resolving deltas:   8% (16/195)
Resolving deltas:  16% (32/195)
Resolving deltas:  17% (35/195)
Resolving deltas:  18% (37/195)
Resolving deltas:  19% (38/195)
Resolving deltas:  20% (40/195)
Resolving deltas:  21% (41/195)
Resolving deltas:  22% (43/195)
Resolving deltas:  27% (53/195)
Resolving deltas:  42% (83/195)
Resolving deltas:  44% (87/195)
Resolving deltas:  47% (93/195)
Resolving deltas:  55% (108/195)
Resolving deltas:  56% (111/195)
Resolving deltas:  57% (112/195)
Resolving deltas:  58% (114/195)
Resolving deltas:  60% (117/195)
Resolving deltas:  61% (119/195)
Resolving deltas:  63% (123/195)
Resolving deltas:  65% (127/195)
Resolving deltas:  69% (135/195)
Resolving deltas:  70% (138/195)
Resolving deltas:  71% (139/195)
Resolving deltas:  72% (142/195)
Resolving deltas:  75% (148/195)
Resolving deltas:  76% (149/195)
Resolving deltas:  78% (153/195)
Resolving deltas:  80% (157/195)
Resolving deltas:  82% (160/195)
Resolving deltas: 100% (195/195)
Resolving deltas: 100% (195/195), done.


.. parsed-literal::

    Cloning into 'EfficientSAM'...


.. parsed-literal::

    remote: Enumerating objects: 424, done.[K
    remote: Counting objects:   0% (1/140)[K
remote: Counting objects:   1% (2/140)[K
remote: Counting objects:   2% (3/140)[K
remote: Counting objects:   3% (5/140)[K
remote: Counting objects:   4% (6/140)[K
remote: Counting objects:   5% (7/140)[K
remote: Counting objects:   6% (9/140)[K
remote: Counting objects:   7% (10/140)[K
remote: Counting objects:   8% (12/140)[K
remote: Counting objects:   9% (13/140)[K
remote: Counting objects:  10% (14/140)[K
remote: Counting objects:  11% (16/140)[K
remote: Counting objects:  12% (17/140)[K
remote: Counting objects:  13% (19/140)[K
remote: Counting objects:  14% (20/140)[K
remote: Counting objects:  15% (21/140)[K
remote: Counting objects:  16% (23/140)[K
remote: Counting objects:  17% (24/140)[K
remote: Counting objects:  18% (26/140)[K
remote: Counting objects:  19% (27/140)[K
remote: Counting objects:  20% (28/140)[K
remote: Counting objects:  21% (30/140)[K
remote: Counting objects:  22% (31/140)[K
remote: Counting objects:  23% (33/140)[K
remote: Counting objects:  24% (34/140)[K
remote: Counting objects:  25% (35/140)[K
remote: Counting objects:  26% (37/140)[K
remote: Counting objects:  27% (38/140)[K
remote: Counting objects:  28% (40/140)[K
remote: Counting objects:  29% (41/140)[K
remote: Counting objects:  30% (42/140)[K
remote: Counting objects:  31% (44/140)[K
remote: Counting objects:  32% (45/140)[K
remote: Counting objects:  33% (47/140)[K
remote: Counting objects:  34% (48/140)[K
remote: Counting objects:  35% (49/140)[K
remote: Counting objects:  36% (51/140)[K
remote: Counting objects:  37% (52/140)[K
remote: Counting objects:  38% (54/140)[K
remote: Counting objects:  39% (55/140)[K
remote: Counting objects:  40% (56/140)[K
remote: Counting objects:  41% (58/140)[K
remote: Counting objects:  42% (59/140)[K
remote: Counting objects:  43% (61/140)[K
remote: Counting objects:  44% (62/140)[K
remote: Counting objects:  45% (63/140)[K
remote: Counting objects:  46% (65/140)[K
remote: Counting objects:  47% (66/140)[K
remote: Counting objects:  48% (68/140)[K
remote: Counting objects:  49% (69/140)[K
remote: Counting objects:  50% (70/140)[K
remote: Counting objects:  51% (72/140)[K
remote: Counting objects:  52% (73/140)[K
remote: Counting objects:  53% (75/140)[K
remote: Counting objects:  54% (76/140)[K
remote: Counting objects:  55% (77/140)[K
remote: Counting objects:  56% (79/140)[K
remote: Counting objects:  57% (80/140)[K
remote: Counting objects:  58% (82/140)[K
remote: Counting objects:  59% (83/140)[K
remote: Counting objects:  60% (84/140)[K
remote: Counting objects:  61% (86/140)[K
remote: Counting objects:  62% (87/140)[K
remote: Counting objects:  63% (89/140)[K
remote: Counting objects:  64% (90/140)[K
remote: Counting objects:  65% (91/140)[K
remote: Counting objects:  66% (93/140)[K
remote: Counting objects:  67% (94/140)[K
remote: Counting objects:  68% (96/140)[K
remote: Counting objects:  69% (97/140)[K
remote: Counting objects:  70% (98/140)[K
remote: Counting objects:  71% (100/140)[K
remote: Counting objects:  72% (101/140)[K
remote: Counting objects:  73% (103/140)[K
remote: Counting objects:  74% (104/140)[K
remote: Counting objects:  75% (105/140)[K
remote: Counting objects:  76% (107/140)[K
remote: Counting objects:  77% (108/140)[K
remote: Counting objects:  78% (110/140)[K
remote: Counting objects:  79% (111/140)[K
remote: Counting objects:  80% (112/140)[K
remote: Counting objects:  81% (114/140)[K
remote: Counting objects:  82% (115/140)[K
remote: Counting objects:  83% (117/140)[K
remote: Counting objects:  84% (118/140)[K
remote: Counting objects:  85% (119/140)[K
remote: Counting objects:  86% (121/140)[K
remote: Counting objects:  87% (122/140)[K

.. parsed-literal::

    remote: Counting objects:  88% (124/140)[K
remote: Counting objects:  89% (125/140)[K
remote: Counting objects:  90% (126/140)[K
remote: Counting objects:  91% (128/140)[K
remote: Counting objects:  92% (129/140)[K
remote: Counting objects:  93% (131/140)[K
remote: Counting objects:  94% (132/140)[K
remote: Counting objects:  95% (133/140)[K
remote: Counting objects:  96% (135/140)[K
remote: Counting objects:  97% (136/140)[K
remote: Counting objects:  98% (138/140)[K
remote: Counting objects:  99% (139/140)[K
remote: Counting objects: 100% (140/140)[K
remote: Counting objects: 100% (140/140), done.[K
    remote: Compressing objects:   1% (1/85)[K
remote: Compressing objects:   2% (2/85)[K
remote: Compressing objects:   3% (3/85)[K
remote: Compressing objects:   4% (4/85)[K
remote: Compressing objects:   5% (5/85)[K
remote: Compressing objects:   7% (6/85)[K
remote: Compressing objects:   8% (7/85)[K
remote: Compressing objects:   9% (8/85)[K
remote: Compressing objects:  10% (9/85)[K
remote: Compressing objects:  11% (10/85)[K

.. parsed-literal::

    remote: Compressing objects:  12% (11/85)[K
remote: Compressing objects:  14% (12/85)[K
remote: Compressing objects:  15% (13/85)[K
remote: Compressing objects:  16% (14/85)[K
remote: Compressing objects:  17% (15/85)[K
remote: Compressing objects:  18% (16/85)[K
remote: Compressing objects:  20% (17/85)[K
remote: Compressing objects:  21% (18/85)[K
remote: Compressing objects:  22% (19/85)[K
remote: Compressing objects:  23% (20/85)[K
remote: Compressing objects:  24% (21/85)[K
remote: Compressing objects:  25% (22/85)[K
remote: Compressing objects:  27% (23/85)[K
remote: Compressing objects:  28% (24/85)[K
remote: Compressing objects:  29% (25/85)[K
remote: Compressing objects:  30% (26/85)[K
remote: Compressing objects:  31% (27/85)[K
remote: Compressing objects:  32% (28/85)[K
remote: Compressing objects:  34% (29/85)[K
remote: Compressing objects:  35% (30/85)[K
remote: Compressing objects:  36% (31/85)[K
remote: Compressing objects:  37% (32/85)[K
remote: Compressing objects:  38% (33/85)[K
remote: Compressing objects:  40% (34/85)[K
remote: Compressing objects:  41% (35/85)[K
remote: Compressing objects:  42% (36/85)[K
remote: Compressing objects:  43% (37/85)[K
remote: Compressing objects:  44% (38/85)[K
remote: Compressing objects:  45% (39/85)[K
remote: Compressing objects:  47% (40/85)[K
remote: Compressing objects:  48% (41/85)[K
remote: Compressing objects:  49% (42/85)[K
remote: Compressing objects:  50% (43/85)[K
remote: Compressing objects:  51% (44/85)[K
remote: Compressing objects:  52% (45/85)[K
remote: Compressing objects:  54% (46/85)[K
remote: Compressing objects:  55% (47/85)[K
remote: Compressing objects:  56% (48/85)[K
remote: Compressing objects:  57% (49/85)[K
remote: Compressing objects:  58% (50/85)[K
remote: Compressing objects:  60% (51/85)[K
remote: Compressing objects:  61% (52/85)[K
remote: Compressing objects:  62% (53/85)[K
remote: Compressing objects:  63% (54/85)[K
remote: Compressing objects:  64% (55/85)[K
remote: Compressing objects:  65% (56/85)[K
remote: Compressing objects:  67% (57/85)[K
remote: Compressing objects:  68% (58/85)[K
remote: Compressing objects:  69% (59/85)[K
remote: Compressing objects:  70% (60/85)[K
remote: Compressing objects:  71% (61/85)[K
remote: Compressing objects:  72% (62/85)[K
remote: Compressing objects:  74% (63/85)[K
remote: Compressing objects:  75% (64/85)[K
remote: Compressing objects:  76% (65/85)[K
remote: Compressing objects:  77% (66/85)[K
remote: Compressing objects:  78% (67/85)[K
remote: Compressing objects:  80% (68/85)[K
remote: Compressing objects:  81% (69/85)[K
remote: Compressing objects:  82% (70/85)[K
remote: Compressing objects:  83% (71/85)[K
remote: Compressing objects:  84% (72/85)[K
remote: Compressing objects:  85% (73/85)[K
remote: Compressing objects:  87% (74/85)[K
remote: Compressing objects:  88% (75/85)[K
remote: Compressing objects:  89% (76/85)[K
remote: Compressing objects:  90% (77/85)[K
remote: Compressing objects:  91% (78/85)[K
remote: Compressing objects:  92% (79/85)[K
remote: Compressing objects:  94% (80/85)[K
remote: Compressing objects:  95% (81/85)[K
remote: Compressing objects:  96% (82/85)[K
remote: Compressing objects:  97% (83/85)[K
remote: Compressing objects:  98% (84/85)[K
remote: Compressing objects: 100% (85/85)[K
remote: Compressing objects: 100% (85/85), done.[K
    Receiving objects:   0% (1/424)

.. parsed-literal::

    Receiving objects:   1% (5/424)
Receiving objects:   2% (9/424)
Receiving objects:   3% (13/424)
Receiving objects:   4% (17/424)

.. parsed-literal::

    Receiving objects:   5% (22/424)
Receiving objects:   6% (26/424)

.. parsed-literal::

    Receiving objects:   6% (26/424), 20.47 MiB | 20.46 MiB/s

.. parsed-literal::

    Receiving objects:   6% (29/424), 42.52 MiB | 21.15 MiB/s

.. parsed-literal::

    Receiving objects:   6% (29/424), 64.88 MiB | 21.43 MiB/s

.. parsed-literal::

    Receiving objects:   7% (30/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:   8% (34/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:   9% (39/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  10% (43/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  11% (47/424), 72.89 MiB | 20.50 MiB/s

.. parsed-literal::

    Receiving objects:  12% (51/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  13% (56/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  14% (60/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  15% (64/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  16% (68/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  17% (73/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  18% (77/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  19% (81/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  20% (85/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  21% (90/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  22% (94/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  23% (98/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  24% (102/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  25% (106/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  26% (111/424), 72.89 MiB | 20.50 MiB/s
Receiving objects:  27% (115/424), 72.89 MiB | 20.50 MiB/s

.. parsed-literal::

    Receiving objects:  27% (115/424), 83.68 MiB | 20.63 MiB/s

.. parsed-literal::

    Receiving objects:  27% (115/424), 107.54 MiB | 21.41 MiB/s

.. parsed-literal::

    Receiving objects:  27% (115/424), 131.54 MiB | 21.79 MiB/s

.. parsed-literal::

    Receiving objects:  27% (115/424), 156.82 MiB | 22.49 MiB/s

.. parsed-literal::

    Receiving objects:  28% (119/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  29% (123/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  30% (128/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  31% (132/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  32% (136/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  33% (140/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  34% (145/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  35% (149/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  36% (153/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  37% (157/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  38% (162/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  39% (166/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  40% (170/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  41% (174/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  42% (179/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  43% (183/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  44% (187/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  45% (191/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  46% (196/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  47% (200/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  48% (204/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  49% (208/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  50% (212/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  51% (217/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  52% (221/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  53% (225/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  54% (229/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  55% (234/424), 169.11 MiB | 22.60 MiB/s
Receiving objects:  56% (238/424), 169.11 MiB | 22.60 MiB/s

.. parsed-literal::

    Receiving objects:  56% (240/424), 181.91 MiB | 23.78 MiB/s

.. parsed-literal::

    Receiving objects:  56% (241/424), 195.30 MiB | 24.35 MiB/s
Receiving objects:  57% (242/424), 195.30 MiB | 24.35 MiB/s
Receiving objects:  58% (246/424), 195.30 MiB | 24.35 MiB/s
Receiving objects:  59% (251/424), 195.30 MiB | 24.35 MiB/s
Receiving objects:  60% (255/424), 195.30 MiB | 24.35 MiB/s
Receiving objects:  61% (259/424), 195.30 MiB | 24.35 MiB/s

.. parsed-literal::

    Receiving objects:  62% (263/424), 209.77 MiB | 25.00 MiB/s

.. parsed-literal::

    Receiving objects:  63% (268/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  64% (272/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  65% (276/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  66% (280/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  67% (285/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  68% (289/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  69% (293/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  70% (297/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  71% (302/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  72% (306/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  73% (310/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  74% (314/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  75% (318/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  76% (323/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  77% (327/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  78% (331/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  79% (335/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  80% (340/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  81% (344/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  82% (348/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  83% (352/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  84% (357/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  85% (361/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  86% (365/424), 209.77 MiB | 25.00 MiB/s
Receiving objects:  87% (369/424), 209.77 MiB | 25.00 MiB/s

.. parsed-literal::

    Receiving objects:  87% (370/424), 239.11 MiB | 26.23 MiB/s

.. parsed-literal::

    Receiving objects:  87% (371/424), 269.59 MiB | 27.42 MiB/s

.. parsed-literal::

    Receiving objects:  87% (372/424), 283.95 MiB | 27.91 MiB/s

.. parsed-literal::

    Receiving objects:  88% (374/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  89% (378/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  90% (382/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  91% (386/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  92% (391/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  93% (395/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  94% (399/424), 283.95 MiB | 27.91 MiB/s
Receiving objects:  95% (403/424), 283.95 MiB | 27.91 MiB/s

.. parsed-literal::

    Receiving objects:  95% (407/424), 331.23 MiB | 29.97 MiB/s

.. parsed-literal::

    Receiving objects:  96% (408/424), 331.23 MiB | 29.97 MiB/s
Receiving objects:  97% (412/424), 331.23 MiB | 29.97 MiB/s
Receiving objects:  98% (416/424), 331.23 MiB | 29.97 MiB/s
Receiving objects:  99% (420/424), 331.23 MiB | 29.97 MiB/s
remote: Total 424 (delta 84), reused 99 (delta 55), pack-reused 284[K
    Receiving objects: 100% (424/424), 331.23 MiB | 29.97 MiB/s
Receiving objects: 100% (424/424), 334.57 MiB | 25.17 MiB/s, done.
    Resolving deltas:   0% (0/226)
Resolving deltas:   5% (12/226)
Resolving deltas:   7% (18/226)
Resolving deltas:   9% (22/226)

.. parsed-literal::

    Resolving deltas:  15% (35/226)
Resolving deltas:  17% (39/226)
Resolving deltas:  19% (44/226)
Resolving deltas:  23% (54/226)
Resolving deltas:  26% (59/226)

.. parsed-literal::

    Resolving deltas:  28% (65/226)
Resolving deltas:  35% (81/226)
Resolving deltas:  36% (83/226)
Resolving deltas:  39% (89/226)
Resolving deltas:  42% (95/226)
Resolving deltas:  46% (104/226)
Resolving deltas:  50% (114/226)
Resolving deltas:  51% (116/226)
Resolving deltas:  55% (125/226)
Resolving deltas:  58% (133/226)
Resolving deltas:  59% (135/226)
Resolving deltas:  60% (136/226)

.. parsed-literal::

    Resolving deltas:  61% (138/226)
Resolving deltas:  69% (157/226)
Resolving deltas:  82% (187/226)
Resolving deltas:  91% (207/226)
Resolving deltas:  92% (208/226)
Resolving deltas:  93% (212/226)
Resolving deltas:  94% (213/226)
Resolving deltas:  95% (215/226)
Resolving deltas:  96% (217/226)
Resolving deltas:  97% (220/226)

.. parsed-literal::

    Resolving deltas:  98% (223/226)
Resolving deltas:  99% (224/226)

.. parsed-literal::

    Resolving deltas: 100% (226/226)
Resolving deltas: 100% (226/226), done.


.. code:: ipython3

    import torch
    import numpy as np
    import supervision as sv
    import openvino as ov
    from PIL import Image, ImageDraw, ImageFont
    from typing import Union, List
    import transformers

    core = ov.Core()

Download checkpoints and load PyTorch models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    IRS_PATH = Path('openvino_irs')
    CKPT_BASE_PATH = Path('checkpoints')
    os.makedirs(IRS_PATH, exist_ok=True)
    os.makedirs(CKPT_BASE_PATH, exist_ok=True)

    PT_DEVICE = 'cpu'
    ov_dino_name = 'openvino_grounding_dino'
    ov_sam_name = 'openvino_segment_anything'

    ground_dino_img_size = (1024, 1280)

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = f"{ground_dino_dir}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = CKPT_BASE_PATH / "groundingdino_swint_ogc.pth"

    # Segment Anything checkpoint
    SAM_CHECKPOINT_PATH = CKPT_BASE_PATH / "sam_vit_h_4b8939.pth"

    # Efficient Segment Anything checkpoint
    EFFICIENT_SAM_CHECKPOINT_PATH = efficient_sam_dir / "weights/efficient_sam_vitt.pt"

.. code:: ipython3

    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    from notebook_utils import download_file

    download_file("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", directory=CKPT_BASE_PATH)
    if not use_efficient_sam:
        download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", directory=CKPT_BASE_PATH)



.. parsed-literal::

    checkpoints/groundingdino_swint_ogc.pth:   0%|          | 0.00/662M [00:00<?, ?B/s]


GroundingDINO imports

.. code:: ipython3

    from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import get_phrases_from_posmap
    from groundingdino.util.inference import Model


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:31: UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
      warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")


.. code:: ipython3

    def load_pt_grounding_dino(model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)

        # modified config
        args.device = PT_DEVICE
        args.use_checkpoint = False
        args.use_transformer_ckpt = False

        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=PT_DEVICE)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()

        return model, args.max_text_len, get_tokenlizer.get_tokenlizer(args.text_encoder_type)

.. code:: ipython3

    # Load GroundingDINO inference model
    pt_grounding_dino_model, max_text_len, dino_tokenizer = load_pt_grounding_dino(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3549.)
      return _VF.meshgrid(tensors, \*\*kwargs)  # type: ignore[attr-defined]


.. parsed-literal::

    final text_encoder_type: bert-base-uncased


.. parsed-literal::

    final text_encoder_type: bert-base-uncased


.. code:: ipython3

    # load SAM model: EfficientSAM or vanilla SAM

    if use_efficient_sam:
        from efficient_sam.efficient_sam import build_efficient_sam
        # Load EfficientSAM
        efficient_sam_model = build_efficient_sam(
            encoder_patch_embed_dim=192,
            encoder_num_heads=3,
            checkpoint=EFFICIENT_SAM_CHECKPOINT_PATH
        ).eval()
    else:
        from segment_anything import build_sam, SamPredictor
        # Load SAM Model and SAM Predictor
        sam = build_sam(checkpoint=SAM_CHECKPOINT_PATH).to(PT_DEVICE)
        sam_predictor = SamPredictor(sam)

Convert GroundingDINO to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    ov_dino_path = IRS_PATH / f'{ov_dino_name}.xml'

    if not ov_dino_path.exists():
        tokenized = pt_grounding_dino_model.tokenizer(["the running dog ."], return_tensors="pt")
        input_ids = tokenized['input_ids']
        token_type_ids = tokenized['token_type_ids']
        attention_mask = tokenized['attention_mask']
        position_ids = torch.arange(input_ids.shape[1]).reshape(1, -1)
        text_token_mask = torch.randint(0, 2, (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.bool)
        img = torch.randn(1, 3, *ground_dino_img_size)

        dummpy_inputs = img, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask

        # without disabling gradients trace error occurs: "Cannot insert a Tensor that requires grad as a constant"
        for par in pt_grounding_dino_model.parameters():
            par.requires_grad = False
        # If we don't trace manually ov.convert_model will try to trace it automatically with default check_trace=True, which fails.
        # Therefore we trace manually with check_trace=False, despite there are warnings after tracing and conversion to OpenVINO IR
        # output boxes are correct.
        traced_model = torch.jit.trace(pt_grounding_dino_model, example_inputs=dummpy_inputs, strict=False, check_trace=False)

        ov_dino_model = ov.convert_model(traced_model, example_input=dummpy_inputs)
        ov.save_model(ov_dino_model, ov_dino_path)
    else:
        ov_dino_model = core.read_model(ov_dino_path)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:962: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py:264: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if encoded_text.shape[1] > self.max_text_len:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/util/misc.py:506: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/util/misc.py:517: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for img in tensor_list:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:486: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if W % self.patch_size[1] != 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:488: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if H % self.patch_size[0] != 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:417: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      Hp = int(np.ceil(H / self.window_size)) * self.window_size
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:418: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      Wp = int(np.ceil(W / self.window_size)) * self.window_size
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:247: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert L == H * W, "input feature has wrong size"


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:71: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      B = int(windows.shape[0] / (H * W / window_size / window_size))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:289: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if pad_r > 0 or pad_b > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:321: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert L == H * W, "input feature has wrong size"
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:326: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      pad_input = (H % 2 == 1) or (W % 2 == 1)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py:327: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if pad_input:


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py:244: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      spatial_shapes = torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py:468: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for lvl, (H_, W_) in enumerate(spatial_shapes):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/fuse_modules.py:176: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/fuse_modules.py:227: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/fuse_modules.py:232: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/transformer_vanilla.py:109: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:287: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:309: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if reference_points.shape[-1] == 2:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:102: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:105: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for level, (H_, W_) in enumerate(value_spatial_shapes):


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/utils.py:72: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
      for lvl, (H_, W_) in enumerate(spatial_shapes):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py:667: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if reference_points.shape[-1] == 4:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/utils.py:216: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if pos_tensor.size(-1) == 2:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/utils.py:218: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif pos_tensor.size(-1) == 4:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:315: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif reference_points.shape[-1] == 4:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py:704: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output.isnan().any() | output.isinf().any():


Run OpenVINO optimized GroundingDINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    device_widget = ipywidgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
    )
    device_widget




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



In order to run inference ``ov_dino_model`` should be compiled.
Resulting ``ov.CompiledModel`` object receives the same arguments as
pytorch ``forward``/``__call__`` methods.

.. code:: ipython3

    device = device_widget.value
    ov_compiled_grounded_dino = core.compile_model(ov_dino_model, device)

We will reuse only tokenizer from the original GroundingDINO model
class, but the inference will be done using OpenVINO optimized model.

.. code:: ipython3

    def transform_image(pil_image: Image.Image) -> torch.Tensor:
        import groundingdino.datasets.transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(pil_image, None)  # 3, h, w
        return image

    # detects boxes usding openvino optimized grounding dino model
    def get_ov_grounding_output(
        model: ov.CompiledModel,
        pil_image: Image.Image,
        caption: Union[str, List[str]],
        box_threshold: float,
        text_threshold: float,
        dino_tokenizer: transformers.PreTrainedTokenizerBase = dino_tokenizer,
        max_text_len: int = max_text_len
    ) -> (torch.Tensor, List[str], torch.Tensor):
        #  for text prompt pre-processing we reuse existing routines from GroundignDINO repo
        if isinstance(caption, list):
            caption = '. '.join(caption)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions = [caption]

        tokenized = dino_tokenizer(captions, padding="longest", return_tensors="pt")
        specical_tokens = dino_tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, specical_tokens, dino_tokenizer)

        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : max_text_len, : max_text_len]

            position_ids = position_ids[:, : max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

        # inputs dictionary which will be fed into the ov.CompiledModel for inference
        inputs = {}
        inputs["attention_mask.1"] = tokenized["attention_mask"]
        inputs["text_self_attention_masks"] = text_self_attention_masks
        inputs["input_ids"] = tokenized["input_ids"]
        inputs["position_ids"] = position_ids
        inputs["token_type_ids"] = tokenized["token_type_ids"]

        # GroundingDINO fails to run with input shapes different than one used for conversion.
        # As a workaround we resize input_image to the size used for conversion. Model does not rely
        # on image resolution to know object sizes therefore no need to resize box_predictions
        from torchvision.transforms.functional import resize, InterpolationMode
        input_img = resize(transform_image(pil_image), ground_dino_img_size, interpolation=InterpolationMode.BICUBIC)[None, ...]
        inputs["samples"] = input_img

        # OpenVINO inference
        request = model.create_infer_request()
        request.start_async(inputs, share_inputs=False)
        request.wait()

        def sig(x):
            return 1 / (1 + np.exp(-x))

        logits = torch.from_numpy(sig(np.squeeze(request.get_tensor("pred_logits").data, 0)))
        boxes = torch.from_numpy(np.squeeze(request.get_tensor("pred_boxes").data, 0))

        # filter output
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits, boxes = logits[filt_mask], boxes[filt_mask]

        # get phrase and build predictions
        tokenized = dino_tokenizer(caption)
        pred_phrases = []
        for logit in logits:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, dino_tokenizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        return boxes, pred_phrases, logits.max(dim=1)[0]

.. code:: ipython3

    SOURCE_IMAGE_PATH = f"{ground_dino_dir}/.asset/demo7.jpg"
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    pil_image = Image.open(SOURCE_IMAGE_PATH)
    classes_prompt = ["Horse", "Cloud"]

.. code:: ipython3

    boxes_filt, pred_phrases, logits_filt = get_ov_grounding_output(
        ov_compiled_grounded_dino,
        pil_image,
        classes_prompt,
        BOX_THRESHOLD, TEXT_THRESHOLD
    )


.. parsed-literal::

    2024-03-13 00:58:26.223873: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-13 00:58:26.261830: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-13 00:58:26.819015: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Convert predicted boxes to supervision box detections format

.. code:: ipython3

    source_w, source_h = pil_image.size
    detections = Model.post_process_result(
        source_h=source_h,
        source_w=source_w,
        boxes=boxes_filt,
        logits=logits_filt)

    class_id = Model.phrases2classes(phrases=pred_phrases, classes=list(map(str.lower, classes_prompt)))
    detections.class_id = class_id

Draw box detections

.. code:: ipython3

    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes_prompt[class_id] if class_id is not None else 'None'} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=np.array(pil_image).copy(), detections=detections, labels=labels)

    Image.fromarray(annotated_frame)




.. image:: 288-grounded-segment-anything-with-output_files/288-grounded-segment-anything-with-output_29_0.png



Great! All clouds and horses are detected. Feel free to play around and
specify other objects you wish to detect.

Convert SAM to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~



And now let’s feed those detection to ``SAM`` model. We will use
``EfficiendSAM`` for faster computation and to save ram, but feel free
to select vanilla ``SAM`` if you wish more detailed and precise
segmentation. First of all let’s convert ``SAM`` model to OpenVINO IR.

.. code:: ipython3

    ov_efficient_sam_name = 'openvino_efficient_sam'
    ov_efficient_sam_path = IRS_PATH / f'{ov_efficient_sam_name}.xml'

    # convert EfficientSAM to OpenVINO IR format
    if not ov_efficient_sam_path.exists() and use_efficient_sam:
        random_input_image = np.random.rand(1, 3, *pil_image.size[::-1]).astype(np.float32)
        bounding_box = np.array([900, 100, 1000, 200]).reshape([1, 1, 2, 2])
        bbox_labels = np.array([2, 3]).reshape([1, 1, 2])
        efficient_sam_dummy_input = tuple(torch.from_numpy(x) for x in (random_input_image, bounding_box, bbox_labels))

        ov_efficient_sam = ov.convert_model(efficient_sam_model, example_input=efficient_sam_dummy_input)
        ov.save_model(ov_efficient_sam, ov_efficient_sam_path)
    elif use_efficient_sam:
        ov_efficient_sam = core.read_model(ov_efficient_sam_path)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam.py:220: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if (
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam_encoder.py:241: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert (
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam_encoder.py:163: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      size = int(math.sqrt(xy_num))
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam_encoder.py:164: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert size * size == xy_num
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam_encoder.py:166: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if size != h or size != w:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam_encoder.py:251: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert x.shape[2] == num_patches


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam.py:85: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if num_pts > self.decoder_max_num_input_points:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam.py:92: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif num_pts < self.decoder_max_num_input_points:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/288-grounded-segment-anything/EfficientSAM/efficient_sam/efficient_sam.py:126: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output_w > 0 and output_h > 0:


Below is conversion of vanilla ``SAM``. This code is not used when
``EfficientSAM`` is selected for segmentation.

.. code:: ipython3

    # In order to convert to OpenVINO IR neeed to patch forward method or the torch.nn.Module for SAM
    class SamMaskFromBoxes(torch.nn.Module):
        def __init__(
            self,
            sam_predictor,
        ) -> None:
            super().__init__()
            self.model = sam_predictor

        @torch.no_grad()
        def forward(
            self,
            input_image: torch.Tensor,
            transformed_boxes: torch.Tensor,
            multimask_output: bool = False,
            hq_token_only: bool = False,
        ):
            pre_processed_image = self.model.model.preprocess(input_image)
            image_embeddings, interm_features = self.model.model.image_encoder(pre_processed_image)

            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.model.prompt_encoder(
                points=None,
                boxes=transformed_boxes,
                masks=None,
            )

            # Predict masks
            low_res_masks, iou_predictions = self.model.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                hq_token_only=hq_token_only,
                interm_embeddings=interm_features,
            )

            return low_res_masks, iou_predictions

.. code:: ipython3

    ov_sam_path = IRS_PATH / f'{ov_sam_name}.xml'

    # example input for vanilla SAM
    input_image_torch = torch.randint(0, 255, size=[1, 3, 683, 1024], dtype=torch.uint8)
    dummy_transformed_boxes = torch.rand(1, 4, dtype=torch.float32) * 200

    # convert vanilla SAM to OpenVINO IR format
    if not ov_sam_path.exists() and not use_efficient_sam:
        # Load pytorch model object and prepare example input for conversion
        exportable = SamMaskFromBoxes(sam_predictor)
        exportable.model.model.eval()
        for par in exportable.model.model.parameters():
            par.requires_grad = False

        traced = torch.jit.trace(exportable, example_inputs=(input_image_torch, dummy_transformed_boxes))
        ov_sam = ov.convert_model(traced, example_input=(input_image_torch, dummy_transformed_boxes))
        ov.save_model(ov_sam, ov_sam_path)
    elif not use_efficient_sam:
        ov_sam = core.read_model(ov_sam_path)

.. code:: ipython3

    if use_efficient_sam:
        compiled_efficient_sam = core.compile_model(ov_efficient_sam, device_name=device)
    else:
        compiled_vanilla_sam = core.compile_model(ov_sam, device_name=device)

Combine GroundingDINO + SAM (GroundedSAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



We have OpenVINO IRs for both GroundingDINO and SAM models. Lets run the
segmentation using predictions from GroundingDINO. Same as above, use
``EfficientSAM`` by default.

.. code:: ipython3

    def predict_efficient_sam_mask(compiled_efficient_sam: ov.CompiledModel, image: Image.Image, bbox: torch.Tensor):
        # input image is scaled so that none of the sizes is greater than 1024, same as in 274-efficient-sam notebook
        input_size = 1024
        w, h = image.size[:2]
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))

        numpy_image = np.array(image, dtype=np.float32) / 255.0
        numpy_image = np.transpose(numpy_image, (2, 0, 1))[None, ...]

        scaled_points = bbox * scale

        bounding_box = scaled_points.reshape([1, 1, 2, 2])
        bbox_labels = np.reshape(np.array([2, 3]), [1, 1, 2])

        res = compiled_efficient_sam((numpy_image, bounding_box, bbox_labels))

        predicted_logits, predicted_iou = res[0], res[1]

        all_masks = torch.ge(torch.sigmoid(torch.from_numpy(predicted_logits[0, 0, :, :, :])), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...]

        # select the mask with the greatest IOU
        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
            ):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou

    # If several detections are fed to EfficientSAM, it merges them to a single mask. Therefore, we call it one by one for each detection.
    def predict_efficient_sam_masks(compiled_efficient_sam: ov.CompiledModel, pil_image: Image.Image, transformed_boxes) -> torch.Tensor:
        masks = []
        for bbox in transformed_boxes:
            mask = predict_efficient_sam_mask(compiled_efficient_sam, pil_image, bbox)
            mask = Image.fromarray(mask).resize(pil_image.size)
            masks.append(np.array(mask))
        masks = torch.from_numpy(np.array(masks))
        return masks

.. code:: ipython3

    def transform_boxes(sam_predictor: torch.nn.Module, boxes: torch.Tensor, size: tuple) -> torch.Tensor:
        H, W = size[0], size[1]
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]

        return sam_predictor.transform.apply_boxes_torch(boxes, size).to(PT_DEVICE)

    def predict_vanilla_sam_masks(compiled_vanilla_sam: ov.CompiledModel, image: np.ndarray, transformed_boxes: torch.Tensor) -> torch.Tensor:
        transfromed_image = exportable.model.transform.apply_image(image)
        input_image_torch = torch.as_tensor(transfromed_image, device=PT_DEVICE)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_size = tuple(image.shape[:2])
        input_size = tuple(input_image_torch.shape[-2:])

        low_res_masks = compiled_vanilla_sam((input_image_torch, transformed_boxes))[0]

        # Upscale the masks to the original image resolution
        masks = exportable.model.model.postprocess_masks(torch.from_numpy(low_res_masks), input_size, original_size)
        masks = masks > exportable.model.model.mask_threshold
        return masks

Run SAM model for the same image with the detected boxes from
GroundingDINO.

Please note that vanilla SAM and EfficientSAM have slightly different
detection formats. But inputs for both of them originate from
``boxes_filt`` which is result of the ``get_ov_grounding_output``. For
EfficientSAM we use ``detections.xyxy`` boxes obtained after
``boxes_filt`` is fed to ``Model.post_process_result``. While vanilla
SAM has it’s own preprocessing function ``transform_boxes``.

.. code:: ipython3

    if use_efficient_sam:
        masks = predict_efficient_sam_masks(compiled_efficient_sam, pil_image, detections.xyxy)
        detections.mask = masks.numpy()
    else:
        transformed_boxes = transform_boxes(sam_predictor, boxes_filt, pil_image.size[::-1])
        masks = predict_vanilla_sam_masks(compiled_vanilla_sam, np.array(pil_image), transformed_boxes)
        detections.mask = masks[:, 0].numpy()

Combine both boxes and segmentation masks and draw them.

.. code:: ipython3

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated_image = np.array(pil_image)
    annotated_image = mask_annotator.annotate(scene=np.array(pil_image).copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    Image.fromarray(annotated_image)




.. image:: 288-grounded-segment-anything-with-output_files/288-grounded-segment-anything-with-output_45_0.png



Great! All detected horses and clouds are segmented as well.

Interactive GroundedSAM
~~~~~~~~~~~~~~~~~~~~~~~



Now, you can try apply grounding sam on your own images using
interactive demo. The code below provides helper functions used in
demonstration.

.. code:: ipython3

    def draw_mask(mask, draw, random_color=False):
        import random
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
        else:
            color = (30, 144, 255, 153)

        nonzero_coords = np.transpose(np.nonzero(mask))

        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)

    def draw_box(box, draw, label):
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=4)

        if label:
            font = ImageFont.load_default(18)
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((box[0], box[1]), str(label), font, anchor='ld')
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (box[0], box[1], box[0] + w, box[1] + h)
            draw.rectangle(bbox, fill=color)
            draw.text((box[0], box[1]), str(label), fill="white", anchor='ld', font=font)

.. code:: ipython3

    """"
    run_grounding_sam is called every time "Submit" button is clicked
    """
    def run_grounding_sam(image, task_type, text_prompt, box_threshold, text_threshold):
        pil_image = Image.fromarray(image)
        size = image.shape[1], image.shape[0]  # size is WH image.shape HWC

        boxes_filt, scores, pred_phrases = get_ov_grounding_output(
            ov_compiled_grounded_dino,
            pil_image,
            text_prompt,
            box_threshold,
            text_threshold
        )

        # process boxes
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        if task_type == 'seg':
            if use_efficient_sam:
                masks = predict_efficient_sam_masks(compiled_efficient_sam, pil_image, boxes_filt.numpy())
            else:
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(PT_DEVICE)
                masks = predict_vanilla_sam_masks(compiled_vanilla_sam, image, transformed_boxes)[:, 0]

            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            for mask in masks:
                draw_mask(mask.numpy(), mask_draw, random_color=True)

            image_draw = ImageDraw.Draw(pil_image)
            for box, label in zip(boxes_filt, pred_phrases):
                draw_box(box, image_draw, label)

            pil_image = pil_image.convert('RGBA')
            pil_image.alpha_composite(mask_image)

            return [pil_image, mask_image]
        if task_type == 'det':
            image_draw = ImageDraw.Draw(pil_image)
            for box, label in zip(boxes_filt, pred_phrases):
                draw_box(box, image_draw, label)
            return [pil_image]
        else:
            gr.Warning(f"task_type:{task_type} error!")

You can run interactive app with your own image and text prompts. To
define prompt specify comma (or conjunction) separated names of objects
you wish to segment. For demonstration, this demo already has two
predefined examples. If many object are crowded and overlapping please
increase threshold values in ``Advanced options``.

.. code:: ipython3

    import gradio as gr

    with gr.Accordion("Advanced options", open=False) as advanced:
        box_threshold = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
        text_threshold = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05)

    demo = gr.Interface(
        run_grounding_sam,
        [
            gr.Image(),
            gr.Dropdown(["det", "seg"], value="seg", label="task_type"),
            gr.Textbox(value='bears', label="Text Prompt"),
        ],
        additional_inputs=[
            box_threshold,
            text_threshold,
        ],
        outputs=gr.Gallery(preview=True, object_fit="scale-down"),
        examples=[[f"{ground_dino_dir}/.asset/demo2.jpg", "seg", 'dog, forest'], [f"{ground_dino_dir}/.asset/demo7.jpg", "seg", 'horses and clouds']],
        additional_inputs_accordion=advanced,
    )

    try:
        demo.launch(server_name='0.0.0.0', debug=False, height=1000)
    except Exception:
        demo.launch(share=True, debug=False, height=1000)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://0.0.0.0:7860

    To create a public link, set `share=True` in `launch()`.








Cleanup
~~~~~~~



.. code:: ipython3

    # import shutil
    # shutil.rmtree(CKPT_BASE_PATH)
    # shutil.rmtree(IRS_PATH)
