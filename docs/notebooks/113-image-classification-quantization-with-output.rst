Quantization of Image Classification Models
===========================================

This tutorial demonstrates how to apply ``INT8`` quantization to Image
Classification model using
`NNCF <https://github.com/openvinotoolkit/nncf>`__. It uses the
MobileNet V2 model, trained on Cifar10 dataset. The code is designed to
be extendable to custom models and datasets. The tutorial uses OpenVINO
backend for performing model quantization in NNCF, if you interested how
to apply quantization on PyTorch model, please check this
`tutorial <112-pytorch-post-training-quantization-nncf-with-output.html>`__.

This tutorial consists of the following steps:

-  Prepare the model for quantization.
-  Define a data loading functionality.
-  Perform quantization.
-  Compare accuracy of the original and quantized models.
-  Compare performance of the original and quantized models.
-  Compare results on one picture.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prepare the Model <#Prepare-the-Model>`__
-  `Prepare Dataset <#Prepare-Dataset>`__
-  `Perform Quantization <#Perform-Quantization>`__

   -  `Create Dataset for Validation <#Create-Dataset-for-Validation>`__

-  `Run nncf.quantize for Getting an Optimized
   Model <#Run-nncf.quantize-for-Getting-an-Optimized-Model>`__
-  `Serialize an OpenVINO IR model <#Serialize-an-OpenVINO-IR-model>`__
-  `Compare Accuracy of the Original and Quantized
   Models <#Compare-Accuracy-of-the-Original-and-Quantized-Models>`__

   -  `Select inference device <#Select-inference-device>`__

-  `Compare Performance of the Original and Quantized
   Models <#Compare-Performance-of-the-Original-and-Quantized-Models>`__
-  `Compare results on four
   pictures <#Compare-results-on-four-pictures>`__

.. code:: ipython3

    # Install openvino package
    %pip install -q "openvino>=2023.1.0" "nncf>=2.6.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    
    # Set the data and model directories
    DATA_DIR = Path("data")
    MODEL_DIR = Path('model')
    model_repo = 'pytorch-cifar-models'
    
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

Prepare the Model
-----------------

`back to top ⬆️ <#Table-of-contents:>`__

Model preparation stage has the following steps:

-  Download a PyTorch model
-  Convert model to OpenVINO Intermediate Representation format (IR)
   using model conversion Python API
-  Serialize converted model on disk

.. code:: ipython3

    import sys
    
    if not Path(model_repo).exists():
        !git clone https://github.com/chenyaofo/pytorch-cifar-models.git
    
    sys.path.append(model_repo)


.. parsed-literal::

    Cloning into 'pytorch-cifar-models'...


.. parsed-literal::

    remote: Enumerating objects: 282, done.[K
    remote: Counting objects:   0% (1/281)[Kremote: Counting objects:   1% (3/281)[Kremote: Counting objects:   2% (6/281)[Kremote: Counting objects:   3% (9/281)[Kremote: Counting objects:   4% (12/281)[Kremote: Counting objects:   5% (15/281)[Kremote: Counting objects:   6% (17/281)[Kremote: Counting objects:   7% (20/281)[Kremote: Counting objects:   8% (23/281)[Kremote: Counting objects:   9% (26/281)[Kremote: Counting objects:  10% (29/281)[Kremote: Counting objects:  11% (31/281)[Kremote: Counting objects:  12% (34/281)[Kremote: Counting objects:  13% (37/281)[Kremote: Counting objects:  14% (40/281)[Kremote: Counting objects:  15% (43/281)[Kremote: Counting objects:  16% (45/281)[Kremote: Counting objects:  17% (48/281)[Kremote: Counting objects:  18% (51/281)[Kremote: Counting objects:  19% (54/281)[Kremote: Counting objects:  20% (57/281)[Kremote: Counting objects:  21% (60/281)[Kremote: Counting objects:  22% (62/281)[Kremote: Counting objects:  23% (65/281)[Kremote: Counting objects:  24% (68/281)[Kremote: Counting objects:  25% (71/281)[Kremote: Counting objects:  26% (74/281)[Kremote: Counting objects:  27% (76/281)[Kremote: Counting objects:  28% (79/281)[Kremote: Counting objects:  29% (82/281)[Kremote: Counting objects:  30% (85/281)[Kremote: Counting objects:  31% (88/281)[Kremote: Counting objects:  32% (90/281)[Kremote: Counting objects:  33% (93/281)[Kremote: Counting objects:  34% (96/281)[Kremote: Counting objects:  35% (99/281)[Kremote: Counting objects:  36% (102/281)[Kremote: Counting objects:  37% (104/281)[Kremote: Counting objects:  38% (107/281)[Kremote: Counting objects:  39% (110/281)[Kremote: Counting objects:  40% (113/281)[Kremote: Counting objects:  41% (116/281)[Kremote: Counting objects:  42% (119/281)[Kremote: Counting objects:  43% (121/281)[Kremote: Counting objects:  44% (124/281)[Kremote: Counting objects:  45% (127/281)[Kremote: Counting objects:  46% (130/281)[Kremote: Counting objects:  47% (133/281)[Kremote: Counting objects:  48% (135/281)[Kremote: Counting objects:  49% (138/281)[Kremote: Counting objects:  50% (141/281)[Kremote: Counting objects:  51% (144/281)[Kremote: Counting objects:  52% (147/281)[Kremote: Counting objects:  53% (149/281)[Kremote: Counting objects:  54% (152/281)[Kremote: Counting objects:  55% (155/281)[Kremote: Counting objects:  56% (158/281)[Kremote: Counting objects:  57% (161/281)[Kremote: Counting objects:  58% (163/281)[Kremote: Counting objects:  59% (166/281)[Kremote: Counting objects:  60% (169/281)[Kremote: Counting objects:  61% (172/281)[Kremote: Counting objects:  62% (175/281)[Kremote: Counting objects:  63% (178/281)[Kremote: Counting objects:  64% (180/281)[Kremote: Counting objects:  65% (183/281)[Kremote: Counting objects:  66% (186/281)[Kremote: Counting objects:  67% (189/281)[Kremote: Counting objects:  68% (192/281)[Kremote: Counting objects:  69% (194/281)[Kremote: Counting objects:  70% (197/281)[Kremote: Counting objects:  71% (200/281)[Kremote: Counting objects:  72% (203/281)[Kremote: Counting objects:  73% (206/281)[Kremote: Counting objects:  74% (208/281)[Kremote: Counting objects:  75% (211/281)[Kremote: Counting objects:  76% (214/281)[Kremote: Counting objects:  77% (217/281)[Kremote: Counting objects:  78% (220/281)[Kremote: Counting objects:  79% (222/281)[Kremote: Counting objects:  80% (225/281)[Kremote: Counting objects:  81% (228/281)[Kremote: Counting objects:  82% (231/281)[Kremote: Counting objects:  83% (234/281)[Kremote: Counting objects:  84% (237/281)[Kremote: Counting objects:  85% (239/281)[Kremote: Counting objects:  86% (242/281)[Kremote: Counting objects:  87% (245/281)[Kremote: Counting objects:  88% (248/281)[Kremote: Counting objects:  89% (251/281)[Kremote: Counting objects:  90% (253/281)[Kremote: Counting objects:  91% (256/281)[Kremote: Counting objects:  92% (259/281)[Kremote: Counting objects:  93% (262/281)[Kremote: Counting objects:  94% (265/281)[Kremote: Counting objects:  95% (267/281)[Kremote: Counting objects:  96% (270/281)[Kremote: Counting objects:  97% (273/281)[Kremote: Counting objects:  98% (276/281)[Kremote: Counting objects:  99% (279/281)[Kremote: Counting objects: 100% (281/281)[Kremote: Counting objects: 100% (281/281), done.[K
    remote: Compressing objects:   1% (1/96)[Kremote: Compressing objects:   2% (2/96)[Kremote: Compressing objects:   3% (3/96)[Kremote: Compressing objects:   4% (4/96)[Kremote: Compressing objects:   5% (5/96)[Kremote: Compressing objects:   6% (6/96)[Kremote: Compressing objects:   7% (7/96)[Kremote: Compressing objects:   8% (8/96)[Kremote: Compressing objects:   9% (9/96)[Kremote: Compressing objects:  10% (10/96)[Kremote: Compressing objects:  11% (11/96)[Kremote: Compressing objects:  12% (12/96)[Kremote: Compressing objects:  13% (13/96)[Kremote: Compressing objects:  14% (14/96)[Kremote: Compressing objects:  15% (15/96)[Kremote: Compressing objects:  16% (16/96)[Kremote: Compressing objects:  17% (17/96)[Kremote: Compressing objects:  18% (18/96)[Kremote: Compressing objects:  19% (19/96)[Kremote: Compressing objects:  20% (20/96)[Kremote: Compressing objects:  21% (21/96)[Kremote: Compressing objects:  22% (22/96)[Kremote: Compressing objects:  23% (23/96)[Kremote: Compressing objects:  25% (24/96)[Kremote: Compressing objects:  26% (25/96)[Kremote: Compressing objects:  27% (26/96)[Kremote: Compressing objects:  28% (27/96)[Kremote: Compressing objects:  29% (28/96)[Kremote: Compressing objects:  30% (29/96)[Kremote: Compressing objects:  31% (30/96)[Kremote: Compressing objects:  32% (31/96)[Kremote: Compressing objects:  33% (32/96)[Kremote: Compressing objects:  34% (33/96)[Kremote: Compressing objects:  35% (34/96)[Kremote: Compressing objects:  36% (35/96)[Kremote: Compressing objects:  37% (36/96)[Kremote: Compressing objects:  38% (37/96)[Kremote: Compressing objects:  39% (38/96)[Kremote: Compressing objects:  40% (39/96)[Kremote: Compressing objects:  41% (40/96)[Kremote: Compressing objects:  42% (41/96)[Kremote: Compressing objects:  43% (42/96)[Kremote: Compressing objects:  44% (43/96)[Kremote: Compressing objects:  45% (44/96)[Kremote: Compressing objects:  46% (45/96)[Kremote: Compressing objects:  47% (46/96)[Kremote: Compressing objects:  48% (47/96)[Kremote: Compressing objects:  50% (48/96)[Kremote: Compressing objects:  51% (49/96)[Kremote: Compressing objects:  52% (50/96)[Kremote: Compressing objects:  53% (51/96)[Kremote: Compressing objects:  54% (52/96)[Kremote: Compressing objects:  55% (53/96)[Kremote: Compressing objects:  56% (54/96)[Kremote: Compressing objects:  57% (55/96)[Kremote: Compressing objects:  58% (56/96)[Kremote: Compressing objects:  59% (57/96)[Kremote: Compressing objects:  60% (58/96)[Kremote: Compressing objects:  61% (59/96)[Kremote: Compressing objects:  62% (60/96)[Kremote: Compressing objects:  63% (61/96)[Kremote: Compressing objects:  64% (62/96)[Kremote: Compressing objects:  65% (63/96)[Kremote: Compressing objects:  66% (64/96)[Kremote: Compressing objects:  67% (65/96)[Kremote: Compressing objects:  68% (66/96)[Kremote: Compressing objects:  69% (67/96)[Kremote: Compressing objects:  70% (68/96)[Kremote: Compressing objects:  71% (69/96)[Kremote: Compressing objects:  72% (70/96)[Kremote: Compressing objects:  73% (71/96)[Kremote: Compressing objects:  75% (72/96)[Kremote: Compressing objects:  76% (73/96)[Kremote: Compressing objects:  77% (74/96)[Kremote: Compressing objects:  78% (75/96)[Kremote: Compressing objects:  79% (76/96)[Kremote: Compressing objects:  80% (77/96)[Kremote: Compressing objects:  81% (78/96)[Kremote: Compressing objects:  82% (79/96)[Kremote: Compressing objects:  83% (80/96)[Kremote: Compressing objects:  84% (81/96)[Kremote: Compressing objects:  85% (82/96)[Kremote: Compressing objects:  86% (83/96)[Kremote: Compressing objects:  87% (84/96)[Kremote: Compressing objects:  88% (85/96)[Kremote: Compressing objects:  89% (86/96)[Kremote: Compressing objects:  90% (87/96)[Kremote: Compressing objects:  91% (88/96)[Kremote: Compressing objects:  92% (89/96)[Kremote: Compressing objects:  93% (90/96)[Kremote: Compressing objects:  94% (91/96)[Kremote: Compressing objects:  95% (92/96)[Kremote: Compressing objects:  96% (93/96)[Kremote: Compressing objects:  97% (94/96)[Kremote: Compressing objects:  98% (95/96)[Kremote: Compressing objects: 100% (96/96)[Kremote: Compressing objects: 100% (96/96), done.[K
    Receiving objects:   0% (1/282)Receiving objects:   1% (3/282)Receiving objects:   2% (6/282)Receiving objects:   3% (9/282)Receiving objects:   4% (12/282)Receiving objects:   5% (15/282)Receiving objects:   6% (17/282)Receiving objects:   7% (20/282)Receiving objects:   8% (23/282)Receiving objects:   9% (26/282)Receiving objects:  10% (29/282)Receiving objects:  11% (32/282)Receiving objects:  12% (34/282)Receiving objects:  13% (37/282)Receiving objects:  14% (40/282)Receiving objects:  15% (43/282)Receiving objects:  16% (46/282)Receiving objects:  17% (48/282)Receiving objects:  18% (51/282)Receiving objects:  19% (54/282)Receiving objects:  20% (57/282)Receiving objects:  21% (60/282)Receiving objects:  22% (63/282)Receiving objects:  23% (65/282)Receiving objects:  24% (68/282)Receiving objects:  25% (71/282)Receiving objects:  26% (74/282)Receiving objects:  27% (77/282)Receiving objects:  28% (79/282)Receiving objects:  29% (82/282)Receiving objects:  30% (85/282)Receiving objects:  31% (88/282)Receiving objects:  32% (91/282)Receiving objects:  33% (94/282)Receiving objects:  34% (96/282)Receiving objects:  35% (99/282)Receiving objects:  36% (102/282)Receiving objects:  37% (105/282)Receiving objects:  38% (108/282)Receiving objects:  39% (110/282)Receiving objects:  40% (113/282)Receiving objects:  41% (116/282)Receiving objects:  42% (119/282)Receiving objects:  43% (122/282)Receiving objects:  44% (125/282)Receiving objects:  45% (127/282)Receiving objects:  46% (130/282)Receiving objects:  47% (133/282)Receiving objects:  48% (136/282)Receiving objects:  49% (139/282)Receiving objects:  50% (141/282)Receiving objects:  51% (144/282)Receiving objects:  52% (147/282)Receiving objects:  53% (150/282)Receiving objects:  54% (153/282)Receiving objects:  55% (156/282)Receiving objects:  56% (158/282)Receiving objects:  57% (161/282)Receiving objects:  58% (164/282)Receiving objects:  59% (167/282)Receiving objects:  60% (170/282)Receiving objects:  61% (173/282)Receiving objects:  62% (175/282)Receiving objects:  63% (178/282)Receiving objects:  64% (181/282)Receiving objects:  65% (184/282)Receiving objects:  66% (187/282)Receiving objects:  67% (189/282)Receiving objects:  68% (192/282)Receiving objects:  69% (195/282)Receiving objects:  70% (198/282)Receiving objects:  71% (201/282)Receiving objects:  72% (204/282)Receiving objects:  73% (206/282)Receiving objects:  74% (209/282)Receiving objects:  75% (212/282)

.. parsed-literal::

    Receiving objects:  76% (215/282)

.. parsed-literal::

    Receiving objects:  77% (218/282)

.. parsed-literal::

    Receiving objects:  78% (220/282), 1.64 MiB | 3.23 MiB/s

.. parsed-literal::

    Receiving objects:  79% (223/282), 1.64 MiB | 3.23 MiB/s

.. parsed-literal::

    Receiving objects:  80% (226/282), 1.64 MiB | 3.23 MiB/s

.. parsed-literal::

    Receiving objects:  80% (227/282), 3.47 MiB | 3.40 MiB/s

.. parsed-literal::

    Receiving objects:  81% (229/282), 3.47 MiB | 3.40 MiB/s

.. parsed-literal::

    Receiving objects:  82% (232/282), 3.47 MiB | 3.40 MiB/s

.. parsed-literal::

    Receiving objects:  83% (235/282), 3.47 MiB | 3.40 MiB/s

.. parsed-literal::

    Receiving objects:  84% (237/282), 5.31 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  85% (240/282), 5.31 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  86% (243/282), 5.31 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  87% (246/282), 7.14 MiB | 3.47 MiB/s

.. parsed-literal::

    Receiving objects:  88% (249/282), 7.14 MiB | 3.47 MiB/s

.. parsed-literal::

    Receiving objects:  89% (251/282), 7.14 MiB | 3.47 MiB/s

.. parsed-literal::

    remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects:  90% (254/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  91% (257/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  92% (260/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  93% (263/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  94% (266/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  95% (268/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  96% (271/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  97% (274/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  98% (277/282), 8.97 MiB | 3.49 MiB/sReceiving objects:  99% (280/282), 8.97 MiB | 3.49 MiB/sReceiving objects: 100% (282/282), 8.97 MiB | 3.49 MiB/sReceiving objects: 100% (282/282), 9.22 MiB | 3.52 MiB/s, done.
    Resolving deltas:   0% (0/135)Resolving deltas:   2% (4/135)Resolving deltas:   5% (7/135)Resolving deltas:   6% (9/135)Resolving deltas:  11% (16/135)Resolving deltas:  17% (24/135)Resolving deltas:  18% (25/135)Resolving deltas:  23% (32/135)Resolving deltas:  25% (34/135)Resolving deltas:  27% (37/135)Resolving deltas:  28% (38/135)Resolving deltas:  34% (46/135)Resolving deltas:  40% (54/135)Resolving deltas:  45% (62/135)Resolving deltas:  46% (63/135)Resolving deltas:  50% (68/135)Resolving deltas:  57% (77/135)Resolving deltas:  58% (79/135)Resolving deltas:  59% (80/135)Resolving deltas:  60% (81/135)Resolving deltas:  61% (83/135)Resolving deltas:  66% (90/135)Resolving deltas:  71% (97/135)

.. parsed-literal::

    Resolving deltas: 100% (135/135)Resolving deltas: 100% (135/135), done.


.. code:: ipython3

    from pytorch_cifar_models import cifar10_mobilenetv2_x1_0
    
    model = cifar10_mobilenetv2_x1_0(pretrained=True)

OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation format using model conversion Python API.
``ov.convert_model`` accept PyTorch model instance and convert it into
``openvino.runtime.Model`` representation of model in OpenVINO.
Optionally, you may specify ``example_input`` which serves as a helper
for model tracing and ``input_shape`` for converting the model with
static shape. The converted model is ready to be loaded on a device for
inference and can be saved on a disk for next usage via the
``save_model`` function. More details about model conversion Python API
can be found on this
`page <https://docs.openvino.ai/2023.3/openvino_docs_model_processing_introduction.html>`__.

.. code:: ipython3

    import openvino as ov
    
    model.eval()
    
    ov_model = ov.convert_model(model, input=[1,3,32,32])
    
    ov.save_model(ov_model, MODEL_DIR / "mobilenet_v2.xml") 

Prepare Dataset
---------------

`back to top ⬆️ <#Table-of-contents:>`__

We will use `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__
dataset from
`torchvision <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>`__.
Preprocessing for model obtained from training
`config <https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf>`__

.. code:: ipython3

    import torch
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    dataset = CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


.. parsed-literal::

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz


.. parsed-literal::

      0%|          | 0/170498071 [00:00<?, ?it/s]

.. parsed-literal::

      0%|          | 32768/170498071 [00:00<17:10, 165400.38it/s]

.. parsed-literal::

      0%|          | 98304/170498071 [00:00<07:53, 360146.13it/s]

.. parsed-literal::

      0%|          | 196608/170498071 [00:00<04:53, 580743.16it/s]

.. parsed-literal::

      0%|          | 425984/170498071 [00:00<02:29, 1138809.29it/s]

.. parsed-literal::

      0%|          | 819200/170498071 [00:00<01:25, 1980588.44it/s]

.. parsed-literal::

      1%|          | 1179648/170498071 [00:00<01:09, 2421723.34it/s]

.. parsed-literal::

      1%|          | 1572864/170498071 [00:00<01:00, 2795941.30it/s]

.. parsed-literal::

      1%|          | 1966080/170498071 [00:00<00:55, 3033037.53it/s]

.. parsed-literal::

      1%|▏         | 2359296/170498071 [00:01<00:52, 3206292.66it/s]

.. parsed-literal::

      2%|▏         | 2752512/170498071 [00:01<00:50, 3308414.12it/s]

.. parsed-literal::

      2%|▏         | 3145728/170498071 [00:01<00:49, 3382848.82it/s]

.. parsed-literal::

      2%|▏         | 3538944/170498071 [00:01<00:48, 3432400.57it/s]

.. parsed-literal::

      2%|▏         | 3932160/170498071 [00:01<00:47, 3479365.07it/s]

.. parsed-literal::

      3%|▎         | 4325376/170498071 [00:01<00:47, 3499116.91it/s]

.. parsed-literal::

      3%|▎         | 4718592/170498071 [00:01<00:47, 3527111.73it/s]

.. parsed-literal::

      3%|▎         | 5111808/170498071 [00:01<00:46, 3531519.62it/s]

.. parsed-literal::

      3%|▎         | 5472256/170498071 [00:01<00:46, 3529318.26it/s]

.. parsed-literal::

      3%|▎         | 5865472/170498071 [00:02<00:46, 3549149.70it/s]

.. parsed-literal::

      4%|▎         | 6258688/170498071 [00:02<00:46, 3562985.35it/s]

.. parsed-literal::

      4%|▍         | 6651904/170498071 [00:02<00:46, 3558474.98it/s]

.. parsed-literal::

      4%|▍         | 7045120/170498071 [00:02<00:45, 3570143.85it/s]

.. parsed-literal::

      4%|▍         | 7438336/170498071 [00:02<00:45, 3576922.18it/s]

.. parsed-literal::

      5%|▍         | 7831552/170498071 [00:02<00:45, 3568632.85it/s]

.. parsed-literal::

      5%|▍         | 8192000/170498071 [00:02<00:45, 3544062.59it/s]

.. parsed-literal::

      5%|▌         | 8585216/170498071 [00:02<00:45, 3544031.60it/s]

.. parsed-literal::

      5%|▌         | 8978432/170498071 [00:02<00:45, 3545268.29it/s]

.. parsed-literal::

      5%|▌         | 9371648/170498071 [00:03<00:45, 3543490.57it/s]

.. parsed-literal::

      6%|▌         | 9764864/170498071 [00:03<00:45, 3545636.67it/s]

.. parsed-literal::

      6%|▌         | 10158080/170498071 [00:03<00:45, 3545069.53it/s]

.. parsed-literal::

      6%|▌         | 10551296/170498071 [00:03<00:44, 3559881.78it/s]

.. parsed-literal::

      6%|▋         | 10944512/170498071 [00:03<00:44, 3551369.20it/s]

.. parsed-literal::

      7%|▋         | 11337728/170498071 [00:03<00:44, 3564879.33it/s]

.. parsed-literal::

      7%|▋         | 11698176/170498071 [00:03<00:44, 3562378.48it/s]

.. parsed-literal::

      7%|▋         | 12091392/170498071 [00:03<00:44, 3571556.14it/s]

.. parsed-literal::

      7%|▋         | 12484608/170498071 [00:03<00:44, 3574867.11it/s]

.. parsed-literal::

      8%|▊         | 12877824/170498071 [00:04<00:44, 3550308.74it/s]

.. parsed-literal::

      8%|▊         | 13271040/170498071 [00:04<00:44, 3561931.30it/s]

.. parsed-literal::

      8%|▊         | 13664256/170498071 [00:04<00:44, 3555095.20it/s]

.. parsed-literal::

      8%|▊         | 14057472/170498071 [00:04<00:44, 3548885.57it/s]

.. parsed-literal::

      8%|▊         | 14450688/170498071 [00:04<00:43, 3548289.35it/s]

.. parsed-literal::

      9%|▊         | 14843904/170498071 [00:04<00:43, 3547324.81it/s]

.. parsed-literal::

      9%|▉         | 15237120/170498071 [00:04<00:43, 3547333.38it/s]

.. parsed-literal::

      9%|▉         | 15630336/170498071 [00:04<00:43, 3560580.39it/s]

.. parsed-literal::

      9%|▉         | 16023552/170498071 [00:04<00:43, 3554961.53it/s]

.. parsed-literal::

     10%|▉         | 16416768/170498071 [00:05<00:43, 3549091.05it/s]

.. parsed-literal::

     10%|▉         | 16809984/170498071 [00:05<00:43, 3563993.52it/s]

.. parsed-literal::

     10%|█         | 17203200/170498071 [00:05<00:43, 3559977.68it/s]

.. parsed-literal::

     10%|█         | 17596416/170498071 [00:05<00:42, 3568951.11it/s]

.. parsed-literal::

     11%|█         | 17989632/170498071 [00:05<00:42, 3562048.91it/s]

.. parsed-literal::

     11%|█         | 18382848/170498071 [00:05<00:42, 3574349.36it/s]

.. parsed-literal::

     11%|█         | 18776064/170498071 [00:05<00:42, 3560324.50it/s]

.. parsed-literal::

     11%|█         | 19136512/170498071 [00:05<00:42, 3564354.68it/s]

.. parsed-literal::

     11%|█▏        | 19529728/170498071 [00:05<00:42, 3555540.30it/s]

.. parsed-literal::

     12%|█▏        | 19922944/170498071 [00:05<00:42, 3568736.92it/s]

.. parsed-literal::

     12%|█▏        | 20316160/170498071 [00:06<00:42, 3558457.50it/s]

.. parsed-literal::

     12%|█▏        | 20709376/170498071 [00:06<00:42, 3554587.73it/s]

.. parsed-literal::

     12%|█▏        | 21102592/170498071 [00:06<00:42, 3552073.22it/s]

.. parsed-literal::

     13%|█▎        | 21495808/170498071 [00:06<00:41, 3564788.39it/s]

.. parsed-literal::

     13%|█▎        | 21889024/170498071 [00:06<00:41, 3551887.34it/s]

.. parsed-literal::

     13%|█▎        | 22282240/170498071 [00:06<00:41, 3563483.58it/s]

.. parsed-literal::

     13%|█▎        | 22675456/170498071 [00:06<00:41, 3556453.43it/s]

.. parsed-literal::

     14%|█▎        | 23068672/170498071 [00:06<00:41, 3553670.60it/s]

.. parsed-literal::

     14%|█▍        | 23461888/170498071 [00:06<00:41, 3550202.48it/s]

.. parsed-literal::

     14%|█▍        | 23855104/170498071 [00:07<00:41, 3548895.43it/s]

.. parsed-literal::

     14%|█▍        | 24248320/170498071 [00:07<00:41, 3546992.15it/s]

.. parsed-literal::

     14%|█▍        | 24608768/170498071 [00:07<00:41, 3549544.21it/s]

.. parsed-literal::

     15%|█▍        | 25001984/170498071 [00:07<00:40, 3549882.59it/s]

.. parsed-literal::

     15%|█▍        | 25395200/170498071 [00:07<00:40, 3549485.17it/s]

.. parsed-literal::

     15%|█▌        | 25788416/170498071 [00:07<00:40, 3561550.83it/s]

.. parsed-literal::

     15%|█▌        | 26181632/170498071 [00:07<00:40, 3568879.63it/s]

.. parsed-literal::

     16%|█▌        | 26574848/170498071 [00:07<00:40, 3558381.72it/s]

.. parsed-literal::

     16%|█▌        | 26968064/170498071 [00:07<00:40, 3548967.93it/s]

.. parsed-literal::

     16%|█▌        | 27361280/170498071 [00:08<00:40, 3562580.09it/s]

.. parsed-literal::

     16%|█▋        | 27754496/170498071 [00:08<00:40, 3559588.22it/s]

.. parsed-literal::

     17%|█▋        | 28147712/170498071 [00:08<00:40, 3558630.93it/s]

.. parsed-literal::

     17%|█▋        | 28540928/170498071 [00:08<00:39, 3555414.89it/s]

.. parsed-literal::

     17%|█▋        | 28934144/170498071 [00:08<00:39, 3551700.71it/s]

.. parsed-literal::

     17%|█▋        | 29327360/170498071 [00:08<00:39, 3549475.40it/s]

.. parsed-literal::

     17%|█▋        | 29720576/170498071 [00:08<00:39, 3543596.42it/s]

.. parsed-literal::

     18%|█▊        | 30113792/170498071 [00:08<00:39, 3557966.72it/s]

.. parsed-literal::

     18%|█▊        | 30507008/170498071 [00:08<00:39, 3552269.41it/s]

.. parsed-literal::

     18%|█▊        | 30900224/170498071 [00:09<00:39, 3549800.19it/s]

.. parsed-literal::

     18%|█▊        | 31293440/170498071 [00:09<00:39, 3558197.12it/s]

.. parsed-literal::

     19%|█▊        | 31686656/170498071 [00:09<00:39, 3553319.24it/s]

.. parsed-literal::

     19%|█▉        | 32079872/170498071 [00:09<00:38, 3565816.87it/s]

.. parsed-literal::

     19%|█▉        | 32440320/170498071 [00:09<00:38, 3559251.24it/s]

.. parsed-literal::

     19%|█▉        | 32833536/170498071 [00:09<00:38, 3568883.69it/s]

.. parsed-literal::

     19%|█▉        | 33193984/170498071 [00:09<00:38, 3529843.65it/s]

.. parsed-literal::

     20%|█▉        | 33587200/170498071 [00:09<00:38, 3533902.78it/s]

.. parsed-literal::

     20%|█▉        | 33980416/170498071 [00:09<00:38, 3552288.24it/s]

.. parsed-literal::

     20%|██        | 34373632/170498071 [00:10<00:38, 3549985.87it/s]

.. parsed-literal::

     20%|██        | 34766848/170498071 [00:10<00:38, 3563816.63it/s]

.. parsed-literal::

     21%|██        | 35160064/170498071 [00:10<00:38, 3550534.02it/s]

.. parsed-literal::

     21%|██        | 35553280/170498071 [00:10<00:38, 3547626.36it/s]

.. parsed-literal::

     21%|██        | 35946496/170498071 [00:10<00:37, 3547746.28it/s]

.. parsed-literal::

     21%|██▏       | 36339712/170498071 [00:10<00:37, 3561814.91it/s]

.. parsed-literal::

     22%|██▏       | 36732928/170498071 [00:10<00:37, 3556634.66it/s]

.. parsed-literal::

     22%|██▏       | 37126144/170498071 [00:10<00:37, 3566448.11it/s]

.. parsed-literal::

     22%|██▏       | 37519360/170498071 [00:10<00:37, 3575430.66it/s]

.. parsed-literal::

     22%|██▏       | 37912576/170498071 [00:11<00:37, 3563826.39it/s]

.. parsed-literal::

     22%|██▏       | 38305792/170498071 [00:11<00:37, 3558700.72it/s]

.. parsed-literal::

     23%|██▎       | 38666240/170498071 [00:11<00:37, 3560403.05it/s]

.. parsed-literal::

     23%|██▎       | 39059456/170498071 [00:11<00:36, 3556571.47it/s]

.. parsed-literal::

     23%|██▎       | 39452672/170498071 [00:11<00:36, 3555197.40it/s]

.. parsed-literal::

     23%|██▎       | 39845888/170498071 [00:11<00:36, 3553779.08it/s]

.. parsed-literal::

     24%|██▎       | 40239104/170498071 [00:11<00:36, 3551417.01it/s]

.. parsed-literal::

     24%|██▍       | 40632320/170498071 [00:11<00:36, 3547718.37it/s]

.. parsed-literal::

     24%|██▍       | 41025536/170498071 [00:11<00:36, 3558821.35it/s]

.. parsed-literal::

     24%|██▍       | 41418752/170498071 [00:12<00:36, 3570243.40it/s]

.. parsed-literal::

     25%|██▍       | 41811968/170498071 [00:12<00:36, 3558069.54it/s]

.. parsed-literal::

     25%|██▍       | 42205184/170498071 [00:12<00:36, 3555392.32it/s]

.. parsed-literal::

     25%|██▍       | 42598400/170498071 [00:12<00:35, 3552945.21it/s]

.. parsed-literal::

     25%|██▌       | 42991616/170498071 [00:12<00:35, 3565505.94it/s]

.. parsed-literal::

     25%|██▌       | 43384832/170498071 [00:12<00:35, 3555857.99it/s]

.. parsed-literal::

     26%|██▌       | 43778048/170498071 [00:12<00:35, 3552596.59it/s]

.. parsed-literal::

     26%|██▌       | 44171264/170498071 [00:12<00:35, 3549274.48it/s]

.. parsed-literal::

     26%|██▌       | 44564480/170498071 [00:12<00:35, 3544741.17it/s]

.. parsed-literal::

     26%|██▋       | 44957696/170498071 [00:13<00:35, 3545828.65it/s]

.. parsed-literal::

     27%|██▋       | 45350912/170498071 [00:13<00:35, 3559659.89it/s]

.. parsed-literal::

     27%|██▋       | 45744128/170498071 [00:13<00:35, 3551098.29it/s]

.. parsed-literal::

     27%|██▋       | 46137344/170498071 [00:13<00:34, 3564204.76it/s]

.. parsed-literal::

     27%|██▋       | 46497792/170498071 [00:13<00:34, 3569395.47it/s]

.. parsed-literal::

     28%|██▊       | 46891008/170498071 [00:13<00:34, 3561863.01it/s]

.. parsed-literal::

     28%|██▊       | 47284224/170498071 [00:13<00:34, 3567880.10it/s]

.. parsed-literal::

     28%|██▊       | 47677440/170498071 [00:13<00:34, 3556957.85it/s]

.. parsed-literal::

     28%|██▊       | 48070656/170498071 [00:13<00:34, 3552223.84it/s]

.. parsed-literal::

     28%|██▊       | 48463872/170498071 [00:14<00:34, 3547628.03it/s]

.. parsed-literal::

     29%|██▊       | 48857088/170498071 [00:14<00:34, 3533134.95it/s]

.. parsed-literal::

     29%|██▉       | 49250304/170498071 [00:14<00:34, 3539026.54it/s]

.. parsed-literal::

     29%|██▉       | 49643520/170498071 [00:14<00:34, 3541345.88it/s]

.. parsed-literal::

     29%|██▉       | 50036736/170498071 [00:14<00:34, 3542782.10it/s]

.. parsed-literal::

     30%|██▉       | 50429952/170498071 [00:14<00:33, 3546294.62it/s]

.. parsed-literal::

     30%|██▉       | 50823168/170498071 [00:14<00:33, 3548538.54it/s]

.. parsed-literal::

     30%|███       | 51216384/170498071 [00:14<00:33, 3546227.48it/s]

.. parsed-literal::

     30%|███       | 51576832/170498071 [00:14<00:33, 3551977.46it/s]

.. parsed-literal::

     30%|███       | 51970048/170498071 [00:15<00:33, 3548814.09it/s]

.. parsed-literal::

     31%|███       | 52363264/170498071 [00:15<00:33, 3548685.66it/s]

.. parsed-literal::

     31%|███       | 52756480/170498071 [00:15<00:33, 3549116.53it/s]

.. parsed-literal::

     31%|███       | 53149696/170498071 [00:15<00:33, 3545436.68it/s]

.. parsed-literal::

     31%|███▏      | 53542912/170498071 [00:15<00:32, 3563283.62it/s]

.. parsed-literal::

     32%|███▏      | 53936128/170498071 [00:15<00:32, 3549644.55it/s]

.. parsed-literal::

     32%|███▏      | 54329344/170498071 [00:15<00:32, 3563056.99it/s]

.. parsed-literal::

     32%|███▏      | 54722560/170498071 [00:15<00:32, 3568583.27it/s]

.. parsed-literal::

     32%|███▏      | 55115776/170498071 [00:15<00:32, 3561331.09it/s]

.. parsed-literal::

     33%|███▎      | 55508992/170498071 [00:16<00:32, 3557590.83it/s]

.. parsed-literal::

     33%|███▎      | 55902208/170498071 [00:16<00:32, 3554532.54it/s]

.. parsed-literal::

     33%|███▎      | 56295424/170498071 [00:16<00:32, 3549222.73it/s]

.. parsed-literal::

     33%|███▎      | 56688640/170498071 [00:16<00:32, 3543742.29it/s]

.. parsed-literal::

     33%|███▎      | 57081856/170498071 [00:16<00:31, 3557420.27it/s]

.. parsed-literal::

     34%|███▎      | 57475072/170498071 [00:16<00:31, 3556208.15it/s]

.. parsed-literal::

     34%|███▍      | 57868288/170498071 [00:16<00:31, 3553788.06it/s]

.. parsed-literal::

     34%|███▍      | 58261504/170498071 [00:16<00:31, 3567339.52it/s]

.. parsed-literal::

     34%|███▍      | 58654720/170498071 [00:16<00:31, 3560489.44it/s]

.. parsed-literal::

     35%|███▍      | 59047936/170498071 [00:17<00:31, 3555730.86it/s]

.. parsed-literal::

     35%|███▍      | 59441152/170498071 [00:17<00:31, 3552386.83it/s]

.. parsed-literal::

     35%|███▌      | 59801600/170498071 [00:17<00:31, 3563190.87it/s]

.. parsed-literal::

     35%|███▌      | 60194816/170498071 [00:17<00:30, 3572593.37it/s]

.. parsed-literal::

     36%|███▌      | 60588032/170498071 [00:17<00:30, 3579105.00it/s]

.. parsed-literal::

     36%|███▌      | 60981248/170498071 [00:17<00:30, 3568909.03it/s]

.. parsed-literal::

     36%|███▌      | 61374464/170498071 [00:17<00:30, 3561013.01it/s]

.. parsed-literal::

     36%|███▌      | 61767680/170498071 [00:17<00:30, 3558419.60it/s]

.. parsed-literal::

     36%|███▋      | 62160896/170498071 [00:17<00:30, 3553798.84it/s]

.. parsed-literal::

     37%|███▋      | 62554112/170498071 [00:17<00:30, 3550594.45it/s]

.. parsed-literal::

     37%|███▋      | 62947328/170498071 [00:18<00:30, 3563118.81it/s]

.. parsed-literal::

     37%|███▋      | 63340544/170498071 [00:18<00:30, 3559864.30it/s]

.. parsed-literal::

     37%|███▋      | 63733760/170498071 [00:18<00:30, 3555422.47it/s]

.. parsed-literal::

     38%|███▊      | 64126976/170498071 [00:18<00:29, 3567573.44it/s]

.. parsed-literal::

     38%|███▊      | 64520192/170498071 [00:18<00:29, 3557283.17it/s]

.. parsed-literal::

     38%|███▊      | 64880640/170498071 [00:18<00:29, 3558505.05it/s]

.. parsed-literal::

     38%|███▊      | 65273856/170498071 [00:18<00:29, 3570029.47it/s]

.. parsed-literal::

     39%|███▊      | 65667072/170498071 [00:18<00:29, 3558652.68it/s]

.. parsed-literal::

     39%|███▊      | 66060288/170498071 [00:18<00:29, 3554782.77it/s]

.. parsed-literal::

     39%|███▉      | 66453504/170498071 [00:19<00:29, 3565469.56it/s]

.. parsed-literal::

     39%|███▉      | 66846720/170498071 [00:19<00:29, 3558661.39it/s]

.. parsed-literal::

     39%|███▉      | 67239936/170498071 [00:19<00:29, 3553700.02it/s]

.. parsed-literal::

     40%|███▉      | 67633152/170498071 [00:19<00:28, 3552242.72it/s]

.. parsed-literal::

     40%|███▉      | 68026368/170498071 [00:19<00:28, 3552679.66it/s]

.. parsed-literal::

     40%|████      | 68419584/170498071 [00:19<00:28, 3549726.56it/s]

.. parsed-literal::

     40%|████      | 68812800/170498071 [00:19<00:28, 3548213.76it/s]

.. parsed-literal::

     41%|████      | 69206016/170498071 [00:19<00:28, 3548083.76it/s]

.. parsed-literal::

     41%|████      | 69599232/170498071 [00:19<00:28, 3560933.55it/s]

.. parsed-literal::

     41%|████      | 69992448/170498071 [00:20<00:28, 3553253.51it/s]

.. parsed-literal::

     41%|████▏     | 70385664/170498071 [00:20<00:28, 3564149.20it/s]

.. parsed-literal::

     42%|████▏     | 70778880/170498071 [00:20<00:27, 3571555.14it/s]

.. parsed-literal::

     42%|████▏     | 71139328/170498071 [00:20<00:27, 3576779.68it/s]

.. parsed-literal::

     42%|████▏     | 71532544/170498071 [00:20<00:27, 3582195.60it/s]

.. parsed-literal::

     42%|████▏     | 71925760/170498071 [00:20<00:27, 3571305.79it/s]

.. parsed-literal::

     42%|████▏     | 72318976/170498071 [00:20<00:27, 3564951.45it/s]

.. parsed-literal::

     43%|████▎     | 72712192/170498071 [00:20<00:27, 3521775.30it/s]

.. parsed-literal::

     43%|████▎     | 73105408/170498071 [00:20<00:27, 3530120.52it/s]

.. parsed-literal::

     43%|████▎     | 73498624/170498071 [00:21<00:27, 3536449.01it/s]

.. parsed-literal::

     43%|████▎     | 73891840/170498071 [00:21<00:27, 3538233.87it/s]

.. parsed-literal::

     44%|████▎     | 74285056/170498071 [00:21<00:27, 3555140.57it/s]

.. parsed-literal::

     44%|████▍     | 74678272/170498071 [00:21<00:26, 3551879.36it/s]

.. parsed-literal::

     44%|████▍     | 75071488/170498071 [00:21<00:26, 3564068.18it/s]

.. parsed-literal::

     44%|████▍     | 75431936/170498071 [00:21<00:26, 3568512.74it/s]

.. parsed-literal::

     44%|████▍     | 75825152/170498071 [00:21<00:26, 3560949.27it/s]

.. parsed-literal::

     45%|████▍     | 76218368/170498071 [00:21<00:26, 3571556.64it/s]

.. parsed-literal::

     45%|████▍     | 76611584/170498071 [00:21<00:26, 3562293.29it/s]

.. parsed-literal::

     45%|████▌     | 77004800/170498071 [00:22<00:26, 3557497.11it/s]

.. parsed-literal::

     45%|████▌     | 77398016/170498071 [00:22<00:26, 3551372.40it/s]

.. parsed-literal::

     46%|████▌     | 77791232/170498071 [00:22<00:26, 3564629.41it/s]

.. parsed-literal::

     46%|████▌     | 78184448/170498071 [00:22<00:25, 3556824.13it/s]

.. parsed-literal::

     46%|████▌     | 78544896/170498071 [00:22<00:25, 3554105.77it/s]

.. parsed-literal::

     46%|████▋     | 78938112/170498071 [00:22<00:25, 3553852.21it/s]

.. parsed-literal::

     47%|████▋     | 79331328/170498071 [00:22<00:25, 3567421.74it/s]

.. parsed-literal::

     47%|████▋     | 79724544/170498071 [00:22<00:25, 3562738.22it/s]

.. parsed-literal::

     47%|████▋     | 80117760/170498071 [00:22<00:25, 3557648.18it/s]

.. parsed-literal::

     47%|████▋     | 80510976/170498071 [00:23<00:25, 3552283.34it/s]

.. parsed-literal::

     47%|████▋     | 80904192/170498071 [00:23<00:25, 3549328.82it/s]

.. parsed-literal::

     48%|████▊     | 81297408/170498071 [00:23<00:25, 3549896.68it/s]

.. parsed-literal::

     48%|████▊     | 81690624/170498071 [00:23<00:25, 3547398.14it/s]

.. parsed-literal::

     48%|████▊     | 82083840/170498071 [00:23<00:24, 3544126.94it/s]

.. parsed-literal::

     48%|████▊     | 82477056/170498071 [00:23<00:24, 3544074.92it/s]

.. parsed-literal::

     49%|████▊     | 82870272/170498071 [00:23<00:24, 3559888.56it/s]

.. parsed-literal::

     49%|████▉     | 83263488/170498071 [00:23<00:24, 3568489.68it/s]

.. parsed-literal::

     49%|████▉     | 83656704/170498071 [00:23<00:24, 3556659.93it/s]

.. parsed-literal::

     49%|████▉     | 84049920/170498071 [00:24<00:24, 3553545.27it/s]

.. parsed-literal::

     50%|████▉     | 84443136/170498071 [00:24<00:24, 3564305.63it/s]

.. parsed-literal::

     50%|████▉     | 84836352/170498071 [00:24<00:24, 3557378.29it/s]

.. parsed-literal::

     50%|████▉     | 85229568/170498071 [00:24<00:23, 3567996.61it/s]

.. parsed-literal::

     50%|█████     | 85622784/170498071 [00:24<00:23, 3562722.28it/s]

.. parsed-literal::

     50%|█████     | 86016000/170498071 [00:24<00:23, 3555692.57it/s]

.. parsed-literal::

     51%|█████     | 86376448/170498071 [00:24<00:23, 3542488.81it/s]

.. parsed-literal::

     51%|█████     | 86769664/170498071 [00:24<00:23, 3542550.24it/s]

.. parsed-literal::

     51%|█████     | 87162880/170498071 [00:24<00:23, 3541552.97it/s]

.. parsed-literal::

     51%|█████▏    | 87556096/170498071 [00:25<00:23, 3540808.59it/s]

.. parsed-literal::

     52%|█████▏    | 87949312/170498071 [00:25<00:23, 3557750.99it/s]

.. parsed-literal::

     52%|█████▏    | 88342528/170498071 [00:25<00:23, 3553225.52it/s]

.. parsed-literal::

     52%|█████▏    | 88735744/170498071 [00:25<00:23, 3551615.40it/s]

.. parsed-literal::

     52%|█████▏    | 89128960/170498071 [00:25<00:22, 3547525.39it/s]

.. parsed-literal::

     53%|█████▎    | 89522176/170498071 [00:25<00:22, 3561150.11it/s]

.. parsed-literal::

     53%|█████▎    | 89915392/170498071 [00:25<00:22, 3556576.40it/s]

.. parsed-literal::

     53%|█████▎    | 90308608/170498071 [00:25<00:22, 3551415.52it/s]

.. parsed-literal::

     53%|█████▎    | 90701824/170498071 [00:25<00:22, 3546835.80it/s]

.. parsed-literal::

     53%|█████▎    | 91095040/170498071 [00:26<00:22, 3545597.47it/s]

.. parsed-literal::

     54%|█████▎    | 91488256/170498071 [00:26<00:22, 3544303.92it/s]

.. parsed-literal::

     54%|█████▍    | 91881472/170498071 [00:26<00:22, 3544413.65it/s]

.. parsed-literal::

     54%|█████▍    | 92274688/170498071 [00:26<00:22, 3542891.30it/s]

.. parsed-literal::

     54%|█████▍    | 92667904/170498071 [00:26<00:21, 3556165.26it/s]

.. parsed-literal::

     55%|█████▍    | 93061120/170498071 [00:26<00:21, 3549941.72it/s]

.. parsed-literal::

     55%|█████▍    | 93454336/170498071 [00:26<00:21, 3561833.14it/s]

.. parsed-literal::

     55%|█████▌    | 93847552/170498071 [00:26<00:21, 3557437.54it/s]

.. parsed-literal::

     55%|█████▌    | 94208000/170498071 [00:26<00:21, 3557836.18it/s]

.. parsed-literal::

     55%|█████▌    | 94601216/170498071 [00:27<00:21, 3554093.08it/s]

.. parsed-literal::

     56%|█████▌    | 94994432/170498071 [00:27<00:21, 3550113.57it/s]

.. parsed-literal::

     56%|█████▌    | 95387648/170498071 [00:27<00:21, 3546949.63it/s]

.. parsed-literal::

     56%|█████▌    | 95780864/170498071 [00:27<00:21, 3546367.06it/s]

.. parsed-literal::

     56%|█████▋    | 96174080/170498071 [00:27<00:20, 3545405.43it/s]

.. parsed-literal::

     57%|█████▋    | 96567296/170498071 [00:27<00:20, 3545202.35it/s]

.. parsed-literal::

     57%|█████▋    | 96960512/170498071 [00:27<00:20, 3519338.12it/s]

.. parsed-literal::

     57%|█████▋    | 97353728/170498071 [00:27<00:20, 3528340.81it/s]

.. parsed-literal::

     57%|█████▋    | 97746944/170498071 [00:27<00:20, 3545296.46it/s]

.. parsed-literal::

     58%|█████▊    | 98140160/170498071 [00:28<00:20, 3544079.10it/s]

.. parsed-literal::

     58%|█████▊    | 98533376/170498071 [00:28<00:20, 3543211.90it/s]

.. parsed-literal::

     58%|█████▊    | 98926592/170498071 [00:28<00:20, 3558151.71it/s]

.. parsed-literal::

     58%|█████▊    | 99319808/170498071 [00:28<00:20, 3552078.65it/s]

.. parsed-literal::

     58%|█████▊    | 99713024/170498071 [00:28<00:19, 3548708.70it/s]

.. parsed-literal::

     59%|█████▊    | 100106240/170498071 [00:28<00:19, 3546946.50it/s]

.. parsed-literal::

     59%|█████▉    | 100499456/170498071 [00:28<00:19, 3546443.82it/s]

.. parsed-literal::

     59%|█████▉    | 100892672/170498071 [00:28<00:19, 3558494.48it/s]

.. parsed-literal::

     59%|█████▉    | 101285888/170498071 [00:28<00:19, 3554301.08it/s]

.. parsed-literal::

     60%|█████▉    | 101679104/170498071 [00:28<00:19, 3566689.42it/s]

.. parsed-literal::

     60%|█████▉    | 102072320/170498071 [00:29<00:19, 3555813.14it/s]

.. parsed-literal::

     60%|██████    | 102432768/170498071 [00:29<00:19, 3551781.54it/s]

.. parsed-literal::

     60%|██████    | 102825984/170498071 [00:29<00:19, 3548156.57it/s]

.. parsed-literal::

     61%|██████    | 103219200/170498071 [00:29<00:18, 3562319.87it/s]

.. parsed-literal::

     61%|██████    | 103612416/170498071 [00:29<00:18, 3555487.05it/s]

.. parsed-literal::

     61%|██████    | 104005632/170498071 [00:29<00:18, 3567306.82it/s]

.. parsed-literal::

     61%|██████    | 104398848/170498071 [00:29<00:18, 3559009.94it/s]

.. parsed-literal::

     61%|██████▏   | 104792064/170498071 [00:29<00:18, 3552927.26it/s]

.. parsed-literal::

     62%|██████▏   | 105185280/170498071 [00:29<00:18, 3551061.30it/s]

.. parsed-literal::

     62%|██████▏   | 105578496/170498071 [00:30<00:18, 3552668.89it/s]

.. parsed-literal::

     62%|██████▏   | 105971712/170498071 [00:30<00:18, 3549976.05it/s]

.. parsed-literal::

     62%|██████▏   | 106364928/170498071 [00:30<00:18, 3540955.08it/s]

.. parsed-literal::

     63%|██████▎   | 106758144/170498071 [00:30<00:17, 3554906.47it/s]

.. parsed-literal::

     63%|██████▎   | 107151360/170498071 [00:30<00:17, 3549802.36it/s]

.. parsed-literal::

     63%|██████▎   | 107544576/170498071 [00:30<00:17, 3544855.34it/s]

.. parsed-literal::

     63%|██████▎   | 107937792/170498071 [00:30<00:17, 3558873.38it/s]

.. parsed-literal::

     64%|██████▎   | 108331008/170498071 [00:30<00:17, 3550763.01it/s]

.. parsed-literal::

     64%|██████▍   | 108724224/170498071 [00:30<00:17, 3562839.49it/s]

.. parsed-literal::

     64%|██████▍   | 109117440/170498071 [00:31<00:17, 3558091.68it/s]

.. parsed-literal::

     64%|██████▍   | 109510656/170498071 [00:31<00:17, 3554932.19it/s]

.. parsed-literal::

     64%|██████▍   | 109903872/170498071 [00:31<00:17, 3549645.43it/s]

.. parsed-literal::

     65%|██████▍   | 110297088/170498071 [00:31<00:16, 3547743.62it/s]

.. parsed-literal::

     65%|██████▍   | 110690304/170498071 [00:31<00:16, 3561166.86it/s]

.. parsed-literal::

     65%|██████▌   | 111083520/170498071 [00:31<00:16, 3555571.34it/s]

.. parsed-literal::

     65%|██████▌   | 111476736/170498071 [00:31<00:16, 3567045.09it/s]

.. parsed-literal::

     66%|██████▌   | 111869952/170498071 [00:31<00:16, 3575571.93it/s]

.. parsed-literal::

     66%|██████▌   | 112263168/170498071 [00:31<00:16, 3564401.95it/s]

.. parsed-literal::

     66%|██████▌   | 112656384/170498071 [00:32<00:16, 3554380.10it/s]

.. parsed-literal::

     66%|██████▋   | 113049600/170498071 [00:32<00:16, 3565026.26it/s]

.. parsed-literal::

     67%|██████▋   | 113410048/170498071 [00:32<00:16, 3565436.96it/s]

.. parsed-literal::

     67%|██████▋   | 113803264/170498071 [00:32<00:15, 3554937.99it/s]

.. parsed-literal::

     67%|██████▋   | 114196480/170498071 [00:32<00:15, 3552372.97it/s]

.. parsed-literal::

     67%|██████▋   | 114589696/170498071 [00:32<00:15, 3548083.36it/s]

.. parsed-literal::

     67%|██████▋   | 114982912/170498071 [00:32<00:15, 3545455.23it/s]

.. parsed-literal::

     68%|██████▊   | 115376128/170498071 [00:32<00:15, 3541892.66it/s]

.. parsed-literal::

     68%|██████▊   | 115769344/170498071 [00:32<00:15, 3557477.55it/s]

.. parsed-literal::

     68%|██████▊   | 116162560/170498071 [00:33<00:15, 3552531.42it/s]

.. parsed-literal::

     68%|██████▊   | 116555776/170498071 [00:33<00:15, 3552198.86it/s]

.. parsed-literal::

     69%|██████▊   | 116948992/170498071 [00:33<00:15, 3550650.53it/s]

.. parsed-literal::

     69%|██████▉   | 117342208/170498071 [00:33<00:14, 3549572.87it/s]

.. parsed-literal::

     69%|██████▉   | 117735424/170498071 [00:33<00:14, 3561613.30it/s]

.. parsed-literal::

     69%|██████▉   | 118128640/170498071 [00:33<00:14, 3553200.25it/s]

.. parsed-literal::

     69%|██████▉   | 118489088/170498071 [00:33<00:14, 3554024.13it/s]

.. parsed-literal::

     70%|██████▉   | 118882304/170498071 [00:33<00:14, 3552101.05it/s]

.. parsed-literal::

     70%|██████▉   | 119275520/170498071 [00:33<00:14, 3549487.68it/s]

.. parsed-literal::

     70%|███████   | 119668736/170498071 [00:34<00:14, 3549003.29it/s]

.. parsed-literal::

     70%|███████   | 120061952/170498071 [00:34<00:14, 3548037.39it/s]

.. parsed-literal::

     71%|███████   | 120455168/170498071 [00:34<00:14, 3547502.50it/s]

.. parsed-literal::

     71%|███████   | 120848384/170498071 [00:34<00:14, 3545621.83it/s]

.. parsed-literal::

     71%|███████   | 121241600/170498071 [00:34<00:13, 3543533.33it/s]

.. parsed-literal::

     71%|███████▏  | 121634816/170498071 [00:34<00:13, 3559830.41it/s]

.. parsed-literal::

     72%|███████▏  | 122028032/170498071 [00:34<00:13, 3555917.69it/s]

.. parsed-literal::

     72%|███████▏  | 122421248/170498071 [00:34<00:13, 3566569.94it/s]

.. parsed-literal::

     72%|███████▏  | 122814464/170498071 [00:34<00:13, 3560113.52it/s]

.. parsed-literal::

     72%|███████▏  | 123207680/170498071 [00:35<00:13, 3554020.66it/s]

.. parsed-literal::

     72%|███████▏  | 123568128/170498071 [00:35<00:13, 3514877.41it/s]

.. parsed-literal::

     73%|███████▎  | 123961344/170498071 [00:35<00:13, 3523722.57it/s]

.. parsed-literal::

     73%|███████▎  | 124354560/170498071 [00:35<00:13, 3543786.50it/s]

.. parsed-literal::

     73%|███████▎  | 124747776/170498071 [00:35<00:12, 3543348.63it/s]

.. parsed-literal::

     73%|███████▎  | 125140992/170498071 [00:35<00:12, 3543733.46it/s]

.. parsed-literal::

     74%|███████▎  | 125534208/170498071 [00:35<00:12, 3559464.13it/s]

.. parsed-literal::

     74%|███████▍  | 125927424/170498071 [00:35<00:12, 3551482.29it/s]

.. parsed-literal::

     74%|███████▍  | 126320640/170498071 [00:35<00:12, 3547469.13it/s]

.. parsed-literal::

     74%|███████▍  | 126713856/170498071 [00:36<00:12, 3551048.56it/s]

.. parsed-literal::

     75%|███████▍  | 127107072/170498071 [00:36<00:12, 3548979.16it/s]

.. parsed-literal::

     75%|███████▍  | 127500288/170498071 [00:36<00:12, 3549917.15it/s]

.. parsed-literal::

     75%|███████▌  | 127893504/170498071 [00:36<00:11, 3563831.11it/s]

.. parsed-literal::

     75%|███████▌  | 128286720/170498071 [00:36<00:11, 3557720.34it/s]

.. parsed-literal::

     75%|███████▌  | 128679936/170498071 [00:36<00:11, 3552742.00it/s]

.. parsed-literal::

     76%|███████▌  | 129040384/170498071 [00:36<00:11, 3544418.21it/s]

.. parsed-literal::

     76%|███████▌  | 129433600/170498071 [00:36<00:11, 3545051.06it/s]

.. parsed-literal::

     76%|███████▌  | 129826816/170498071 [00:36<00:11, 3560077.86it/s]

.. parsed-literal::

     76%|███████▋  | 130220032/170498071 [00:37<00:11, 3570038.39it/s]

.. parsed-literal::

     77%|███████▋  | 130613248/170498071 [00:37<00:11, 3562803.21it/s]

.. parsed-literal::

     77%|███████▋  | 131006464/170498071 [00:37<00:11, 3571768.84it/s]

.. parsed-literal::

     77%|███████▋  | 131399680/170498071 [00:37<00:10, 3560613.04it/s]

.. parsed-literal::

     77%|███████▋  | 131760128/170498071 [00:37<00:10, 3565262.02it/s]

.. parsed-literal::

     78%|███████▊  | 132153344/170498071 [00:37<00:10, 3572260.37it/s]

.. parsed-literal::

     78%|███████▊  | 132546560/170498071 [00:37<00:10, 3564278.48it/s]

.. parsed-literal::

     78%|███████▊  | 132939776/170498071 [00:37<00:10, 3574072.18it/s]

.. parsed-literal::

     78%|███████▊  | 133332992/170498071 [00:37<00:10, 3565708.01it/s]

.. parsed-literal::

     78%|███████▊  | 133726208/170498071 [00:38<00:10, 3560228.28it/s]

.. parsed-literal::

     79%|███████▊  | 134119424/170498071 [00:38<00:10, 3552926.56it/s]

.. parsed-literal::

     79%|███████▉  | 134512640/170498071 [00:38<00:10, 3544397.09it/s]

.. parsed-literal::

     79%|███████▉  | 134905856/170498071 [00:38<00:10, 3544387.43it/s]

.. parsed-literal::

     79%|███████▉  | 135299072/170498071 [00:38<00:09, 3548294.56it/s]

.. parsed-literal::

     80%|███████▉  | 135692288/170498071 [00:38<00:09, 3546502.42it/s]

.. parsed-literal::

     80%|███████▉  | 136085504/170498071 [00:38<00:09, 3542527.41it/s]

.. parsed-literal::

     80%|████████  | 136478720/170498071 [00:38<00:09, 3546064.50it/s]

.. parsed-literal::

     80%|████████  | 136871936/170498071 [00:38<00:09, 3543807.74it/s]

.. parsed-literal::

     81%|████████  | 137265152/170498071 [00:39<00:09, 3544326.84it/s]

.. parsed-literal::

     81%|████████  | 137658368/170498071 [00:39<00:09, 3557599.42it/s]

.. parsed-literal::

     81%|████████  | 138051584/170498071 [00:39<00:09, 3556013.72it/s]

.. parsed-literal::

     81%|████████  | 138444800/170498071 [00:39<00:08, 3567232.38it/s]

.. parsed-literal::

     81%|████████▏ | 138838016/170498071 [00:39<00:08, 3575713.29it/s]

.. parsed-literal::

     82%|████████▏ | 139231232/170498071 [00:39<00:08, 3565482.51it/s]

.. parsed-literal::

     82%|████████▏ | 139624448/170498071 [00:39<00:08, 3558123.28it/s]

.. parsed-literal::

     82%|████████▏ | 139984896/170498071 [00:39<00:08, 3543480.33it/s]

.. parsed-literal::

     82%|████████▏ | 140378112/170498071 [00:39<00:08, 3543001.90it/s]

.. parsed-literal::

     83%|████████▎ | 140771328/170498071 [00:40<00:08, 3470629.56it/s]

.. parsed-literal::

     83%|████████▎ | 141164544/170498071 [00:40<00:08, 3492644.95it/s]

.. parsed-literal::

     83%|████████▎ | 141557760/170498071 [00:40<00:08, 3521598.59it/s]

.. parsed-literal::

     83%|████████▎ | 141950976/170498071 [00:40<00:08, 3527360.23it/s]

.. parsed-literal::

     83%|████████▎ | 142344192/170498071 [00:40<00:07, 3531615.14it/s]

.. parsed-literal::

     84%|████████▎ | 142737408/170498071 [00:40<00:07, 3536090.04it/s]

.. parsed-literal::

     84%|████████▍ | 143130624/170498071 [00:40<00:07, 3539379.67it/s]

.. parsed-literal::

     84%|████████▍ | 143523840/170498071 [00:40<00:07, 3541309.10it/s]

.. parsed-literal::

     84%|████████▍ | 143917056/170498071 [00:40<00:07, 3554961.29it/s]

.. parsed-literal::

     85%|████████▍ | 144310272/170498071 [00:41<00:07, 3547996.99it/s]

.. parsed-literal::

     85%|████████▍ | 144703488/170498071 [00:41<00:07, 3561871.66it/s]

.. parsed-literal::

     85%|████████▌ | 145096704/170498071 [00:41<00:07, 3552281.02it/s]

.. parsed-literal::

     85%|████████▌ | 145457152/170498071 [00:41<00:07, 3564571.33it/s]

.. parsed-literal::

     86%|████████▌ | 145850368/170498071 [00:41<00:06, 3558392.71it/s]

.. parsed-literal::

     86%|████████▌ | 146243584/170498071 [00:41<00:06, 3554636.64it/s]

.. parsed-literal::

     86%|████████▌ | 146636800/170498071 [00:41<00:06, 3549606.79it/s]

.. parsed-literal::

     86%|████████▌ | 147030016/170498071 [00:41<00:06, 3547905.56it/s]

.. parsed-literal::

     86%|████████▋ | 147423232/170498071 [00:41<00:06, 3561789.76it/s]

.. parsed-literal::

     87%|████████▋ | 147816448/170498071 [00:41<00:06, 3553690.34it/s]

.. parsed-literal::

     87%|████████▋ | 148209664/170498071 [00:42<00:06, 3549629.63it/s]

.. parsed-literal::

     87%|████████▋ | 148602880/170498071 [00:42<00:06, 3548236.64it/s]

.. parsed-literal::

     87%|████████▋ | 148996096/170498071 [00:42<00:06, 3548457.32it/s]

.. parsed-literal::

     88%|████████▊ | 149389312/170498071 [00:42<00:05, 3548341.24it/s]

.. parsed-literal::

     88%|████████▊ | 149782528/170498071 [00:42<00:05, 3562493.02it/s]

.. parsed-literal::

     88%|████████▊ | 150175744/170498071 [00:42<00:05, 3556637.31it/s]

.. parsed-literal::

     88%|████████▊ | 150568960/170498071 [00:42<00:05, 3552272.97it/s]

.. parsed-literal::

     89%|████████▊ | 150962176/170498071 [00:42<00:05, 3563906.82it/s]

.. parsed-literal::

     89%|████████▉ | 151355392/170498071 [00:42<00:05, 3571352.29it/s]

.. parsed-literal::

     89%|████████▉ | 151748608/170498071 [00:43<00:05, 3564194.24it/s]

.. parsed-literal::

     89%|████████▉ | 152109056/170498071 [00:43<00:05, 3574236.74it/s]

.. parsed-literal::

     89%|████████▉ | 152502272/170498071 [00:43<00:05, 3566238.90it/s]

.. parsed-literal::

     90%|████████▉ | 152895488/170498071 [00:43<00:04, 3573805.81it/s]

.. parsed-literal::

     90%|████████▉ | 153288704/170498071 [00:43<00:04, 3539817.49it/s]

.. parsed-literal::

     90%|█████████ | 153681920/170498071 [00:43<00:04, 3541920.95it/s]

.. parsed-literal::

     90%|█████████ | 154075136/170498071 [00:43<00:04, 3542102.18it/s]

.. parsed-literal::

     91%|█████████ | 154468352/170498071 [00:43<00:04, 3541889.91it/s]

.. parsed-literal::

     91%|█████████ | 154861568/170498071 [00:43<00:04, 3554412.58it/s]

.. parsed-literal::

     91%|█████████ | 155254784/170498071 [00:44<00:04, 3551188.08it/s]

.. parsed-literal::

     91%|█████████▏| 155648000/170498071 [00:44<00:04, 3545452.05it/s]

.. parsed-literal::

     92%|█████████▏| 156041216/170498071 [00:44<00:04, 3559154.25it/s]

.. parsed-literal::

     92%|█████████▏| 156434432/170498071 [00:44<00:03, 3556938.93it/s]

.. parsed-literal::

     92%|█████████▏| 156827648/170498071 [00:44<00:03, 3555533.02it/s]

.. parsed-literal::

     92%|█████████▏| 157220864/170498071 [00:44<00:03, 3552773.69it/s]

.. parsed-literal::

     92%|█████████▏| 157614080/170498071 [00:44<00:03, 3565047.88it/s]

.. parsed-literal::

     93%|█████████▎| 157974528/170498071 [00:44<00:03, 3561898.14it/s]

.. parsed-literal::

     93%|█████████▎| 158367744/170498071 [00:44<00:03, 3556335.97it/s]

.. parsed-literal::

     93%|█████████▎| 158760960/170498071 [00:45<00:03, 3548957.55it/s]

.. parsed-literal::

     93%|█████████▎| 159154176/170498071 [00:45<00:03, 3547748.97it/s]

.. parsed-literal::

     94%|█████████▎| 159547392/170498071 [00:45<00:03, 3547360.10it/s]

.. parsed-literal::

     94%|█████████▍| 159940608/170498071 [00:45<00:02, 3559828.38it/s]

.. parsed-literal::

     94%|█████████▍| 160333824/170498071 [00:45<00:02, 3559406.98it/s]

.. parsed-literal::

     94%|█████████▍| 160727040/170498071 [00:45<00:02, 3526942.41it/s]

.. parsed-literal::

     94%|█████████▍| 161120256/170498071 [00:45<00:02, 3531013.66it/s]

.. parsed-literal::

     95%|█████████▍| 161513472/170498071 [00:45<00:02, 3551388.82it/s]

.. parsed-literal::

     95%|█████████▍| 161873920/170498071 [00:45<00:02, 3532859.36it/s]

.. parsed-literal::

     95%|█████████▌| 162267136/170498071 [00:46<00:02, 3536903.87it/s]

.. parsed-literal::

     95%|█████████▌| 162660352/170498071 [00:46<00:02, 3539128.20it/s]

.. parsed-literal::

     96%|█████████▌| 163053568/170498071 [00:46<00:02, 3541574.54it/s]

.. parsed-literal::

     96%|█████████▌| 163446784/170498071 [00:46<00:01, 3541224.77it/s]

.. parsed-literal::

     96%|█████████▌| 163840000/170498071 [00:46<00:01, 3539641.56it/s]

.. parsed-literal::

     96%|█████████▋| 164233216/170498071 [00:46<00:01, 3540892.26it/s]

.. parsed-literal::

     97%|█████████▋| 164626432/170498071 [00:46<00:01, 3556394.45it/s]

.. parsed-literal::

     97%|█████████▋| 165019648/170498071 [00:46<00:01, 3553834.30it/s]

.. parsed-literal::

     97%|█████████▋| 165412864/170498071 [00:46<00:01, 3550888.80it/s]

.. parsed-literal::

     97%|█████████▋| 165806080/170498071 [00:47<00:01, 3562674.82it/s]

.. parsed-literal::

     97%|█████████▋| 166199296/170498071 [00:47<00:01, 3570832.69it/s]

.. parsed-literal::

     98%|█████████▊| 166592512/170498071 [00:47<00:01, 3557184.20it/s]

.. parsed-literal::

     98%|█████████▊| 166985728/170498071 [00:47<00:00, 3568438.74it/s]

.. parsed-literal::

     98%|█████████▊| 167378944/170498071 [00:47<00:00, 3564578.27it/s]

.. parsed-literal::

     98%|█████████▊| 167772160/170498071 [00:47<00:00, 3560268.66it/s]

.. parsed-literal::

     99%|█████████▊| 168165376/170498071 [00:47<00:00, 3555086.68it/s]

.. parsed-literal::

     99%|█████████▉| 168558592/170498071 [00:47<00:00, 3567776.00it/s]

.. parsed-literal::

     99%|█████████▉| 168919040/170498071 [00:47<00:00, 3573773.94it/s]

.. parsed-literal::

     99%|█████████▉| 169312256/170498071 [00:48<00:00, 3530104.46it/s]

.. parsed-literal::

    100%|█████████▉| 169705472/170498071 [00:48<00:00, 3547694.43it/s]

.. parsed-literal::

    100%|█████████▉| 170098688/170498071 [00:48<00:00, 3548674.51it/s]

.. parsed-literal::

    100%|█████████▉| 170491904/170498071 [00:48<00:00, 3548967.00it/s]

.. parsed-literal::

    100%|██████████| 170498071/170498071 [00:48<00:00, 3524459.22it/s]

.. parsed-literal::

    


.. parsed-literal::

    Extracting data/cifar-10-python.tar.gz to data


Perform Quantization
--------------------

`back to top ⬆️ <#Table-of-contents:>`__

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
MobileNetV2. The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``openvino.save_model``
   function.

Create Dataset for Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

NNCF is compatible with ``torch.utils.data.DataLoader`` interface. For
performing quantization it should be passed into ``nncf.Dataset`` object
with transformation function, which prepares input data to fit into
model during quantization, in our case, to pick input tensor from pair
(input tensor and label) and convert PyTorch tensor to numpy.

.. code:: ipython3

    import nncf
    
    def transform_fn(data_item):
        image_tensor = data_item[0]
        return image_tensor.numpy()
    
    quantization_dataset = nncf.Dataset(val_loader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Run nncf.quantize for Getting an Optimized Model
------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

``nncf.quantize`` function accepts model and prepared quantization
dataset for performing basic quantization. Optionally, additional
parameters like ``subset_size``, ``preset``, ``ignored_scope`` can be
provided to improve quantization result if applicable. More details
about supported parameters can be found on this
`page <https://docs.openvino.ai/2023.3/basic_quantization_flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)


.. parsed-literal::

    2024-01-25 23:03:23.248680: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-25 23:03:23.308103: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-25 23:03:23.827518: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Serialize an OpenVINO IR model
------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Similar to ``ov.convert_model``, quantized model is ``ov.Model`` object
which ready to be loaded into device and can be serialized on disk using
``ov.save_model``.

.. code:: ipython3

    ov.save_model(quant_ov_model, MODEL_DIR / "quantized_mobilenet_v2.xml")

Compare Accuracy of the Original and Quantized Models
-----------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    from tqdm.notebook import tqdm
    import numpy as np
    
    def test_accuracy(ov_model, data_loader):
        correct = 0
        total = 0
        for (batch_imgs, batch_labels) in tqdm(data_loader):
            result = ov_model(batch_imgs)[0]
            top_label = np.argmax(result)
            correct += top_label == batch_labels.numpy()
            total += 1
        return correct / total

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

`back to top ⬆️ <#Table-of-contents:>`__

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

    core = ov.Core()
    compiled_model = core.compile_model(ov_model, device.value)
    optimized_compiled_model = core.compile_model(quant_ov_model, device.value)
    
    orig_accuracy = test_accuracy(compiled_model, val_loader)
    optimized_accuracy = test_accuracy(optimized_compiled_model, val_loader)



.. parsed-literal::

      0%|          | 0/10000 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/10000 [00:00<?, ?it/s]


.. code:: ipython3

    print(f"Accuracy of the original model: {orig_accuracy[0] * 100 :.2f}%")
    print(f"Accuracy of the optimized model: {optimized_accuracy[0] * 100 :.2f}%")


.. parsed-literal::

    Accuracy of the original model: 93.61%
    Accuracy of the optimized model: 93.54%


Compare Performance of the Original and Quantized Models
--------------------------------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2023.3/openvino_sample_benchmark_tool.html>`__
- an inference performance measurement tool in OpenVINO.

   **NOTE**: For more accurate performance, it is recommended to run
   benchmark_app in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change CPU to GPU to benchmark
   on GPU. Run ``benchmark_app --help`` to see an overview of all
   command-line options.

.. code:: ipython3

    # Inference FP16 model (OpenVINO IR)
    !benchmark_app -m "model/mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
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
    [ INFO ] Read model took 9.59 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     x.17 (node: aten::linear/Add) : f32 / [...] / [1,10]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     x.17 (node: aten::linear/Add) : f32 / [...] / [1,10]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 215.86 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model2
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
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
    [ INFO ]     NETWORK_NAME: Model2
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 3.33 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            88704 iterations
    [ INFO ] Duration:         15002.65 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.84 ms
    [ INFO ]    Average:       1.85 ms
    [ INFO ]    Min:           1.19 ms
    [ INFO ]    Max:           8.90 ms
    [ INFO ] Throughput:   5912.55 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m "model/quantized_mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
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
    [ INFO ] Read model took 19.08 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     x.17 (node: aten::linear/Add) : f32 / [...] / [1,10]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,32,32]
    [ INFO ] Model outputs:
    [ INFO ]     x.17 (node: aten::linear/Add) : f32 / [...] / [1,10]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 331.25 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model2
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
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
    [ INFO ]     NETWORK_NAME: Model2
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.87 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            167340 iterations
    [ INFO ] Duration:         15001.30 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.00 ms
    [ INFO ]    Average:       1.03 ms
    [ INFO ]    Min:           0.72 ms
    [ INFO ]    Max:           7.10 ms
    [ INFO ] Throughput:   11155.04 FPS


Compare results on four pictures
--------------------------------

`back to top ⬆️ <#Table-of-contents:>`__

.. code:: ipython3

    # Define all possible labels from the CIFAR10 dataset
    labels_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    all_pictures = []
    all_labels = []
    
    # Get all pictures and their labels.
    for i, batch in enumerate(val_loader):
        all_pictures.append(batch[0].numpy())
        all_labels.append(batch[1].item())

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    def plot_pictures(indexes: list, all_pictures=all_pictures, all_labels=all_labels):
        """Plot 4 pictures.
        :param indexes: a list of indexes of pictures to be displayed.
        :param all_batches: batches with pictures.
        """
        images, labels = [], []
        num_pics = len(indexes)
        assert num_pics == 4, f'No enough indexes for pictures to be displayed, got {num_pics}'
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            pic = np.rollaxis(all_pictures[idx].squeeze(), 0, 3)
            images.append(pic)
    
            labels.append(labels_names[all_labels[idx]])
    
        f, axarr = plt.subplots(1, 4)
        axarr[0].imshow(images[0])
        axarr[0].set_title(labels[0])
    
        axarr[1].imshow(images[1])
        axarr[1].set_title(labels[1])
    
        axarr[2].imshow(images[2])
        axarr[2].set_title(labels[2])
    
        axarr[3].imshow(images[3])
        axarr[3].set_title(labels[3])

.. code:: ipython3

    def infer_on_pictures(model, indexes: list, all_pictures=all_pictures):
        """ Inference model on a few pictures.
        :param net: model on which do inference
        :param indexes: list of indexes 
        """
        output_key = model.output(0)
        predicted_labels = []
        for idx in indexes:
            assert idx < 10000, 'Cannot get such index, there are only 10000'
            result = model(all_pictures[idx])[output_key]
            result = labels_names[np.argmax(result[0])]
            predicted_labels.append(result)
        return predicted_labels

.. code:: ipython3

    indexes_to_infer = [7, 12, 15, 20]  # To plot, specify 4 indexes.
    
    plot_pictures(indexes_to_infer)
    
    results_float = infer_on_pictures(compiled_model, indexes_to_infer)
    results_quanized = infer_on_pictures(optimized_compiled_model, indexes_to_infer)
    
    print(f"Labels for picture from float model : {results_float}.")
    print(f"Labels for picture from quantized model : {results_quanized}.")


.. parsed-literal::

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Labels for picture from float model : ['frog', 'dog', 'ship', 'horse'].
    Labels for picture from quantized model : ['frog', 'dog', 'ship', 'horse'].



.. image:: 113-image-classification-quantization-with-output_files/113-image-classification-quantization-with-output_30_5.png

