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

-  `Prepare the Model <#prepare-the-model>`__
-  `Prepare Dataset <#prepare-dataset>`__
-  `Perform Quantization <#perform-quantization>`__

   -  `Create Dataset for Validation <#create-dataset-for-validation>`__

-  `Run nncf.quantize for Getting an Optimized
   Model <#run-nncf-quantize-for-getting-an-optimized-model>`__
-  `Serialize an OpenVINO IR model <#serialize-an-openvino-ir-model>`__
-  `Compare Accuracy of the Original and Quantized
   Models <#compare-accuracy-of-the-original-and-quantized-models>`__

   -  `Select inference device <#select-inference-device>`__

-  `Compare Performance of the Original and Quantized
   Models <#compare-performance-of-the-original-and-quantized-models>`__
-  `Compare results on four
   pictures <#compare-results-on-four-pictures>`__

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
    remote: Counting objects:   0% (1/281)[K
remote: Counting objects:   1% (3/281)[K
remote: Counting objects:   2% (6/281)[K
remote: Counting objects:   3% (9/281)[K
remote: Counting objects:   4% (12/281)[K
remote: Counting objects:   5% (15/281)[K
remote: Counting objects:   6% (17/281)[K
remote: Counting objects:   7% (20/281)[K
remote: Counting objects:   8% (23/281)[K
remote: Counting objects:   9% (26/281)[K
remote: Counting objects:  10% (29/281)[K
remote: Counting objects:  11% (31/281)[K
remote: Counting objects:  12% (34/281)[K
remote: Counting objects:  13% (37/281)[K
remote: Counting objects:  14% (40/281)[K
remote: Counting objects:  15% (43/281)[K
remote: Counting objects:  16% (45/281)[K
remote: Counting objects:  17% (48/281)[K
remote: Counting objects:  18% (51/281)[K
remote: Counting objects:  19% (54/281)[K
remote: Counting objects:  20% (57/281)[K
remote: Counting objects:  21% (60/281)[K
remote: Counting objects:  22% (62/281)[K
remote: Counting objects:  23% (65/281)[K
remote: Counting objects:  24% (68/281)[K
remote: Counting objects:  25% (71/281)[K
remote: Counting objects:  26% (74/281)[K
remote: Counting objects:  27% (76/281)[K
remote: Counting objects:  28% (79/281)[K
remote: Counting objects:  29% (82/281)[K
remote: Counting objects:  30% (85/281)[K
remote: Counting objects:  31% (88/281)[K
remote: Counting objects:  32% (90/281)[K
remote: Counting objects:  33% (93/281)[K
remote: Counting objects:  34% (96/281)[K
remote: Counting objects:  35% (99/281)[K
remote: Counting objects:  36% (102/281)[K
remote: Counting objects:  37% (104/281)[K
remote: Counting objects:  38% (107/281)[K
remote: Counting objects:  39% (110/281)[K
remote: Counting objects:  40% (113/281)[K
remote: Counting objects:  41% (116/281)[K
remote: Counting objects:  42% (119/281)[K
remote: Counting objects:  43% (121/281)[K
remote: Counting objects:  44% (124/281)[K
remote: Counting objects:  45% (127/281)[K
remote: Counting objects:  46% (130/281)[K
remote: Counting objects:  47% (133/281)[K
remote: Counting objects:  48% (135/281)[K
remote: Counting objects:  49% (138/281)[K
remote: Counting objects:  50% (141/281)[K
remote: Counting objects:  51% (144/281)[K
remote: Counting objects:  52% (147/281)[K
remote: Counting objects:  53% (149/281)[K
remote: Counting objects:  54% (152/281)[K
remote: Counting objects:  55% (155/281)[K
remote: Counting objects:  56% (158/281)[K
remote: Counting objects:  57% (161/281)[K
remote: Counting objects:  58% (163/281)[K
remote: Counting objects:  59% (166/281)[K
remote: Counting objects:  60% (169/281)[K
remote: Counting objects:  61% (172/281)[K
remote: Counting objects:  62% (175/281)[K
remote: Counting objects:  63% (178/281)[K
remote: Counting objects:  64% (180/281)[K
remote: Counting objects:  65% (183/281)[K
remote: Counting objects:  66% (186/281)[K
remote: Counting objects:  67% (189/281)[K
remote: Counting objects:  68% (192/281)[K
remote: Counting objects:  69% (194/281)[K
remote: Counting objects:  70% (197/281)[K
remote: Counting objects:  71% (200/281)[K
remote: Counting objects:  72% (203/281)[K
remote: Counting objects:  73% (206/281)[K
remote: Counting objects:  74% (208/281)[K
remote: Counting objects:  75% (211/281)[K
remote: Counting objects:  76% (214/281)[K
remote: Counting objects:  77% (217/281)[K
remote: Counting objects:  78% (220/281)[K
remote: Counting objects:  79% (222/281)[K
remote: Counting objects:  80% (225/281)[K
remote: Counting objects:  81% (228/281)[K
remote: Counting objects:  82% (231/281)[K
remote: Counting objects:  83% (234/281)[K
remote: Counting objects:  84% (237/281)[K
remote: Counting objects:  85% (239/281)[K
remote: Counting objects:  86% (242/281)[K
remote: Counting objects:  87% (245/281)[K
remote: Counting objects:  88% (248/281)[K
remote: Counting objects:  89% (251/281)[K
remote: Counting objects:  90% (253/281)[K
remote: Counting objects:  91% (256/281)[K
remote: Counting objects:  92% (259/281)[K
remote: Counting objects:  93% (262/281)[K
remote: Counting objects:  94% (265/281)[K
remote: Counting objects:  95% (267/281)[K
remote: Counting objects:  96% (270/281)[K
remote: Counting objects:  97% (273/281)[K
remote: Counting objects:  98% (276/281)[K
remote: Counting objects:  99% (279/281)[K
remote: Counting objects: 100% (281/281)[K
remote: Counting objects: 100% (281/281), done.[K
    remote: Compressing objects:   1% (1/96)[K
remote: Compressing objects:   2% (2/96)[K
remote: Compressing objects:   3% (3/96)[K
remote: Compressing objects:   4% (4/96)[K
remote: Compressing objects:   5% (5/96)[K
remote: Compressing objects:   6% (6/96)[K
remote: Compressing objects:   7% (7/96)[K
remote: Compressing objects:   8% (8/96)[K
remote: Compressing objects:   9% (9/96)[K
remote: Compressing objects:  10% (10/96)[K
remote: Compressing objects:  11% (11/96)[K
remote: Compressing objects:  12% (12/96)[K
remote: Compressing objects:  13% (13/96)[K
remote: Compressing objects:  14% (14/96)[K
remote: Compressing objects:  15% (15/96)[K
remote: Compressing objects:  16% (16/96)[K
remote: Compressing objects:  17% (17/96)[K
remote: Compressing objects:  18% (18/96)[K
remote: Compressing objects:  19% (19/96)[K
remote: Compressing objects:  20% (20/96)[K
remote: Compressing objects:  21% (21/96)[K
remote: Compressing objects:  22% (22/96)[K
remote: Compressing objects:  23% (23/96)[K
remote: Compressing objects:  25% (24/96)[K
remote: Compressing objects:  26% (25/96)[K
remote: Compressing objects:  27% (26/96)[K
remote: Compressing objects:  28% (27/96)[K
remote: Compressing objects:  29% (28/96)[K
remote: Compressing objects:  30% (29/96)[K
remote: Compressing objects:  31% (30/96)[K
remote: Compressing objects:  32% (31/96)[K
remote: Compressing objects:  33% (32/96)[K
remote: Compressing objects:  34% (33/96)[K
remote: Compressing objects:  35% (34/96)[K
remote: Compressing objects:  36% (35/96)[K
remote: Compressing objects:  37% (36/96)[K
remote: Compressing objects:  38% (37/96)[K
remote: Compressing objects:  39% (38/96)[K
remote: Compressing objects:  40% (39/96)[K
remote: Compressing objects:  41% (40/96)[K
remote: Compressing objects:  42% (41/96)[K
remote: Compressing objects:  43% (42/96)[K
remote: Compressing objects:  44% (43/96)[K
remote: Compressing objects:  45% (44/96)[K
remote: Compressing objects:  46% (45/96)[K
remote: Compressing objects:  47% (46/96)[K
remote: Compressing objects:  48% (47/96)[K
remote: Compressing objects:  50% (48/96)[K
remote: Compressing objects:  51% (49/96)[K
remote: Compressing objects:  52% (50/96)[K
remote: Compressing objects:  53% (51/96)[K
remote: Compressing objects:  54% (52/96)[K
remote: Compressing objects:  55% (53/96)[K
remote: Compressing objects:  56% (54/96)[K
remote: Compressing objects:  57% (55/96)[K
remote: Compressing objects:  58% (56/96)[K
remote: Compressing objects:  59% (57/96)[K
remote: Compressing objects:  60% (58/96)[K
remote: Compressing objects:  61% (59/96)[K
remote: Compressing objects:  62% (60/96)[K
remote: Compressing objects:  63% (61/96)[K
remote: Compressing objects:  64% (62/96)[K
remote: Compressing objects:  65% (63/96)[K
remote: Compressing objects:  66% (64/96)[K
remote: Compressing objects:  67% (65/96)[K
remote: Compressing objects:  68% (66/96)[K
remote: Compressing objects:  69% (67/96)[K
remote: Compressing objects:  70% (68/96)[K
remote: Compressing objects:  71% (69/96)[K
remote: Compressing objects:  72% (70/96)[K
remote: Compressing objects:  73% (71/96)[K
remote: Compressing objects:  75% (72/96)[K
remote: Compressing objects:  76% (73/96)[K
remote: Compressing objects:  77% (74/96)[K
remote: Compressing objects:  78% (75/96)[K
remote: Compressing objects:  79% (76/96)[K
remote: Compressing objects:  80% (77/96)[K
remote: Compressing objects:  81% (78/96)[K
remote: Compressing objects:  82% (79/96)[K
remote: Compressing objects:  83% (80/96)[K
remote: Compressing objects:  84% (81/96)[K
remote: Compressing objects:  85% (82/96)[K
remote: Compressing objects:  86% (83/96)[K
remote: Compressing objects:  87% (84/96)[K
remote: Compressing objects:  88% (85/96)[K
remote: Compressing objects:  89% (86/96)[K
remote: Compressing objects:  90% (87/96)[K
remote: Compressing objects:  91% (88/96)[K
remote: Compressing objects:  92% (89/96)[K
remote: Compressing objects:  93% (90/96)[K
remote: Compressing objects:  94% (91/96)[K
remote: Compressing objects:  95% (92/96)[K
remote: Compressing objects:  96% (93/96)[K
remote: Compressing objects:  97% (94/96)[K
remote: Compressing objects:  98% (95/96)[K
remote: Compressing objects: 100% (96/96)[K
remote: Compressing objects: 100% (96/96), done.[K
    Receiving objects:   0% (1/282)
Receiving objects:   1% (3/282)
Receiving objects:   2% (6/282)
Receiving objects:   3% (9/282)
Receiving objects:   4% (12/282)
Receiving objects:   5% (15/282)
Receiving objects:   6% (17/282)
Receiving objects:   7% (20/282)
Receiving objects:   8% (23/282)
Receiving objects:   9% (26/282)
Receiving objects:  10% (29/282)
Receiving objects:  11% (32/282)
Receiving objects:  12% (34/282)
Receiving objects:  13% (37/282)
Receiving objects:  14% (40/282)
Receiving objects:  15% (43/282)
Receiving objects:  16% (46/282)
Receiving objects:  17% (48/282)
Receiving objects:  18% (51/282)
Receiving objects:  19% (54/282)
Receiving objects:  20% (57/282)
Receiving objects:  21% (60/282)
Receiving objects:  22% (63/282)
Receiving objects:  23% (65/282)
Receiving objects:  24% (68/282)
Receiving objects:  25% (71/282)
Receiving objects:  26% (74/282)
Receiving objects:  27% (77/282)
Receiving objects:  28% (79/282)
Receiving objects:  29% (82/282)
Receiving objects:  30% (85/282)
Receiving objects:  31% (88/282)
Receiving objects:  32% (91/282)
Receiving objects:  33% (94/282)
Receiving objects:  34% (96/282)
Receiving objects:  35% (99/282)
Receiving objects:  36% (102/282)
Receiving objects:  37% (105/282)
Receiving objects:  38% (108/282)
Receiving objects:  39% (110/282)
Receiving objects:  40% (113/282)
Receiving objects:  41% (116/282)
Receiving objects:  42% (119/282)
Receiving objects:  43% (122/282)
Receiving objects:  44% (125/282)
Receiving objects:  45% (127/282)
Receiving objects:  46% (130/282)
Receiving objects:  47% (133/282)
Receiving objects:  48% (136/282)
Receiving objects:  49% (139/282)
Receiving objects:  50% (141/282)
Receiving objects:  51% (144/282)
Receiving objects:  52% (147/282)
Receiving objects:  53% (150/282)
Receiving objects:  54% (153/282)
Receiving objects:  55% (156/282)
Receiving objects:  56% (158/282)
Receiving objects:  57% (161/282)
Receiving objects:  58% (164/282)
Receiving objects:  59% (167/282)
Receiving objects:  60% (170/282)
Receiving objects:  61% (173/282)
Receiving objects:  62% (175/282)
Receiving objects:  63% (178/282)
Receiving objects:  64% (181/282)
Receiving objects:  65% (184/282)
Receiving objects:  66% (187/282)
Receiving objects:  67% (189/282)
Receiving objects:  68% (192/282)
Receiving objects:  69% (195/282)
Receiving objects:  70% (198/282)
Receiving objects:  71% (201/282)
Receiving objects:  72% (204/282)
Receiving objects:  73% (206/282)
Receiving objects:  74% (209/282)
Receiving objects:  75% (212/282)

.. parsed-literal::

    Receiving objects:  76% (215/282)

.. parsed-literal::

    Receiving objects:  77% (218/282)

.. parsed-literal::

    Receiving objects:  78% (220/282)

.. parsed-literal::

    Receiving objects:  79% (223/282), 1.71 MiB | 3.35 MiB/s

.. parsed-literal::

    Receiving objects:  80% (226/282), 1.71 MiB | 3.35 MiB/s

.. parsed-literal::

    Receiving objects:  80% (228/282), 1.71 MiB | 3.35 MiB/s

.. parsed-literal::

    Receiving objects:  81% (229/282), 3.54 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  82% (232/282), 3.54 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  83% (235/282), 3.54 MiB | 3.45 MiB/s

.. parsed-literal::

    Receiving objects:  84% (237/282), 5.36 MiB | 3.48 MiB/s

.. parsed-literal::

    Receiving objects:  85% (240/282), 5.36 MiB | 3.48 MiB/s

.. parsed-literal::

    Receiving objects:  86% (243/282), 5.36 MiB | 3.48 MiB/s

.. parsed-literal::

    Receiving objects:  86% (243/282), 7.20 MiB | 3.49 MiB/s

.. parsed-literal::

    Receiving objects:  87% (246/282), 7.20 MiB | 3.49 MiB/s

.. parsed-literal::

    Receiving objects:  88% (249/282), 7.20 MiB | 3.49 MiB/s

.. parsed-literal::

    Receiving objects:  89% (251/282), 7.20 MiB | 3.49 MiB/s

.. parsed-literal::

    remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects:  90% (254/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  91% (257/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  92% (260/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  93% (263/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  94% (266/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  95% (268/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  96% (271/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  97% (274/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  98% (277/282), 9.03 MiB | 3.50 MiB/s
Receiving objects:  99% (280/282), 9.03 MiB | 3.50 MiB/s
Receiving objects: 100% (282/282), 9.03 MiB | 3.50 MiB/s
Receiving objects: 100% (282/282), 9.22 MiB | 3.52 MiB/s, done.
    Resolving deltas:   0% (0/135)
Resolving deltas:   2% (3/135)
Resolving deltas:   5% (8/135)
Resolving deltas:   6% (9/135)
Resolving deltas:  12% (17/135)
Resolving deltas:  17% (24/135)
Resolving deltas:  23% (32/135)
Resolving deltas:  27% (37/135)
Resolving deltas:  28% (39/135)
Resolving deltas:  29% (40/135)
Resolving deltas:  30% (41/135)
Resolving deltas:  31% (42/135)
Resolving deltas:  32% (44/135)
Resolving deltas:  34% (47/135)
Resolving deltas:  40% (54/135)
Resolving deltas:  45% (61/135)
Resolving deltas:  47% (64/135)
Resolving deltas:  51% (69/135)
Resolving deltas:  57% (78/135)
Resolving deltas:  58% (79/135)
Resolving deltas:  60% (81/135)
Resolving deltas:  62% (84/135)
Resolving deltas:  71% (96/135)

.. parsed-literal::

    Resolving deltas: 100% (135/135)
Resolving deltas: 100% (135/135), done.


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


  0%|          | 32768/170498071 [00:00<17:05, 166212.70it/s]

.. parsed-literal::


  0%|          | 98304/170498071 [00:00<07:47, 364627.91it/s]

.. parsed-literal::


  0%|          | 196608/170498071 [00:00<04:49, 587907.81it/s]

.. parsed-literal::


  0%|          | 393216/170498071 [00:00<02:42, 1049548.10it/s]

.. parsed-literal::


  0%|          | 786432/170498071 [00:00<01:26, 1970714.89it/s]

.. parsed-literal::


  1%|          | 1179648/170498071 [00:00<01:06, 2549180.54it/s]

.. parsed-literal::


  1%|          | 1572864/170498071 [00:00<00:57, 2941902.04it/s]

.. parsed-literal::


  1%|          | 1966080/170498071 [00:00<00:52, 3207502.78it/s]

.. parsed-literal::


  1%|▏         | 2359296/170498071 [00:01<00:49, 3373409.86it/s]

.. parsed-literal::


  2%|▏         | 2752512/170498071 [00:01<00:48, 3484764.01it/s]

.. parsed-literal::


  2%|▏         | 3145728/170498071 [00:01<00:46, 3564118.02it/s]

.. parsed-literal::


  2%|▏         | 3538944/170498071 [00:01<00:45, 3634473.45it/s]

.. parsed-literal::


  2%|▏         | 3932160/170498071 [00:01<00:45, 3681580.62it/s]

.. parsed-literal::


  3%|▎         | 4325376/170498071 [00:01<00:44, 3700893.16it/s]

.. parsed-literal::


  3%|▎         | 4718592/170498071 [00:01<00:44, 3727400.96it/s]

.. parsed-literal::


  3%|▎         | 5111808/170498071 [00:01<00:44, 3729318.04it/s]

.. parsed-literal::


  3%|▎         | 5505024/170498071 [00:01<00:44, 3730061.67it/s]

.. parsed-literal::


  3%|▎         | 5898240/170498071 [00:01<00:43, 3750969.28it/s]

.. parsed-literal::


  4%|▎         | 6291456/170498071 [00:02<00:43, 3749865.11it/s]

.. parsed-literal::


  4%|▍         | 6684672/170498071 [00:02<00:43, 3763008.02it/s]

.. parsed-literal::


  4%|▍         | 7077888/170498071 [00:02<00:43, 3760280.87it/s]

.. parsed-literal::


  4%|▍         | 7471104/170498071 [00:02<00:43, 3772836.07it/s]

.. parsed-literal::


  5%|▍         | 7864320/170498071 [00:02<00:43, 3763036.38it/s]

.. parsed-literal::


  5%|▍         | 8257536/170498071 [00:02<00:43, 3756139.25it/s]

.. parsed-literal::


  5%|▌         | 8650752/170498071 [00:02<00:42, 3772264.41it/s]

.. parsed-literal::


  5%|▌         | 9043968/170498071 [00:02<00:42, 3761271.41it/s]

.. parsed-literal::


  6%|▌         | 9437184/170498071 [00:02<00:42, 3758117.66it/s]

.. parsed-literal::


  6%|▌         | 9830400/170498071 [00:03<00:42, 3771903.27it/s]

.. parsed-literal::


  6%|▌         | 10223616/170498071 [00:03<00:42, 3765693.38it/s]

.. parsed-literal::


  6%|▌         | 10616832/170498071 [00:03<00:42, 3760355.48it/s]

.. parsed-literal::


  6%|▋         | 11010048/170498071 [00:03<00:42, 3771889.89it/s]

.. parsed-literal::


  7%|▋         | 11403264/170498071 [00:03<00:42, 3765896.00it/s]

.. parsed-literal::


  7%|▋         | 11796480/170498071 [00:03<00:42, 3770225.19it/s]

.. parsed-literal::


  7%|▋         | 12189696/170498071 [00:03<00:42, 3761499.01it/s]

.. parsed-literal::


  7%|▋         | 12582912/170498071 [00:03<00:41, 3772024.31it/s]

.. parsed-literal::


  8%|▊         | 12976128/170498071 [00:03<00:43, 3646344.25it/s]

.. parsed-literal::


  8%|▊         | 13369344/170498071 [00:03<00:42, 3689742.42it/s]

.. parsed-literal::


  8%|▊         | 13762560/170498071 [00:04<00:42, 3722174.24it/s]

.. parsed-literal::


  8%|▊         | 14155776/170498071 [00:04<00:41, 3729804.42it/s]

.. parsed-literal::


  9%|▊         | 14548992/170498071 [00:04<00:41, 3748900.65it/s]

.. parsed-literal::


  9%|▉         | 14942208/170498071 [00:04<00:41, 3747256.60it/s]

.. parsed-literal::


  9%|▉         | 15335424/170498071 [00:04<00:41, 3760600.26it/s]

.. parsed-literal::


  9%|▉         | 15728640/170498071 [00:04<00:41, 3756190.99it/s]

.. parsed-literal::


  9%|▉         | 16121856/170498071 [00:04<00:40, 3769291.22it/s]

.. parsed-literal::


 10%|▉         | 16515072/170498071 [00:04<00:40, 3765129.18it/s]

.. parsed-literal::


 10%|▉         | 16908288/170498071 [00:04<00:40, 3775219.04it/s]

.. parsed-literal::


 10%|█         | 17301504/170498071 [00:05<00:40, 3768045.55it/s]

.. parsed-literal::


 10%|█         | 17694720/170498071 [00:05<00:40, 3779356.16it/s]

.. parsed-literal::


 11%|█         | 18087936/170498071 [00:05<00:40, 3767156.95it/s]

.. parsed-literal::


 11%|█         | 18481152/170498071 [00:05<00:40, 3774439.30it/s]

.. parsed-literal::


 11%|█         | 18874368/170498071 [00:05<00:40, 3767749.65it/s]

.. parsed-literal::


 11%|█▏        | 19267584/170498071 [00:05<00:40, 3778430.90it/s]

.. parsed-literal::


 12%|█▏        | 19660800/170498071 [00:05<00:39, 3772794.54it/s]

.. parsed-literal::


 12%|█▏        | 20054016/170498071 [00:05<00:39, 3764866.27it/s]

.. parsed-literal::


 12%|█▏        | 20447232/170498071 [00:05<00:39, 3776940.26it/s]

.. parsed-literal::


 12%|█▏        | 20840448/170498071 [00:05<00:39, 3766939.32it/s]

.. parsed-literal::


 12%|█▏        | 21233664/170498071 [00:06<00:39, 3778488.77it/s]

.. parsed-literal::


 13%|█▎        | 21626880/170498071 [00:06<00:39, 3767888.28it/s]

.. parsed-literal::


 13%|█▎        | 22020096/170498071 [00:06<00:39, 3775536.50it/s]

.. parsed-literal::


 13%|█▎        | 22413312/170498071 [00:06<00:39, 3784667.31it/s]

.. parsed-literal::


 13%|█▎        | 22806528/170498071 [00:06<00:39, 3771695.19it/s]

.. parsed-literal::


 14%|█▎        | 23199744/170498071 [00:06<00:38, 3781098.36it/s]

.. parsed-literal::


 14%|█▍        | 23592960/170498071 [00:06<00:38, 3768526.04it/s]

.. parsed-literal::


 14%|█▍        | 23986176/170498071 [00:06<00:38, 3763839.52it/s]

.. parsed-literal::


 14%|█▍        | 24379392/170498071 [00:06<00:38, 3762730.51it/s]

.. parsed-literal::


 15%|█▍        | 24772608/170498071 [00:06<00:38, 3759140.43it/s]

.. parsed-literal::


 15%|█▍        | 25165824/170498071 [00:07<00:38, 3754317.43it/s]

.. parsed-literal::


 15%|█▍        | 25559040/170498071 [00:07<00:38, 3755740.60it/s]

.. parsed-literal::


 15%|█▌        | 25952256/170498071 [00:07<00:38, 3749437.85it/s]

.. parsed-literal::


 15%|█▌        | 26345472/170498071 [00:07<00:38, 3750633.95it/s]

.. parsed-literal::


 16%|█▌        | 26738688/170498071 [00:07<00:38, 3751863.38it/s]

.. parsed-literal::


 16%|█▌        | 27131904/170498071 [00:07<00:38, 3749632.52it/s]

.. parsed-literal::


 16%|█▌        | 27525120/170498071 [00:07<00:38, 3750481.16it/s]

.. parsed-literal::


 16%|█▋        | 27918336/170498071 [00:07<00:37, 3767501.75it/s]

.. parsed-literal::


 17%|█▋        | 28311552/170498071 [00:07<00:37, 3762282.12it/s]

.. parsed-literal::


 17%|█▋        | 28704768/170498071 [00:08<00:37, 3757208.73it/s]

.. parsed-literal::


 17%|█▋        | 29097984/170498071 [00:08<00:37, 3767048.10it/s]

.. parsed-literal::


 17%|█▋        | 29491200/170498071 [00:08<00:37, 3775503.33it/s]

.. parsed-literal::


 18%|█▊        | 29884416/170498071 [00:08<00:37, 3777539.29it/s]

.. parsed-literal::


 18%|█▊        | 30277632/170498071 [00:08<00:37, 3771522.68it/s]

.. parsed-literal::


 18%|█▊        | 30670848/170498071 [00:08<00:37, 3768097.10it/s]

.. parsed-literal::


 18%|█▊        | 31064064/170498071 [00:08<00:37, 3760340.19it/s]

.. parsed-literal::


 18%|█▊        | 31457280/170498071 [00:08<00:36, 3772414.30it/s]

.. parsed-literal::


 19%|█▊        | 31850496/170498071 [00:08<00:36, 3763910.45it/s]

.. parsed-literal::


 19%|█▉        | 32243712/170498071 [00:08<00:36, 3759185.71it/s]

.. parsed-literal::


 19%|█▉        | 32636928/170498071 [00:09<00:36, 3752188.93it/s]

.. parsed-literal::


 19%|█▉        | 33030144/170498071 [00:09<00:36, 3766797.06it/s]

.. parsed-literal::


 20%|█▉        | 33423360/170498071 [00:09<00:36, 3761136.47it/s]

.. parsed-literal::


 20%|█▉        | 33816576/170498071 [00:09<00:36, 3756244.61it/s]

.. parsed-literal::


 20%|██        | 34209792/170498071 [00:09<00:36, 3768659.77it/s]

.. parsed-literal::


 20%|██        | 34603008/170498071 [00:09<00:36, 3762173.64it/s]

.. parsed-literal::


 21%|██        | 34996224/170498071 [00:09<00:35, 3774047.75it/s]

.. parsed-literal::


 21%|██        | 35389440/170498071 [00:09<00:35, 3767572.06it/s]

.. parsed-literal::


 21%|██        | 35782656/170498071 [00:09<00:35, 3779984.10it/s]

.. parsed-literal::


 21%|██        | 36175872/170498071 [00:10<00:35, 3771377.51it/s]

.. parsed-literal::


 21%|██▏       | 36569088/170498071 [00:10<00:35, 3761817.99it/s]

.. parsed-literal::


 22%|██▏       | 36962304/170498071 [00:10<00:35, 3769026.33it/s]

.. parsed-literal::


 22%|██▏       | 37355520/170498071 [00:10<00:35, 3763490.51it/s]

.. parsed-literal::


 22%|██▏       | 37748736/170498071 [00:10<00:35, 3775898.32it/s]

.. parsed-literal::


 22%|██▏       | 38141952/170498071 [00:10<00:35, 3765847.42it/s]

.. parsed-literal::


 23%|██▎       | 38535168/170498071 [00:10<00:34, 3778934.96it/s]

.. parsed-literal::


 23%|██▎       | 38928384/170498071 [00:10<00:34, 3764725.55it/s]

.. parsed-literal::


 23%|██▎       | 39321600/170498071 [00:10<00:34, 3775183.84it/s]

.. parsed-literal::


 23%|██▎       | 39714816/170498071 [00:10<00:34, 3784262.94it/s]

.. parsed-literal::


 24%|██▎       | 40108032/170498071 [00:11<00:34, 3770568.22it/s]

.. parsed-literal::


 24%|██▍       | 40501248/170498071 [00:11<00:34, 3764569.11it/s]

.. parsed-literal::


 24%|██▍       | 40894464/170498071 [00:11<00:34, 3761003.64it/s]

.. parsed-literal::


 24%|██▍       | 41287680/170498071 [00:11<00:34, 3756783.29it/s]

.. parsed-literal::


 24%|██▍       | 41680896/170498071 [00:11<00:34, 3753770.60it/s]

.. parsed-literal::


 25%|██▍       | 42074112/170498071 [00:11<00:34, 3769468.00it/s]

.. parsed-literal::


 25%|██▍       | 42467328/170498071 [00:11<00:33, 3781392.65it/s]

.. parsed-literal::


 25%|██▌       | 42860544/170498071 [00:11<00:33, 3774928.49it/s]

.. parsed-literal::


 25%|██▌       | 43253760/170498071 [00:11<00:33, 3766898.03it/s]

.. parsed-literal::


 26%|██▌       | 43646976/170498071 [00:11<00:33, 3777327.77it/s]

.. parsed-literal::


 26%|██▌       | 44040192/170498071 [00:12<00:33, 3765844.02it/s]

.. parsed-literal::


 26%|██▌       | 44433408/170498071 [00:12<00:33, 3775911.40it/s]

.. parsed-literal::


 26%|██▋       | 44826624/170498071 [00:12<00:33, 3782290.53it/s]

.. parsed-literal::


 27%|██▋       | 45219840/170498071 [00:12<00:33, 3772901.52it/s]

.. parsed-literal::


 27%|██▋       | 45613056/170498071 [00:12<00:33, 3781793.44it/s]

.. parsed-literal::


 27%|██▋       | 46006272/170498071 [00:12<00:33, 3714231.21it/s]

.. parsed-literal::


 27%|██▋       | 46399488/170498071 [00:12<00:33, 3739005.08it/s]

.. parsed-literal::


 27%|██▋       | 46792704/170498071 [00:12<00:33, 3739775.61it/s]

.. parsed-literal::


 28%|██▊       | 47185920/170498071 [00:12<00:32, 3741206.05it/s]

.. parsed-literal::


 28%|██▊       | 47579136/170498071 [00:13<00:32, 3760037.28it/s]

.. parsed-literal::


 28%|██▊       | 47972352/170498071 [00:13<00:32, 3757286.61it/s]

.. parsed-literal::


 28%|██▊       | 48365568/170498071 [00:13<00:32, 3769642.07it/s]

.. parsed-literal::


 29%|██▊       | 48758784/170498071 [00:13<00:32, 3761617.83it/s]

.. parsed-literal::


 29%|██▉       | 49152000/170498071 [00:13<00:32, 3751118.48it/s]

.. parsed-literal::


 29%|██▉       | 49545216/170498071 [00:13<00:32, 3764275.43it/s]

.. parsed-literal::


 29%|██▉       | 49938432/170498071 [00:13<00:32, 3760927.09it/s]

.. parsed-literal::


 30%|██▉       | 50331648/170498071 [00:13<00:31, 3758265.61it/s]

.. parsed-literal::


 30%|██▉       | 50724864/170498071 [00:13<00:31, 3756510.05it/s]

.. parsed-literal::


 30%|██▉       | 51118080/170498071 [00:13<00:31, 3767986.62it/s]

.. parsed-literal::


 30%|███       | 51511296/170498071 [00:14<00:31, 3762906.44it/s]

.. parsed-literal::


 30%|███       | 51904512/170498071 [00:14<00:31, 3754549.63it/s]

.. parsed-literal::


 31%|███       | 52297728/170498071 [00:14<00:31, 3767516.91it/s]

.. parsed-literal::


 31%|███       | 52690944/170498071 [00:14<00:31, 3763114.22it/s]

.. parsed-literal::


 31%|███       | 53084160/170498071 [00:14<00:31, 3775150.93it/s]

.. parsed-literal::


 31%|███▏      | 53477376/170498071 [00:14<00:31, 3768186.59it/s]

.. parsed-literal::


 32%|███▏      | 53870592/170498071 [00:14<00:31, 3761714.83it/s]

.. parsed-literal::


 32%|███▏      | 54263808/170498071 [00:14<00:30, 3772328.82it/s]

.. parsed-literal::


 32%|███▏      | 54657024/170498071 [00:14<00:30, 3761829.12it/s]

.. parsed-literal::


 32%|███▏      | 55050240/170498071 [00:15<00:30, 3758328.18it/s]

.. parsed-literal::


 33%|███▎      | 55443456/170498071 [00:15<00:30, 3753068.63it/s]

.. parsed-literal::


 33%|███▎      | 55836672/170498071 [00:15<00:30, 3768905.29it/s]

.. parsed-literal::


 33%|███▎      | 56229888/170498071 [00:15<00:30, 3778844.32it/s]

.. parsed-literal::


 33%|███▎      | 56623104/170498071 [00:15<00:30, 3768435.39it/s]

.. parsed-literal::


 33%|███▎      | 57016320/170498071 [00:15<00:30, 3762990.47it/s]

.. parsed-literal::


 34%|███▎      | 57409536/170498071 [00:15<00:30, 3758085.96it/s]

.. parsed-literal::


 34%|███▍      | 57802752/170498071 [00:15<00:29, 3756772.02it/s]

.. parsed-literal::


 34%|███▍      | 58195968/170498071 [00:15<00:29, 3769499.15it/s]

.. parsed-literal::


 34%|███▍      | 58589184/170498071 [00:15<00:29, 3764506.06it/s]

.. parsed-literal::


 35%|███▍      | 58982400/170498071 [00:16<00:29, 3760897.84it/s]

.. parsed-literal::


 35%|███▍      | 59375616/170498071 [00:16<00:29, 3756765.88it/s]

.. parsed-literal::


 35%|███▌      | 59768832/170498071 [00:16<00:29, 3751757.73it/s]

.. parsed-literal::


 35%|███▌      | 60162048/170498071 [00:16<00:29, 3766301.90it/s]

.. parsed-literal::


 36%|███▌      | 60555264/170498071 [00:16<00:29, 3760145.25it/s]

.. parsed-literal::


 36%|███▌      | 60948480/170498071 [00:16<00:29, 3714212.38it/s]

.. parsed-literal::


 36%|███▌      | 61341696/170498071 [00:16<00:29, 3739248.58it/s]

.. parsed-literal::


 36%|███▌      | 61734912/170498071 [00:16<00:29, 3738806.65it/s]

.. parsed-literal::


 36%|███▋      | 62128128/170498071 [00:16<00:29, 3736675.43it/s]

.. parsed-literal::


 37%|███▋      | 62521344/170498071 [00:17<00:28, 3753159.93it/s]

.. parsed-literal::


 37%|███▋      | 62914560/170498071 [00:17<00:28, 3768827.64it/s]

.. parsed-literal::


 37%|███▋      | 63307776/170498071 [00:17<00:28, 3763594.01it/s]

.. parsed-literal::


 37%|███▋      | 63700992/170498071 [00:17<00:28, 3776920.71it/s]

.. parsed-literal::


 38%|███▊      | 64094208/170498071 [00:17<00:28, 3769133.84it/s]

.. parsed-literal::


 38%|███▊      | 64487424/170498071 [00:17<00:28, 3765620.13it/s]

.. parsed-literal::


 38%|███▊      | 64880640/170498071 [00:17<00:28, 3758135.19it/s]

.. parsed-literal::


 38%|███▊      | 65273856/170498071 [00:17<00:27, 3770537.32it/s]

.. parsed-literal::


 39%|███▊      | 65667072/170498071 [00:17<00:27, 3775732.41it/s]

.. parsed-literal::


 39%|███▊      | 66060288/170498071 [00:17<00:27, 3768003.19it/s]

.. parsed-literal::


 39%|███▉      | 66453504/170498071 [00:18<00:27, 3779523.79it/s]

.. parsed-literal::


 39%|███▉      | 66846720/170498071 [00:18<00:27, 3768753.36it/s]

.. parsed-literal::


 39%|███▉      | 67239936/170498071 [00:18<00:27, 3759491.17it/s]

.. parsed-literal::


 40%|███▉      | 67633152/170498071 [00:18<00:27, 3771748.85it/s]

.. parsed-literal::


 40%|███▉      | 68026368/170498071 [00:18<00:27, 3767640.49it/s]

.. parsed-literal::


 40%|████      | 68419584/170498071 [00:18<00:27, 3762587.54it/s]

.. parsed-literal::


 40%|████      | 68812800/170498071 [00:18<00:26, 3777158.11it/s]

.. parsed-literal::


 41%|████      | 69206016/170498071 [00:18<00:26, 3769433.72it/s]

.. parsed-literal::


 41%|████      | 69599232/170498071 [00:18<00:26, 3781594.79it/s]

.. parsed-literal::


 41%|████      | 69992448/170498071 [00:19<00:26, 3767066.03it/s]

.. parsed-literal::


 41%|████▏     | 70385664/170498071 [00:19<00:26, 3763214.03it/s]

.. parsed-literal::


 42%|████▏     | 70778880/170498071 [00:19<00:29, 3434526.08it/s]

.. parsed-literal::


 42%|████▏     | 71172096/170498071 [00:19<00:28, 3520058.99it/s]

.. parsed-literal::


 42%|████▏     | 71565312/170498071 [00:19<00:27, 3598620.04it/s]

.. parsed-literal::


 42%|████▏     | 71958528/170498071 [00:19<00:27, 3642036.73it/s]

.. parsed-literal::


 42%|████▏     | 72351744/170498071 [00:19<00:27, 3506407.73it/s]

.. parsed-literal::


 43%|████▎     | 72744960/170498071 [00:19<00:27, 3576702.92it/s]

.. parsed-literal::


 43%|████▎     | 73138176/170498071 [00:19<00:26, 3625529.10it/s]

.. parsed-literal::


 43%|████▎     | 73531392/170498071 [00:19<00:26, 3662237.13it/s]

.. parsed-literal::


 43%|████▎     | 73924608/170498071 [00:20<00:26, 3702844.17it/s]

.. parsed-literal::


 44%|████▎     | 74317824/170498071 [00:20<00:25, 3716503.21it/s]

.. parsed-literal::


 44%|████▍     | 74711040/170498071 [00:20<00:25, 3726051.42it/s]

.. parsed-literal::


 44%|████▍     | 75104256/170498071 [00:20<00:25, 3731609.04it/s]

.. parsed-literal::


 44%|████▍     | 75497472/170498071 [00:20<00:25, 3753031.19it/s]

.. parsed-literal::


 45%|████▍     | 75890688/170498071 [00:20<00:25, 3767804.32it/s]

.. parsed-literal::


 45%|████▍     | 76283904/170498071 [00:20<00:25, 3764425.58it/s]

.. parsed-literal::


 45%|████▍     | 76677120/170498071 [00:20<00:24, 3764655.74it/s]

.. parsed-literal::


 45%|████▌     | 77070336/170498071 [00:20<00:24, 3759945.21it/s]

.. parsed-literal::


 45%|████▌     | 77463552/170498071 [00:21<00:24, 3758807.26it/s]

.. parsed-literal::


 46%|████▌     | 77856768/170498071 [00:21<00:24, 3748640.09it/s]

.. parsed-literal::


 46%|████▌     | 78249984/170498071 [00:21<00:24, 3766890.10it/s]

.. parsed-literal::


 46%|████▌     | 78643200/170498071 [00:21<00:24, 3761306.91it/s]

.. parsed-literal::


 46%|████▋     | 79036416/170498071 [00:21<00:24, 3760142.93it/s]

.. parsed-literal::


 47%|████▋     | 79429632/170498071 [00:21<00:24, 3774041.47it/s]

.. parsed-literal::


 47%|████▋     | 79822848/170498071 [00:21<00:24, 3769210.53it/s]

.. parsed-literal::


 47%|████▋     | 80216064/170498071 [00:21<00:23, 3763848.40it/s]

.. parsed-literal::


 47%|████▋     | 80609280/170498071 [00:21<00:23, 3756304.19it/s]

.. parsed-literal::


 48%|████▊     | 81002496/170498071 [00:21<00:23, 3754976.54it/s]

.. parsed-literal::


 48%|████▊     | 81395712/170498071 [00:22<00:23, 3752958.58it/s]

.. parsed-literal::


 48%|████▊     | 81788928/170498071 [00:22<00:23, 3754252.67it/s]

.. parsed-literal::


 48%|████▊     | 82182144/170498071 [00:22<00:23, 3753966.73it/s]

.. parsed-literal::


 48%|████▊     | 82575360/170498071 [00:22<00:23, 3755066.53it/s]

.. parsed-literal::


 49%|████▊     | 82968576/170498071 [00:22<00:23, 3752191.57it/s]

.. parsed-literal::


 49%|████▉     | 83361792/170498071 [00:22<00:23, 3765676.55it/s]

.. parsed-literal::


 49%|████▉     | 83755008/170498071 [00:22<00:22, 3777129.55it/s]

.. parsed-literal::


 49%|████▉     | 84148224/170498071 [00:22<00:22, 3770375.49it/s]

.. parsed-literal::


 50%|████▉     | 84541440/170498071 [00:22<00:22, 3763334.28it/s]

.. parsed-literal::


 50%|████▉     | 84934656/170498071 [00:23<00:22, 3778076.87it/s]

.. parsed-literal::


 50%|█████     | 85327872/170498071 [00:23<00:22, 3768337.58it/s]

.. parsed-literal::


 50%|█████     | 85721088/170498071 [00:23<00:22, 3760973.47it/s]

.. parsed-literal::


 51%|█████     | 86114304/170498071 [00:23<00:22, 3768936.65it/s]

.. parsed-literal::


 51%|█████     | 86507520/170498071 [00:23<00:22, 3765521.04it/s]

.. parsed-literal::


 51%|█████     | 86900736/170498071 [00:23<00:22, 3781085.49it/s]

.. parsed-literal::


 51%|█████     | 87293952/170498071 [00:23<00:22, 3772124.21it/s]

.. parsed-literal::


 51%|█████▏    | 87687168/170498071 [00:23<00:21, 3770274.83it/s]

.. parsed-literal::


 52%|█████▏    | 88080384/170498071 [00:23<00:21, 3762316.32it/s]

.. parsed-literal::


 52%|█████▏    | 88473600/170498071 [00:23<00:21, 3758455.29it/s]

.. parsed-literal::


 52%|█████▏    | 88866816/170498071 [00:24<00:21, 3771236.23it/s]

.. parsed-literal::


 52%|█████▏    | 89260032/170498071 [00:24<00:21, 3759615.27it/s]

.. parsed-literal::


 53%|█████▎    | 89653248/170498071 [00:24<00:21, 3755436.36it/s]

.. parsed-literal::


 53%|█████▎    | 90046464/170498071 [00:24<00:21, 3756036.73it/s]

.. parsed-literal::


 53%|█████▎    | 90439680/170498071 [00:24<00:21, 3753971.55it/s]

.. parsed-literal::


 53%|█████▎    | 90832896/170498071 [00:24<00:21, 3749829.49it/s]

.. parsed-literal::


 54%|█████▎    | 91226112/170498071 [00:24<00:21, 3750905.71it/s]

.. parsed-literal::


 54%|█████▎    | 91619328/170498071 [00:24<00:21, 3750858.25it/s]

.. parsed-literal::


 54%|█████▍    | 92012544/170498071 [00:24<00:20, 3764478.61it/s]

.. parsed-literal::


 54%|█████▍    | 92405760/170498071 [00:25<00:20, 3763960.85it/s]

.. parsed-literal::


 54%|█████▍    | 92798976/170498071 [00:25<00:20, 3761463.76it/s]

.. parsed-literal::


 55%|█████▍    | 93192192/170498071 [00:25<00:20, 3757538.63it/s]

.. parsed-literal::


 55%|█████▍    | 93585408/170498071 [00:25<00:20, 3755975.95it/s]

.. parsed-literal::


 55%|█████▌    | 93978624/170498071 [00:25<00:20, 3771495.25it/s]

.. parsed-literal::


 55%|█████▌    | 94371840/170498071 [00:25<00:20, 3765226.23it/s]

.. parsed-literal::


 56%|█████▌    | 94765056/170498071 [00:25<00:20, 3773957.83it/s]

.. parsed-literal::


 56%|█████▌    | 95158272/170498071 [00:25<00:19, 3768632.79it/s]

.. parsed-literal::


 56%|█████▌    | 95551488/170498071 [00:25<00:19, 3778052.69it/s]

.. parsed-literal::


 56%|█████▋    | 95944704/170498071 [00:25<00:19, 3767096.80it/s]

.. parsed-literal::


 57%|█████▋    | 96337920/170498071 [00:26<00:19, 3764377.05it/s]

.. parsed-literal::


 57%|█████▋    | 96731136/170498071 [00:26<00:19, 3774184.37it/s]

.. parsed-literal::


 57%|█████▋    | 97124352/170498071 [00:26<00:19, 3769760.04it/s]

.. parsed-literal::


 57%|█████▋    | 97517568/170498071 [00:26<00:19, 3779783.57it/s]

.. parsed-literal::


 57%|█████▋    | 97910784/170498071 [00:26<00:19, 3772285.81it/s]

.. parsed-literal::


 58%|█████▊    | 98304000/170498071 [00:26<00:19, 3761935.61it/s]

.. parsed-literal::


 58%|█████▊    | 98697216/170498071 [00:26<00:19, 3772621.44it/s]

.. parsed-literal::


 58%|█████▊    | 99090432/170498071 [00:26<00:18, 3782197.43it/s]

.. parsed-literal::


 58%|█████▊    | 99483648/170498071 [00:26<00:18, 3775287.65it/s]

.. parsed-literal::


 59%|█████▊    | 99876864/170498071 [00:26<00:18, 3769302.49it/s]

.. parsed-literal::


 59%|█████▉    | 100270080/170498071 [00:27<00:18, 3766604.86it/s]

.. parsed-literal::


 59%|█████▉    | 100663296/170498071 [00:27<00:18, 3764020.30it/s]

.. parsed-literal::


 59%|█████▉    | 101056512/170498071 [00:27<00:18, 3758876.86it/s]

.. parsed-literal::


 60%|█████▉    | 101449728/170498071 [00:27<00:18, 3772144.06it/s]

.. parsed-literal::


 60%|█████▉    | 101842944/170498071 [00:27<00:18, 3767449.16it/s]

.. parsed-literal::


 60%|█████▉    | 102236160/170498071 [00:27<00:18, 3762042.02it/s]

.. parsed-literal::


 60%|██████    | 102629376/170498071 [00:27<00:18, 3760734.34it/s]

.. parsed-literal::


 60%|██████    | 103022592/170498071 [00:27<00:17, 3758986.56it/s]

.. parsed-literal::


 61%|██████    | 103415808/170498071 [00:27<00:17, 3757751.24it/s]

.. parsed-literal::


 61%|██████    | 103809024/170498071 [00:28<00:17, 3754123.99it/s]

.. parsed-literal::


 61%|██████    | 104202240/170498071 [00:28<00:17, 3765993.66it/s]

.. parsed-literal::


 61%|██████▏   | 104595456/170498071 [00:28<00:17, 3763108.69it/s]

.. parsed-literal::


 62%|██████▏   | 104988672/170498071 [00:28<00:17, 3762183.16it/s]

.. parsed-literal::


 62%|██████▏   | 105381888/170498071 [00:28<00:17, 3761381.14it/s]

.. parsed-literal::


 62%|██████▏   | 105775104/170498071 [00:28<00:17, 3755504.47it/s]

.. parsed-literal::


 62%|██████▏   | 106168320/170498071 [00:28<00:17, 3751194.37it/s]

.. parsed-literal::


 63%|██████▎   | 106561536/170498071 [00:28<00:17, 3749042.02it/s]

.. parsed-literal::


 63%|██████▎   | 106954752/170498071 [00:28<00:16, 3749824.60it/s]

.. parsed-literal::


 63%|██████▎   | 107347968/170498071 [00:28<00:16, 3766206.98it/s]

.. parsed-literal::


 63%|██████▎   | 107741184/170498071 [00:29<00:16, 3758937.53it/s]

.. parsed-literal::


 63%|██████▎   | 108134400/170498071 [00:29<00:16, 3773359.69it/s]

.. parsed-literal::


 64%|██████▎   | 108527616/170498071 [00:29<00:16, 3784456.27it/s]

.. parsed-literal::


 64%|██████▍   | 108920832/170498071 [00:29<00:16, 3767335.66it/s]

.. parsed-literal::


 64%|██████▍   | 109314048/170498071 [00:29<00:16, 3764732.20it/s]

.. parsed-literal::


 64%|██████▍   | 109707264/170498071 [00:29<00:16, 3762049.29it/s]

.. parsed-literal::


 65%|██████▍   | 110100480/170498071 [00:29<00:16, 3759818.65it/s]

.. parsed-literal::


 65%|██████▍   | 110493696/170498071 [00:29<00:15, 3757842.60it/s]

.. parsed-literal::


 65%|██████▌   | 110886912/170498071 [00:29<00:15, 3772059.24it/s]

.. parsed-literal::


 65%|██████▌   | 111280128/170498071 [00:30<00:15, 3762349.40it/s]

.. parsed-literal::


 65%|██████▌   | 111673344/170498071 [00:30<00:15, 3761649.31it/s]

.. parsed-literal::


 66%|██████▌   | 112066560/170498071 [00:30<00:15, 3773569.59it/s]

.. parsed-literal::


 66%|██████▌   | 112459776/170498071 [00:30<00:15, 3768188.72it/s]

.. parsed-literal::


 66%|██████▌   | 112852992/170498071 [00:30<00:15, 3763856.50it/s]

.. parsed-literal::


 66%|██████▋   | 113246208/170498071 [00:30<00:15, 3776955.24it/s]

.. parsed-literal::


 67%|██████▋   | 113639424/170498071 [00:30<00:15, 3771366.07it/s]

.. parsed-literal::


 67%|██████▋   | 114032640/170498071 [00:30<00:15, 3760610.88it/s]

.. parsed-literal::


 67%|██████▋   | 114425856/170498071 [00:30<00:14, 3770951.43it/s]

.. parsed-literal::


 67%|██████▋   | 114819072/170498071 [00:30<00:14, 3780278.74it/s]

.. parsed-literal::


 68%|██████▊   | 115212288/170498071 [00:31<00:14, 3775268.41it/s]

.. parsed-literal::


 68%|██████▊   | 115605504/170498071 [00:31<00:14, 3769343.33it/s]

.. parsed-literal::


 68%|██████▊   | 115998720/170498071 [00:31<00:14, 3767588.51it/s]

.. parsed-literal::


 68%|██████▊   | 116391936/170498071 [00:31<00:14, 3761135.48it/s]

.. parsed-literal::


 68%|██████▊   | 116785152/170498071 [00:31<00:14, 3757717.65it/s]

.. parsed-literal::


 69%|██████▊   | 117178368/170498071 [00:31<00:14, 3771344.91it/s]

.. parsed-literal::


 69%|██████▉   | 117571584/170498071 [00:31<00:14, 3763180.64it/s]

.. parsed-literal::


 69%|██████▉   | 117964800/170498071 [00:31<00:13, 3772459.49it/s]

.. parsed-literal::


 69%|██████▉   | 118358016/170498071 [00:31<00:13, 3780387.80it/s]

.. parsed-literal::


 70%|██████▉   | 118751232/170498071 [00:32<00:13, 3770510.27it/s]

.. parsed-literal::


 70%|██████▉   | 119144448/170498071 [00:32<00:13, 3761031.26it/s]

.. parsed-literal::


 70%|███████   | 119537664/170498071 [00:32<00:13, 3773955.27it/s]

.. parsed-literal::


 70%|███████   | 119930880/170498071 [00:32<00:13, 3780453.60it/s]

.. parsed-literal::


 71%|███████   | 120324096/170498071 [00:32<00:13, 3785503.14it/s]

.. parsed-literal::


 71%|███████   | 120717312/170498071 [00:32<00:13, 3774809.42it/s]

.. parsed-literal::


 71%|███████   | 121110528/170498071 [00:32<00:13, 3783752.16it/s]

.. parsed-literal::


 71%|███████▏  | 121503744/170498071 [00:32<00:12, 3773196.80it/s]

.. parsed-literal::


 71%|███████▏  | 121896960/170498071 [00:32<00:12, 3764355.18it/s]

.. parsed-literal::


 72%|███████▏  | 122290176/170498071 [00:32<00:12, 3759457.66it/s]

.. parsed-literal::


 72%|███████▏  | 122683392/170498071 [00:33<00:12, 3773581.52it/s]

.. parsed-literal::


 72%|███████▏  | 123076608/170498071 [00:33<00:12, 3764600.00it/s]

.. parsed-literal::


 72%|███████▏  | 123469824/170498071 [00:33<00:12, 3760261.19it/s]

.. parsed-literal::


 73%|███████▎  | 123863040/170498071 [00:33<00:12, 3755010.11it/s]

.. parsed-literal::


 73%|███████▎  | 124256256/170498071 [00:33<00:12, 3748225.29it/s]

.. parsed-literal::


 73%|███████▎  | 124649472/170498071 [00:33<00:12, 3763922.67it/s]

.. parsed-literal::


 73%|███████▎  | 125042688/170498071 [00:33<00:12, 3757115.85it/s]

.. parsed-literal::


 74%|███████▎  | 125435904/170498071 [00:33<00:11, 3755192.82it/s]

.. parsed-literal::


 74%|███████▍  | 125829120/170498071 [00:33<00:11, 3751982.78it/s]

.. parsed-literal::


 74%|███████▍  | 126222336/170498071 [00:33<00:11, 3766878.74it/s]

.. parsed-literal::


 74%|███████▍  | 126615552/170498071 [00:34<00:11, 3777080.62it/s]

.. parsed-literal::


 74%|███████▍  | 127008768/170498071 [00:34<00:11, 3763215.52it/s]

.. parsed-literal::


 75%|███████▍  | 127401984/170498071 [00:34<00:11, 3764102.30it/s]

.. parsed-literal::


 75%|███████▍  | 127795200/170498071 [00:34<00:11, 3762754.66it/s]

.. parsed-literal::


 75%|███████▌  | 128188416/170498071 [00:34<00:11, 3758599.60it/s]

.. parsed-literal::


 75%|███████▌  | 128581632/170498071 [00:34<00:11, 3757914.61it/s]

.. parsed-literal::


 76%|███████▌  | 128974848/170498071 [00:34<00:11, 3772837.45it/s]

.. parsed-literal::


 76%|███████▌  | 129368064/170498071 [00:34<00:10, 3761313.15it/s]

.. parsed-literal::


 76%|███████▌  | 129761280/170498071 [00:34<00:10, 3773824.92it/s]

.. parsed-literal::


 76%|███████▋  | 130154496/170498071 [00:35<00:10, 3766665.46it/s]

.. parsed-literal::


 77%|███████▋  | 130547712/170498071 [00:35<00:10, 3775986.24it/s]

.. parsed-literal::


 77%|███████▋  | 130940928/170498071 [00:35<00:10, 3766881.43it/s]

.. parsed-literal::


 77%|███████▋  | 131334144/170498071 [00:35<00:10, 3780097.76it/s]

.. parsed-literal::


 77%|███████▋  | 131727360/170498071 [00:35<00:10, 3787877.11it/s]

.. parsed-literal::


 77%|███████▋  | 132120576/170498071 [00:35<00:10, 3765610.19it/s]

.. parsed-literal::


 78%|███████▊  | 132513792/170498071 [00:35<00:10, 3776675.43it/s]

.. parsed-literal::


 78%|███████▊  | 132907008/170498071 [00:35<00:09, 3765580.99it/s]

.. parsed-literal::


 78%|███████▊  | 133300224/170498071 [00:35<00:09, 3774671.16it/s]

.. parsed-literal::


 78%|███████▊  | 133693440/170498071 [00:35<00:09, 3764817.56it/s]

.. parsed-literal::


 79%|███████▊  | 134086656/170498071 [00:36<00:09, 3773367.37it/s]

.. parsed-literal::


 79%|███████▉  | 134479872/170498071 [00:36<00:09, 3780360.35it/s]

.. parsed-literal::


 79%|███████▉  | 134873088/170498071 [00:36<00:09, 3766982.66it/s]

.. parsed-literal::


 79%|███████▉  | 135266304/170498071 [00:36<00:09, 3763673.62it/s]

.. parsed-literal::


 80%|███████▉  | 135659520/170498071 [00:36<00:09, 3758732.49it/s]

.. parsed-literal::


 80%|███████▉  | 136052736/170498071 [00:36<00:09, 3774349.71it/s]

.. parsed-literal::


 80%|████████  | 136445952/170498071 [00:36<00:08, 3784022.58it/s]

.. parsed-literal::


 80%|████████  | 136839168/170498071 [00:36<00:08, 3775320.72it/s]

.. parsed-literal::


 80%|████████  | 137232384/170498071 [00:36<00:08, 3766097.85it/s]

.. parsed-literal::


 81%|████████  | 137625600/170498071 [00:37<00:08, 3763114.54it/s]

.. parsed-literal::


 81%|████████  | 138018816/170498071 [00:37<00:08, 3760658.56it/s]

.. parsed-literal::


 81%|████████  | 138412032/170498071 [00:37<00:08, 3773977.54it/s]

.. parsed-literal::


 81%|████████▏ | 138805248/170498071 [00:37<00:08, 3768452.80it/s]

.. parsed-literal::


 82%|████████▏ | 139198464/170498071 [00:37<00:08, 3756843.86it/s]

.. parsed-literal::


 82%|████████▏ | 139591680/170498071 [00:37<00:08, 3770079.72it/s]

.. parsed-literal::


 82%|████████▏ | 139984896/170498071 [00:37<00:08, 3761868.81it/s]

.. parsed-literal::


 82%|████████▏ | 140378112/170498071 [00:37<00:08, 3760358.52it/s]

.. parsed-literal::


 83%|████████▎ | 140771328/170498071 [00:37<00:07, 3756730.56it/s]

.. parsed-literal::


 83%|████████▎ | 141164544/170498071 [00:37<00:07, 3769958.48it/s]

.. parsed-literal::


 83%|████████▎ | 141557760/170498071 [00:38<00:07, 3763301.05it/s]

.. parsed-literal::


 83%|████████▎ | 141950976/170498071 [00:38<00:07, 3757601.50it/s]

.. parsed-literal::


 83%|████████▎ | 142344192/170498071 [00:38<00:07, 3748301.37it/s]

.. parsed-literal::


 84%|████████▎ | 142737408/170498071 [00:38<00:07, 3746893.64it/s]

.. parsed-literal::


 84%|████████▍ | 143130624/170498071 [00:38<00:07, 3748391.41it/s]

.. parsed-literal::


 84%|████████▍ | 143523840/170498071 [00:38<00:07, 3764792.60it/s]

.. parsed-literal::


 84%|████████▍ | 143917056/170498071 [00:38<00:07, 3776628.90it/s]

.. parsed-literal::


 85%|████████▍ | 144310272/170498071 [00:38<00:06, 3766660.58it/s]

.. parsed-literal::


 85%|████████▍ | 144703488/170498071 [00:38<00:06, 3777983.47it/s]

.. parsed-literal::


 85%|████████▌ | 145096704/170498071 [00:39<00:06, 3764556.77it/s]

.. parsed-literal::


 85%|████████▌ | 145489920/170498071 [00:39<00:06, 3760238.71it/s]

.. parsed-literal::


 86%|████████▌ | 145883136/170498071 [00:39<00:06, 3771864.00it/s]

.. parsed-literal::


 86%|████████▌ | 146276352/170498071 [00:39<00:06, 3782640.25it/s]

.. parsed-literal::


 86%|████████▌ | 146669568/170498071 [00:39<00:06, 3774494.78it/s]

.. parsed-literal::


 86%|████████▋ | 147062784/170498071 [00:39<00:06, 3769793.26it/s]

.. parsed-literal::


 86%|████████▋ | 147456000/170498071 [00:39<00:06, 3763335.21it/s]

.. parsed-literal::


 87%|████████▋ | 147849216/170498071 [00:39<00:05, 3775602.18it/s]

.. parsed-literal::


 87%|████████▋ | 148242432/170498071 [00:39<00:05, 3782945.18it/s]

.. parsed-literal::


 87%|████████▋ | 148635648/170498071 [00:39<00:05, 3775031.31it/s]

.. parsed-literal::


 87%|████████▋ | 149028864/170498071 [00:40<00:05, 3767217.50it/s]

.. parsed-literal::


 88%|████████▊ | 149422080/170498071 [00:40<00:05, 3762696.51it/s]

.. parsed-literal::


 88%|████████▊ | 149815296/170498071 [00:40<00:05, 3758672.05it/s]

.. parsed-literal::


 88%|████████▊ | 150208512/170498071 [00:40<00:05, 3757179.41it/s]

.. parsed-literal::


 88%|████████▊ | 150601728/170498071 [00:40<00:05, 3768055.08it/s]

.. parsed-literal::


 89%|████████▊ | 150994944/170498071 [00:40<00:05, 3775000.39it/s]

.. parsed-literal::


 89%|████████▉ | 151388160/170498071 [00:40<00:05, 3782368.72it/s]

.. parsed-literal::


 89%|████████▉ | 151781376/170498071 [00:40<00:04, 3775736.42it/s]

.. parsed-literal::


 89%|████████▉ | 152174592/170498071 [00:40<00:04, 3766263.55it/s]

.. parsed-literal::


 89%|████████▉ | 152567808/170498071 [00:40<00:04, 3774060.16it/s]

.. parsed-literal::


 90%|████████▉ | 152961024/170498071 [00:41<00:04, 3765686.48it/s]

.. parsed-literal::


 90%|████████▉ | 153354240/170498071 [00:41<00:04, 3776775.86it/s]

.. parsed-literal::


 90%|█████████ | 153747456/170498071 [00:41<00:04, 3770524.40it/s]

.. parsed-literal::


 90%|█████████ | 154140672/170498071 [00:41<00:04, 3766421.29it/s]

.. parsed-literal::


 91%|█████████ | 154533888/170498071 [00:41<00:04, 3761749.18it/s]

.. parsed-literal::


 91%|█████████ | 154927104/170498071 [00:41<00:04, 3759462.27it/s]

.. parsed-literal::


 91%|█████████ | 155320320/170498071 [00:41<00:04, 3754211.39it/s]

.. parsed-literal::


 91%|█████████▏| 155713536/170498071 [00:41<00:03, 3769998.86it/s]

.. parsed-literal::


 92%|█████████▏| 156106752/170498071 [00:41<00:03, 3766165.14it/s]

.. parsed-literal::


 92%|█████████▏| 156499968/170498071 [00:42<00:03, 3760618.26it/s]

.. parsed-literal::


 92%|█████████▏| 156893184/170498071 [00:42<00:03, 3773490.63it/s]

.. parsed-literal::


 92%|█████████▏| 157286400/170498071 [00:42<00:03, 3748320.15it/s]

.. parsed-literal::


 92%|█████████▏| 157679616/170498071 [00:42<00:03, 3751022.81it/s]

.. parsed-literal::


 93%|█████████▎| 158072832/170498071 [00:42<00:03, 3748362.27it/s]

.. parsed-literal::


 93%|█████████▎| 158466048/170498071 [00:42<00:03, 3761330.78it/s]

.. parsed-literal::


 93%|█████████▎| 158859264/170498071 [00:42<00:03, 3758928.12it/s]

.. parsed-literal::


 93%|█████████▎| 159252480/170498071 [00:42<00:02, 3755371.93it/s]

.. parsed-literal::


 94%|█████████▎| 159645696/170498071 [00:42<00:02, 3770386.13it/s]

.. parsed-literal::


 94%|█████████▍| 160038912/170498071 [00:42<00:02, 3760785.28it/s]

.. parsed-literal::


 94%|█████████▍| 160432128/170498071 [00:43<00:02, 3754896.29it/s]

.. parsed-literal::


 94%|█████████▍| 160825344/170498071 [00:43<00:02, 3768308.76it/s]

.. parsed-literal::


 95%|█████████▍| 161218560/170498071 [00:43<00:02, 3777946.75it/s]

.. parsed-literal::


 95%|█████████▍| 161611776/170498071 [00:43<00:02, 3769947.29it/s]

.. parsed-literal::


 95%|█████████▌| 162004992/170498071 [00:43<00:02, 3765141.22it/s]

.. parsed-literal::


 95%|█████████▌| 162398208/170498071 [00:43<00:02, 3776947.12it/s]

.. parsed-literal::


 95%|█████████▌| 162791424/170498071 [00:43<00:02, 3769051.47it/s]

.. parsed-literal::


 96%|█████████▌| 163184640/170498071 [00:43<00:01, 3761803.24it/s]

.. parsed-literal::


 96%|█████████▌| 163577856/170498071 [00:43<00:01, 3772973.58it/s]

.. parsed-literal::


 96%|█████████▌| 163971072/170498071 [00:44<00:01, 3764297.59it/s]

.. parsed-literal::


 96%|█████████▋| 164364288/170498071 [00:44<00:01, 3761181.87it/s]

.. parsed-literal::


 97%|█████████▋| 164757504/170498071 [00:44<00:01, 3759060.47it/s]

.. parsed-literal::


 97%|█████████▋| 165150720/170498071 [00:44<00:01, 3756318.88it/s]

.. parsed-literal::


 97%|█████████▋| 165543936/170498071 [00:44<00:01, 3751555.73it/s]

.. parsed-literal::


 97%|█████████▋| 165937152/170498071 [00:44<00:01, 3751377.22it/s]

.. parsed-literal::


 98%|█████████▊| 166330368/170498071 [00:44<00:01, 3768697.78it/s]

.. parsed-literal::


 98%|█████████▊| 166723584/170498071 [00:44<00:01, 3762334.04it/s]

.. parsed-literal::


 98%|█████████▊| 167116800/170498071 [00:44<00:00, 3774603.85it/s]

.. parsed-literal::


 98%|█████████▊| 167510016/170498071 [00:44<00:00, 3782592.29it/s]

.. parsed-literal::


 98%|█████████▊| 167903232/170498071 [00:45<00:00, 3774653.13it/s]

.. parsed-literal::


 99%|█████████▊| 168296448/170498071 [00:45<00:00, 3766749.96it/s]

.. parsed-literal::


 99%|█████████▉| 168689664/170498071 [00:45<00:00, 3762807.79it/s]

.. parsed-literal::


 99%|█████████▉| 169082880/170498071 [00:45<00:00, 3759747.16it/s]

.. parsed-literal::


   99%|█████████▉| 169476096/170498071 [00:45<00:00, 3757142.87it/s]

.. parsed-literal::


   100%|█████████▉| 169869312/170498071 [00:45<00:00, 3755006.52it/s]

.. parsed-literal::


   100%|█████████▉| 170262528/170498071 [00:45<00:00, 3770133.46it/s]

.. parsed-literal::


   100%|██████████| 170498071/170498071 [00:45<00:00, 3727583.78it/s]



.. parsed-literal::

    Extracting data/cifar-10-python.tar.gz to data


Perform Quantization
--------------------



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



``nncf.quantize`` function accepts model and prepared quantization
dataset for performing basic quantization. Optionally, additional
parameters like ``subset_size``, ``preset``, ``ignored_scope`` can be
provided to improve quantization result if applicable. More details
about supported parameters can be found on this
`page <https://docs.openvino.ai/2023.3/basic_quantization_flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)


.. parsed-literal::

    2024-02-09 22:59:25.557733: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-09 22:59:25.614970: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-09 22:59:26.140513: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()








.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()








.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Serialize an OpenVINO IR model
------------------------------



Similar to ``ov.convert_model``, quantized model is ``ov.Model`` object
which ready to be loaded into device and can be serialized on disk using
``ov.save_model``.

.. code:: ipython3

    ov.save_model(quant_ov_model, MODEL_DIR / "quantized_mobilenet_v2.xml")

Compare Accuracy of the Original and Quantized Models
-----------------------------------------------------



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
    [ INFO ] Read model took 9.75 ms
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

    [ INFO ] Compile model took 198.44 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model2
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU


.. parsed-literal::

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
    [ INFO ] First inference took 2.83 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            88488 iterations
    [ INFO ] Duration:         15003.02 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.85 ms
    [ INFO ]    Average:       1.85 ms
    [ INFO ]    Min:           1.17 ms
    [ INFO ]    Max:           8.69 ms
    [ INFO ] Throughput:   5898.01 FPS


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
    [ INFO ] Read model took 18.77 ms
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

    [ INFO ] Compile model took 330.74 ms
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
    [ INFO ] First inference took 2.08 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            167472 iterations
    [ INFO ] Duration:         15001.04 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.00 ms
    [ INFO ]    Average:       1.03 ms
    [ INFO ]    Min:           0.68 ms
    [ INFO ]    Max:           7.03 ms
    [ INFO ] Throughput:   11164.03 FPS


Compare results on four pictures
--------------------------------



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

