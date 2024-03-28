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


.. parsed-literal::

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
    Receiving objects:  77% (218/282)

.. parsed-literal::

    Receiving objects:  78% (220/282)
    Receiving objects:  79% (223/282)
    Receiving objects:  80% (226/282)
    Receiving objects:  81% (229/282)

.. parsed-literal::

    Receiving objects:  82% (232/282)
    Receiving objects:  83% (235/282)
    Receiving objects:  84% (237/282)

.. parsed-literal::

    Receiving objects:  85% (240/282)
    Receiving objects:  86% (243/282)

.. parsed-literal::

    Receiving objects:  87% (246/282)

.. parsed-literal::

    Receiving objects:  88% (249/282)
    Receiving objects:  89% (251/282)
    remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects:  90% (254/282)
    Receiving objects:  91% (257/282)
    Receiving objects:  92% (260/282)
    Receiving objects:  93% (263/282)
    Receiving objects:  94% (266/282)
    Receiving objects:  95% (268/282)
    Receiving objects:  96% (271/282)
    Receiving objects:  97% (274/282)
    Receiving objects:  98% (277/282)
    Receiving objects:  99% (280/282)
    Receiving objects: 100% (282/282)
    Receiving objects: 100% (282/282), 9.22 MiB | 19.46 MiB/s, done.
    Resolving deltas:   0% (0/135)
    Resolving deltas:   1% (2/135)
    Resolving deltas:   3% (5/135)
    Resolving deltas:   8% (11/135)
    Resolving deltas:  10% (14/135)
    Resolving deltas:  16% (22/135)
    Resolving deltas:  20% (28/135)
    Resolving deltas:  23% (32/135)
    Resolving deltas:  25% (35/135)
    Resolving deltas:  26% (36/135)
    Resolving deltas:  27% (37/135)
    Resolving deltas:  28% (38/135)
    Resolving deltas:  29% (40/135)
    Resolving deltas:  30% (41/135)
    Resolving deltas:  31% (42/135)
    Resolving deltas:  34% (47/135)
    Resolving deltas:  40% (54/135)
    Resolving deltas:  45% (62/135)
    Resolving deltas:  47% (64/135)
    Resolving deltas:  50% (68/135)
    Resolving deltas:  57% (77/135)
    Resolving deltas:  58% (79/135)
    Resolving deltas:  59% (80/135)
    Resolving deltas:  60% (81/135)
    Resolving deltas:  61% (83/135)
    Resolving deltas:  69% (94/135)
    Resolving deltas:  71% (97/135)

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
`page <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

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


  0%|          | 32768/170498071 [00:00<10:00, 283927.24it/s]

.. parsed-literal::


  0%|          | 65536/170498071 [00:00<10:40, 265903.57it/s]

.. parsed-literal::


  0%|          | 98304/170498071 [00:00<10:18, 275319.56it/s]

.. parsed-literal::


  0%|          | 229376/170498071 [00:00<04:47, 591792.24it/s]

.. parsed-literal::


  0%|          | 458752/170498071 [00:00<02:39, 1064782.70it/s]

.. parsed-literal::


  1%|          | 884736/170498071 [00:00<01:27, 1932402.03it/s]

.. parsed-literal::


  1%|          | 1376256/170498071 [00:00<01:05, 2582519.82it/s]

.. parsed-literal::


  1%|          | 1835008/170498071 [00:00<00:55, 3022915.63it/s]

.. parsed-literal::


  1%|▏         | 2326528/170498071 [00:01<00:47, 3518057.23it/s]

.. parsed-literal::


  2%|▏         | 2752512/170498071 [00:01<00:45, 3677634.47it/s]

.. parsed-literal::


  2%|▏         | 3211264/170498071 [00:01<00:43, 3882041.66it/s]

.. parsed-literal::


  2%|▏         | 3702784/170498071 [00:01<00:40, 4126993.42it/s]

.. parsed-literal::


  2%|▏         | 4194304/170498071 [00:01<00:38, 4348789.83it/s]

.. parsed-literal::


  3%|▎         | 4718592/170498071 [00:01<00:36, 4584385.74it/s]

.. parsed-literal::


  3%|▎         | 5275648/170498071 [00:01<00:34, 4834650.62it/s]

.. parsed-literal::


  3%|▎         | 5865472/170498071 [00:01<00:32, 5080896.08it/s]

.. parsed-literal::


  4%|▍         | 6455296/170498071 [00:01<00:30, 5311689.91it/s]

.. parsed-literal::


  4%|▍         | 7077888/170498071 [00:01<00:29, 5541755.44it/s]

.. parsed-literal::


  5%|▍         | 7733248/170498071 [00:02<00:28, 5788975.59it/s]

.. parsed-literal::


  5%|▍         | 8421376/170498071 [00:02<00:26, 6024015.03it/s]

.. parsed-literal::


  5%|▌         | 9142272/170498071 [00:02<00:25, 6312114.84it/s]

.. parsed-literal::


  6%|▌         | 9895936/170498071 [00:02<00:24, 6554603.63it/s]

.. parsed-literal::


  6%|▋         | 10682368/170498071 [00:02<00:23, 6904833.33it/s]

.. parsed-literal::


  7%|▋         | 11468800/170498071 [00:02<00:22, 7128388.61it/s]

.. parsed-literal::


  7%|▋         | 12288000/170498071 [00:02<00:21, 7340568.59it/s]

.. parsed-literal::


  8%|▊         | 13139968/170498071 [00:02<00:20, 7645657.54it/s]

.. parsed-literal::


  8%|▊         | 14024704/170498071 [00:02<00:19, 7907828.21it/s]

.. parsed-literal::


  9%|▉         | 14942208/170498071 [00:03<00:18, 8235418.29it/s]

.. parsed-literal::


  9%|▉         | 15925248/170498071 [00:03<00:17, 8627022.82it/s]

.. parsed-literal::


 10%|▉         | 16941056/170498071 [00:03<00:16, 9050435.87it/s]

.. parsed-literal::


 11%|█         | 17956864/170498071 [00:03<00:16, 9324887.66it/s]

.. parsed-literal::


 11%|█         | 19070976/170498071 [00:03<00:15, 9822086.35it/s]

.. parsed-literal::


 12%|█▏        | 20250624/170498071 [00:03<00:14, 10306090.35it/s]

.. parsed-literal::


 13%|█▎        | 21495808/170498071 [00:03<00:13, 10885735.05it/s]

.. parsed-literal::


 13%|█▎        | 22773760/170498071 [00:03<00:12, 11392919.31it/s]

.. parsed-literal::


 14%|█▍        | 23920640/170498071 [00:03<00:16, 8936643.54it/s]

.. parsed-literal::


 15%|█▌        | 26214400/170498071 [00:04<00:11, 12036210.58it/s]

.. parsed-literal::


 16%|█▌        | 27525120/170498071 [00:04<00:12, 11566207.13it/s]

.. parsed-literal::


 17%|█▋        | 28770304/170498071 [00:04<00:12, 11084810.42it/s]

.. parsed-literal::


 18%|█▊        | 29949952/170498071 [00:04<00:13, 10749248.08it/s]

.. parsed-literal::


 18%|█▊        | 31064064/170498071 [00:04<00:13, 10493386.55it/s]

.. parsed-literal::


 19%|█▉        | 32145408/170498071 [00:04<00:13, 10182993.35it/s]

.. parsed-literal::


 19%|█▉        | 33193984/170498071 [00:04<00:13, 10254170.13it/s]

.. parsed-literal::


 20%|██        | 34242560/170498071 [00:04<00:13, 9977525.48it/s]

.. parsed-literal::


 21%|██        | 35422208/170498071 [00:04<00:12, 10468485.47it/s]

.. parsed-literal::


 21%|██▏       | 36503552/170498071 [00:05<00:13, 10238693.08it/s]

.. parsed-literal::


 22%|██▏       | 37683200/170498071 [00:05<00:12, 10643199.05it/s]

.. parsed-literal::


 23%|██▎       | 38764544/170498071 [00:05<00:12, 10678388.43it/s]

.. parsed-literal::


 23%|██▎       | 39878656/170498071 [00:05<00:12, 10808401.03it/s]

.. parsed-literal::


 24%|██▍       | 40992768/170498071 [00:05<00:12, 10533455.09it/s]

.. parsed-literal::


 25%|██▍       | 42106880/170498071 [00:05<00:12, 10511427.91it/s]

.. parsed-literal::


 25%|██▌       | 43253760/170498071 [00:05<00:11, 10739555.52it/s]

.. parsed-literal::


 26%|██▌       | 44498944/170498071 [00:05<00:11, 11144715.38it/s]

.. parsed-literal::


 27%|██▋       | 45645824/170498071 [00:05<00:11, 11020810.39it/s]

.. parsed-literal::


 28%|██▊       | 46891008/170498071 [00:05<00:10, 11407936.56it/s]

.. parsed-literal::


 28%|██▊       | 48070656/170498071 [00:06<00:10, 11407753.92it/s]

.. parsed-literal::


 29%|██▉       | 49283072/170498071 [00:06<00:10, 11431487.36it/s]

.. parsed-literal::


 30%|██▉       | 50429952/170498071 [00:06<00:10, 11324573.74it/s]

.. parsed-literal::


 30%|███       | 51576832/170498071 [00:06<00:10, 11168808.69it/s]

.. parsed-literal::


 31%|███       | 52822016/170498071 [00:06<00:10, 11489158.99it/s]

.. parsed-literal::


 32%|███▏      | 54067200/170498071 [00:06<00:09, 11747461.18it/s]

.. parsed-literal::


 32%|███▏      | 55312384/170498071 [00:06<00:09, 11681092.91it/s]

.. parsed-literal::


 33%|███▎      | 56590336/170498071 [00:06<00:09, 11912678.55it/s]

.. parsed-literal::


 34%|███▍      | 57802752/170498071 [00:06<00:09, 11692508.37it/s]

.. parsed-literal::


 35%|███▍      | 58982400/170498071 [00:07<00:09, 11688532.47it/s]

.. parsed-literal::


 35%|███▌      | 60162048/170498071 [00:07<00:09, 11623913.91it/s]

.. parsed-literal::


 36%|███▌      | 61440000/170498071 [00:07<00:09, 11928362.14it/s]

.. parsed-literal::


 37%|███▋      | 62685184/170498071 [00:07<00:08, 12058183.44it/s]

.. parsed-literal::


 37%|███▋      | 63897600/170498071 [00:07<00:08, 12062590.12it/s]

.. parsed-literal::


 38%|███▊      | 65110016/170498071 [00:07<00:08, 12015735.75it/s]

.. parsed-literal::


 39%|███▉      | 66322432/170498071 [00:07<00:08, 11894203.12it/s]

.. parsed-literal::


 40%|███▉      | 67600384/170498071 [00:07<00:08, 12107867.13it/s]

.. parsed-literal::


 40%|████      | 68911104/170498071 [00:07<00:08, 12391125.43it/s]

.. parsed-literal::


 41%|████      | 70156288/170498071 [00:07<00:08, 12380968.80it/s]

.. parsed-literal::


 42%|████▏     | 71401472/170498071 [00:08<00:08, 12341060.16it/s]

.. parsed-literal::


 43%|████▎     | 72646656/170498071 [00:08<00:07, 12316266.05it/s]

.. parsed-literal::


 43%|████▎     | 73891840/170498071 [00:08<00:07, 12235410.16it/s]

.. parsed-literal::


 44%|████▍     | 75169792/170498071 [00:08<00:07, 12324119.94it/s]

.. parsed-literal::


 45%|████▍     | 76546048/170498071 [00:08<00:07, 12526567.17it/s]

.. parsed-literal::


 46%|████▌     | 77856768/170498071 [00:08<00:07, 12629945.67it/s]

.. parsed-literal::


 46%|████▋     | 79200256/170498071 [00:08<00:07, 12794574.31it/s]

.. parsed-literal::


 47%|████▋     | 80510976/170498071 [00:08<00:07, 12714478.97it/s]

.. parsed-literal::


 48%|████▊     | 81821696/170498071 [00:08<00:06, 12825187.00it/s]

.. parsed-literal::


 49%|████▉     | 83132416/170498071 [00:08<00:06, 12720119.35it/s]

.. parsed-literal::


 50%|████▉     | 84410368/170498071 [00:09<00:06, 12735552.42it/s]

.. parsed-literal::


 50%|█████     | 85786624/170498071 [00:09<00:06, 12954615.41it/s]

.. parsed-literal::


 51%|█████     | 87130112/170498071 [00:09<00:06, 13066468.64it/s]

.. parsed-literal::


 52%|█████▏    | 88473600/170498071 [00:09<00:06, 13134946.81it/s]

.. parsed-literal::


 53%|█████▎    | 89817088/170498071 [00:09<00:06, 13033540.08it/s]

.. parsed-literal::


 53%|█████▎    | 91160576/170498071 [00:09<00:06, 13087445.00it/s]

.. parsed-literal::


 54%|█████▍    | 92471296/170498071 [00:09<00:06, 12975513.21it/s]

.. parsed-literal::


 55%|█████▌    | 93782016/170498071 [00:09<00:05, 12886873.79it/s]

.. parsed-literal::


 56%|█████▌    | 95158272/170498071 [00:09<00:05, 13139018.79it/s]

.. parsed-literal::


 57%|█████▋    | 96534528/170498071 [00:09<00:05, 13308395.86it/s]

.. parsed-literal::


 57%|█████▋    | 97878016/170498071 [00:10<00:05, 13322827.20it/s]

.. parsed-literal::


 58%|█████▊    | 99221504/170498071 [00:10<00:05, 13181673.04it/s]

.. parsed-literal::


 59%|█████▉    | 100597760/170498071 [00:10<00:05, 13134178.23it/s]

.. parsed-literal::


 60%|█████▉    | 101941248/170498071 [00:10<00:05, 13210607.23it/s]

.. parsed-literal::


 61%|██████    | 103284736/170498071 [00:10<00:05, 13036002.08it/s]

.. parsed-literal::


 61%|██████▏   | 104628224/170498071 [00:10<00:05, 13113089.69it/s]

.. parsed-literal::


 62%|██████▏   | 106070016/170498071 [00:10<00:04, 13400061.37it/s]

.. parsed-literal::


 63%|██████▎   | 107446272/170498071 [00:10<00:04, 13475287.69it/s]

.. parsed-literal::


 64%|██████▍   | 108822528/170498071 [00:10<00:04, 13313750.98it/s]

.. parsed-literal::


 65%|██████▍   | 110166016/170498071 [00:11<00:04, 13080989.48it/s]

.. parsed-literal::


 65%|██████▌   | 111542272/170498071 [00:11<00:04, 13234853.77it/s]

.. parsed-literal::


 66%|██████▌   | 112951296/170498071 [00:11<00:04, 13270267.01it/s]

.. parsed-literal::


 67%|██████▋   | 114294784/170498071 [00:11<00:04, 13302307.20it/s]

.. parsed-literal::


 68%|██████▊   | 115638272/170498071 [00:11<00:04, 13310513.20it/s]

.. parsed-literal::


 69%|██████▊   | 117047296/170498071 [00:11<00:03, 13521525.32it/s]

.. parsed-literal::


 69%|██████▉   | 118423552/170498071 [00:11<00:05, 9097286.49it/s]

.. parsed-literal::


 71%|███████   | 121077760/170498071 [00:11<00:03, 12880987.33it/s]

.. parsed-literal::


 72%|███████▏  | 122650624/170498071 [00:12<00:04, 11623721.84it/s]

.. parsed-literal::


 73%|███████▎  | 124026880/170498071 [00:12<00:04, 10694869.90it/s]

.. parsed-literal::


 73%|███████▎  | 125272064/170498071 [00:12<00:04, 10461454.45it/s]

.. parsed-literal::


 74%|███████▍  | 126418944/170498071 [00:12<00:04, 10269059.77it/s]

.. parsed-literal::


 75%|███████▍  | 127533056/170498071 [00:12<00:04, 10092008.14it/s]

.. parsed-literal::


 75%|███████▌  | 128614400/170498071 [00:12<00:04, 9991116.75it/s]

.. parsed-literal::


 76%|███████▌  | 129728512/170498071 [00:12<00:04, 10024635.70it/s]

.. parsed-literal::


 77%|███████▋  | 130842624/170498071 [00:12<00:03, 10264583.93it/s]

.. parsed-literal::


 77%|███████▋  | 131956736/170498071 [00:13<00:03, 10414479.41it/s]

.. parsed-literal::


 78%|███████▊  | 133038080/170498071 [00:13<00:03, 10207685.94it/s]

.. parsed-literal::


 79%|███████▊  | 134086656/170498071 [00:13<00:03, 10253117.62it/s]

.. parsed-literal::


 79%|███████▉  | 135200768/170498071 [00:13<00:03, 10267523.71it/s]

.. parsed-literal::


 80%|███████▉  | 136249344/170498071 [00:13<00:05, 6445389.32it/s]

.. parsed-literal::


 81%|████████  | 138248192/170498071 [00:13<00:03, 8999143.92it/s]

.. parsed-literal::


 82%|████████▏ | 139395072/170498071 [00:13<00:04, 7502968.08it/s]

.. parsed-literal::


 82%|████████▏ | 140345344/170498071 [00:14<00:04, 6780837.26it/s]

.. parsed-literal::


 83%|████████▎ | 141164544/170498071 [00:14<00:04, 6406113.63it/s]

.. parsed-literal::


 83%|████████▎ | 141918208/170498071 [00:14<00:04, 6136849.80it/s]

.. parsed-literal::


 84%|████████▎ | 142606336/170498071 [00:14<00:04, 5897511.08it/s]

.. parsed-literal::


 84%|████████▍ | 143261696/170498071 [00:14<00:04, 5793550.59it/s]

.. parsed-literal::


 84%|████████▍ | 143884288/170498071 [00:14<00:04, 5821252.36it/s]

.. parsed-literal::


 85%|████████▍ | 144506880/170498071 [00:14<00:04, 5670929.28it/s]

.. parsed-literal::


 85%|████████▌ | 145096704/170498071 [00:15<00:04, 5721299.33it/s]

.. parsed-literal::


 85%|████████▌ | 145686528/170498071 [00:15<00:04, 5589242.07it/s]

.. parsed-literal::


 86%|████████▌ | 146276352/170498071 [00:15<00:04, 5573544.98it/s]

.. parsed-literal::


 86%|████████▌ | 146866176/170498071 [00:15<00:04, 5653812.72it/s]

.. parsed-literal::


 86%|████████▋ | 147456000/170498071 [00:15<00:04, 5615466.08it/s]

.. parsed-literal::


 87%|████████▋ | 148078592/170498071 [00:15<00:03, 5727414.03it/s]

.. parsed-literal::


 87%|████████▋ | 148668416/170498071 [00:15<00:03, 5734945.03it/s]

.. parsed-literal::


 88%|████████▊ | 149291008/170498071 [00:15<00:03, 5720384.70it/s]

.. parsed-literal::


 88%|████████▊ | 149913600/170498071 [00:15<00:03, 5773789.44it/s]

.. parsed-literal::


 88%|████████▊ | 150536192/170498071 [00:15<00:03, 5898587.34it/s]

.. parsed-literal::


 89%|████████▊ | 151158784/170498071 [00:16<00:03, 5783102.09it/s]

.. parsed-literal::


 89%|████████▉ | 151748608/170498071 [00:16<00:03, 5744869.22it/s]

.. parsed-literal::


 89%|████████▉ | 152403968/170498071 [00:16<00:03, 5892137.14it/s]

.. parsed-literal::


 90%|████████▉ | 153026560/170498071 [00:16<00:02, 5894363.21it/s]

.. parsed-literal::


 90%|█████████ | 153649152/170498071 [00:16<00:02, 5922842.41it/s]

.. parsed-literal::


 90%|█████████ | 154271744/170498071 [00:16<00:02, 5933696.54it/s]

.. parsed-literal::


 91%|█████████ | 154927104/170498071 [00:16<00:02, 5932279.96it/s]

.. parsed-literal::


 91%|█████████ | 155549696/170498071 [00:16<00:02, 5972512.87it/s]

.. parsed-literal::


 92%|█████████▏| 156205056/170498071 [00:16<00:02, 6136941.06it/s]

.. parsed-literal::


 92%|█████████▏| 156827648/170498071 [00:17<00:02, 5963890.02it/s]

.. parsed-literal::


 92%|█████████▏| 157450240/170498071 [00:17<00:02, 5896825.04it/s]

.. parsed-literal::


 93%|█████████▎| 158105600/170498071 [00:17<00:02, 6069686.73it/s]

.. parsed-literal::


 93%|█████████▎| 158728192/170498071 [00:17<00:01, 5993375.93it/s]

.. parsed-literal::


 93%|█████████▎| 159383552/170498071 [00:17<00:01, 6099810.77it/s]

.. parsed-literal::


 94%|█████████▍| 160006144/170498071 [00:17<00:01, 6072335.15it/s]

.. parsed-literal::


   94%|█████████▍| 160661504/170498071 [00:17<00:01, 5985901.27it/s]

.. parsed-literal::


   95%|█████████▍| 161284096/170498071 [00:17<00:01, 6030540.06it/s]

.. parsed-literal::


   95%|█████████▍| 161906688/170498071 [00:17<00:01, 5995654.97it/s]

.. parsed-literal::


   95%|█████████▌| 162529280/170498071 [00:17<00:01, 5981934.93it/s]

.. parsed-literal::


   96%|█████████▌| 163151872/170498071 [00:18<00:01, 6001547.69it/s]

.. parsed-literal::


   96%|█████████▌| 163774464/170498071 [00:18<00:01, 5990972.04it/s]

.. parsed-literal::


   96%|█████████▋| 164397056/170498071 [00:18<00:01, 6055698.66it/s]

.. parsed-literal::


   97%|█████████▋| 165019648/170498071 [00:18<00:00, 6049158.95it/s]

.. parsed-literal::


   97%|█████████▋| 165642240/170498071 [00:18<00:00, 6016041.86it/s]

.. parsed-literal::


   98%|█████████▊| 166264832/170498071 [00:18<00:00, 6011049.16it/s]

.. parsed-literal::


   98%|█████████▊| 166887424/170498071 [00:18<00:00, 6054762.80it/s]

.. parsed-literal::


   98%|█████████▊| 167510016/170498071 [00:18<00:00, 6048047.89it/s]

.. parsed-literal::


   99%|█████████▊| 168132608/170498071 [00:18<00:00, 6031345.39it/s]

.. parsed-literal::


   99%|█████████▉| 168755200/170498071 [00:19<00:00, 6039672.92it/s]

.. parsed-literal::


   99%|█████████▉| 169377792/170498071 [00:19<00:00, 6034107.61it/s]

.. parsed-literal::


   100%|█████████▉| 170000384/170498071 [00:19<00:00, 6008716.05it/s]

.. parsed-literal::


   100%|██████████| 170498071/170498071 [00:19<00:00, 8827169.25it/s]






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
`page <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)


.. parsed-literal::

    2024-03-25 22:44:56.961421: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-25 22:44:56.993144: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-25 22:44:57.515616: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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
    Accuracy of the optimized model: 93.57%


Compare Performance of the Original and Quantized Models
--------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
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
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 9.79 ms
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

    [ INFO ] Compile model took 202.92 ms
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
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
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
    [ INFO ] First inference took 2.61 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            88824 iterations
    [ INFO ] Duration:         15002.32 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.84 ms
    [ INFO ]    Average:       1.85 ms
    [ INFO ]    Min:           1.18 ms
    [ INFO ]    Max:           8.74 ms
    [ INFO ] Throughput:   5920.68 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m "model/quantized_mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 19.60 ms
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

    [ INFO ] Compile model took 335.01 ms
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
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
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


.. parsed-literal::

    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.91 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            167496 iterations
    [ INFO ] Duration:         15001.47 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.01 ms
    [ INFO ]    Average:       1.03 ms
    [ INFO ]    Min:           0.75 ms
    [ INFO ]    Max:           7.00 ms
    [ INFO ] Throughput:   11165.31 FPS


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

