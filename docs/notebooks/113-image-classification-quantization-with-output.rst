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
   Model <#run-nncfquantize-for-getting-an-optimized-model>`__
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

.. parsed-literal::

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

    Receiving objects:  78% (220/282), 1.54 MiB | 3.00 MiB/s

.. parsed-literal::

    Receiving objects:  79% (223/282), 1.54 MiB | 3.00 MiB/s

.. parsed-literal::

    Receiving objects:  80% (226/282), 1.54 MiB | 3.00 MiB/s

.. parsed-literal::

    Receiving objects:  80% (226/282), 3.25 MiB | 3.15 MiB/s

.. parsed-literal::

    Receiving objects:  81% (229/282), 3.25 MiB | 3.15 MiB/s

.. parsed-literal::

    Receiving objects:  82% (232/282), 3.25 MiB | 3.15 MiB/s

.. parsed-literal::

    Receiving objects:  83% (235/282), 4.88 MiB | 3.19 MiB/s

.. parsed-literal::

    Receiving objects:  84% (237/282), 4.88 MiB | 3.19 MiB/s

.. parsed-literal::

    Receiving objects:  84% (239/282), 6.26 MiB | 3.08 MiB/s
Receiving objects:  85% (240/282), 6.26 MiB | 3.08 MiB/s

.. parsed-literal::

    Receiving objects:  86% (243/282), 6.26 MiB | 3.08 MiB/s

.. parsed-literal::

    Receiving objects:  87% (246/282), 6.26 MiB | 3.08 MiB/s

.. parsed-literal::

    Receiving objects:  88% (249/282), 7.88 MiB | 3.11 MiB/s

.. parsed-literal::

    Receiving objects:  89% (251/282), 7.88 MiB | 3.11 MiB/s

.. parsed-literal::

    remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects:  90% (254/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  91% (257/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  92% (260/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  93% (263/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  94% (266/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  95% (268/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  96% (271/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  97% (274/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  98% (277/282), 7.88 MiB | 3.11 MiB/s
Receiving objects:  99% (280/282), 7.88 MiB | 3.11 MiB/s
Receiving objects: 100% (282/282), 7.88 MiB | 3.11 MiB/s
Receiving objects: 100% (282/282), 9.22 MiB | 3.15 MiB/s, done.
    Resolving deltas:   0% (0/135)
Resolving deltas:   2% (3/135)
Resolving deltas:   5% (7/135)
Resolving deltas:   8% (11/135)
Resolving deltas:  11% (16/135)
Resolving deltas:  17% (23/135)
Resolving deltas:  20% (28/135)
Resolving deltas:  21% (29/135)
Resolving deltas:  23% (32/135)
Resolving deltas:  25% (35/135)
Resolving deltas:  26% (36/135)
Resolving deltas:  27% (37/135)
Resolving deltas:  28% (38/135)
Resolving deltas:  30% (41/135)
Resolving deltas:  34% (47/135)
Resolving deltas:  40% (54/135)
Resolving deltas:  45% (61/135)
Resolving deltas:  47% (64/135)
Resolving deltas:  50% (68/135)

.. parsed-literal::

    Resolving deltas:  57% (78/135)
Resolving deltas:  58% (79/135)
Resolving deltas:  60% (81/135)
Resolving deltas:  62% (84/135)
Resolving deltas:  70% (95/135)
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
`page <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.

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


  0%|          | 32768/170498071 [00:00<17:49, 159321.06it/s]

.. parsed-literal::


  0%|          | 98304/170498071 [00:00<08:12, 345836.42it/s]

.. parsed-literal::


  0%|          | 229376/170498071 [00:00<04:06, 690624.17it/s]

.. parsed-literal::


  0%|          | 425984/170498071 [00:00<02:34, 1102701.70it/s]

.. parsed-literal::


  0%|          | 819200/170498071 [00:00<01:26, 1951381.28it/s]

.. parsed-literal::


  1%|          | 1212416/170498071 [00:00<01:08, 2477781.42it/s]

.. parsed-literal::


  1%|          | 1605632/170498071 [00:00<01:00, 2796558.99it/s]

.. parsed-literal::


  1%|          | 1966080/170498071 [00:00<00:56, 2965552.29it/s]

.. parsed-literal::


  1%|▏         | 2359296/170498071 [00:01<00:53, 3142150.81it/s]

.. parsed-literal::


  2%|▏         | 2752512/170498071 [00:01<00:51, 3235258.70it/s]

.. parsed-literal::


  2%|▏         | 3145728/170498071 [00:01<00:50, 3326951.07it/s]

.. parsed-literal::


  2%|▏         | 3538944/170498071 [00:01<00:49, 3376725.14it/s]

.. parsed-literal::


  2%|▏         | 3932160/170498071 [00:01<00:49, 3363148.10it/s]

.. parsed-literal::


  3%|▎         | 4325376/170498071 [00:01<00:48, 3399905.79it/s]

.. parsed-literal::


  3%|▎         | 4718592/170498071 [00:01<00:48, 3431221.10it/s]

.. parsed-literal::


  3%|▎         | 5111808/170498071 [00:01<00:47, 3467201.39it/s]

.. parsed-literal::


  3%|▎         | 5472256/170498071 [00:01<00:48, 3434476.25it/s]

.. parsed-literal::


  3%|▎         | 5865472/170498071 [00:02<00:47, 3455893.80it/s]

.. parsed-literal::


  4%|▎         | 6258688/170498071 [00:02<00:46, 3494985.80it/s]

.. parsed-literal::


  4%|▍         | 6651904/170498071 [00:02<00:47, 3421353.73it/s]

.. parsed-literal::


  4%|▍         | 7045120/170498071 [00:02<00:47, 3468699.04it/s]

.. parsed-literal::


  4%|▍         | 7438336/170498071 [00:02<00:47, 3417786.18it/s]

.. parsed-literal::


  5%|▍         | 7831552/170498071 [00:02<00:47, 3428067.47it/s]

.. parsed-literal::


  5%|▍         | 8224768/170498071 [00:02<00:47, 3398951.31it/s]

.. parsed-literal::


  5%|▌         | 8617984/170498071 [00:02<00:47, 3434543.58it/s]

.. parsed-literal::


  5%|▌         | 8978432/170498071 [00:02<00:47, 3417708.85it/s]

.. parsed-literal::


  5%|▌         | 9371648/170498071 [00:03<00:47, 3427501.95it/s]

.. parsed-literal::


  6%|▌         | 9764864/170498071 [00:03<00:46, 3453461.91it/s]

.. parsed-literal::


  6%|▌         | 10158080/170498071 [00:03<00:47, 3346858.44it/s]

.. parsed-literal::


  6%|▌         | 10551296/170498071 [00:03<00:47, 3358931.64it/s]

.. parsed-literal::


  6%|▋         | 10944512/170498071 [00:03<00:48, 3264557.29it/s]

.. parsed-literal::


  7%|▋         | 11337728/170498071 [00:03<00:47, 3319786.42it/s]

.. parsed-literal::


  7%|▋         | 11730944/170498071 [00:03<00:47, 3367114.69it/s]

.. parsed-literal::


  7%|▋         | 12124160/170498071 [00:03<00:46, 3398293.56it/s]

.. parsed-literal::


  7%|▋         | 12517376/170498071 [00:04<00:46, 3400297.18it/s]

.. parsed-literal::


  8%|▊         | 12877824/170498071 [00:04<00:46, 3424443.12it/s]

.. parsed-literal::


  8%|▊         | 13271040/170498071 [00:04<00:45, 3456715.99it/s]

.. parsed-literal::


  8%|▊         | 13664256/170498071 [00:04<00:45, 3428106.70it/s]

.. parsed-literal::


  8%|▊         | 14057472/170498071 [00:04<00:45, 3470949.09it/s]

.. parsed-literal::


  8%|▊         | 14450688/170498071 [00:04<00:45, 3430270.28it/s]

.. parsed-literal::


  9%|▊         | 14843904/170498071 [00:04<00:45, 3453754.00it/s]

.. parsed-literal::


  9%|▉         | 15237120/170498071 [00:04<00:44, 3480396.18it/s]

.. parsed-literal::


  9%|▉         | 15630336/170498071 [00:04<00:44, 3493033.27it/s]

.. parsed-literal::


  9%|▉         | 16023552/170498071 [00:05<00:44, 3500549.01it/s]

.. parsed-literal::


 10%|▉         | 16384000/170498071 [00:05<00:44, 3456065.13it/s]

.. parsed-literal::


 10%|▉         | 16777216/170498071 [00:05<00:43, 3494040.57it/s]

.. parsed-literal::


 10%|█         | 17170432/170498071 [00:05<00:43, 3501382.15it/s]

.. parsed-literal::


 10%|█         | 17563648/170498071 [00:05<00:43, 3520679.77it/s]

.. parsed-literal::


 11%|█         | 17956864/170498071 [00:05<00:43, 3514533.80it/s]

.. parsed-literal::


 11%|█         | 18350080/170498071 [00:05<00:43, 3534849.25it/s]

.. parsed-literal::


 11%|█         | 18743296/170498071 [00:05<00:43, 3509971.57it/s]

.. parsed-literal::


 11%|█         | 19136512/170498071 [00:05<00:43, 3509723.58it/s]

.. parsed-literal::


 11%|█▏        | 19529728/170498071 [00:06<00:43, 3444511.97it/s]

.. parsed-literal::


 12%|█▏        | 19922944/170498071 [00:06<00:43, 3472767.50it/s]

.. parsed-literal::


 12%|█▏        | 20316160/170498071 [00:06<00:43, 3474103.75it/s]

.. parsed-literal::


 12%|█▏        | 20709376/170498071 [00:06<00:42, 3490817.69it/s]

.. parsed-literal::


 12%|█▏        | 21102592/170498071 [00:06<00:42, 3504234.96it/s]

.. parsed-literal::


 13%|█▎        | 21495808/170498071 [00:06<00:42, 3524221.23it/s]

.. parsed-literal::


 13%|█▎        | 21856256/170498071 [00:06<00:42, 3512243.89it/s]

.. parsed-literal::


 13%|█▎        | 22249472/170498071 [00:06<00:42, 3511413.96it/s]

.. parsed-literal::


 13%|█▎        | 22642688/170498071 [00:06<00:41, 3521574.73it/s]

.. parsed-literal::


 14%|█▎        | 23035904/170498071 [00:07<00:42, 3509418.90it/s]

.. parsed-literal::


 14%|█▎        | 23429120/170498071 [00:07<00:42, 3499076.07it/s]

.. parsed-literal::


 14%|█▍        | 23822336/170498071 [00:07<00:41, 3498335.24it/s]

.. parsed-literal::


 14%|█▍        | 24215552/170498071 [00:07<00:41, 3507602.24it/s]

.. parsed-literal::


 14%|█▍        | 24576000/170498071 [00:07<00:41, 3512076.83it/s]

.. parsed-literal::


 15%|█▍        | 24969216/170498071 [00:07<00:41, 3486072.92it/s]

.. parsed-literal::


 15%|█▍        | 25362432/170498071 [00:07<00:41, 3497632.50it/s]

.. parsed-literal::


 15%|█▌        | 25755648/170498071 [00:07<00:41, 3503453.36it/s]

.. parsed-literal::


 15%|█▌        | 26148864/170498071 [00:07<00:41, 3504151.92it/s]

.. parsed-literal::


 16%|█▌        | 26542080/170498071 [00:08<00:40, 3512527.53it/s]

.. parsed-literal::


 16%|█▌        | 26935296/170498071 [00:08<00:40, 3509319.17it/s]

.. parsed-literal::


 16%|█▌        | 27328512/170498071 [00:08<00:40, 3512228.87it/s]

.. parsed-literal::


 16%|█▋        | 27721728/170498071 [00:08<00:40, 3503348.61it/s]

.. parsed-literal::


 16%|█▋        | 28114944/170498071 [00:08<00:40, 3499294.30it/s]

.. parsed-literal::


 17%|█▋        | 28508160/170498071 [00:08<00:40, 3509029.57it/s]

.. parsed-literal::


 17%|█▋        | 28901376/170498071 [00:08<00:40, 3509898.63it/s]

.. parsed-literal::


 17%|█▋        | 29294592/170498071 [00:08<00:40, 3504663.81it/s]

.. parsed-literal::


 17%|█▋        | 29687808/170498071 [00:08<00:40, 3475483.76it/s]

.. parsed-literal::


 18%|█▊        | 30081024/170498071 [00:09<00:40, 3486148.77it/s]

.. parsed-literal::


 18%|█▊        | 30474240/170498071 [00:09<00:40, 3487390.72it/s]

.. parsed-literal::


 18%|█▊        | 30867456/170498071 [00:09<00:40, 3473317.03it/s]

.. parsed-literal::


 18%|█▊        | 31260672/170498071 [00:09<00:39, 3484362.79it/s]

.. parsed-literal::


 19%|█▊        | 31653888/170498071 [00:09<00:40, 3467577.82it/s]

.. parsed-literal::


 19%|█▉        | 32047104/170498071 [00:09<00:39, 3483614.57it/s]

.. parsed-literal::


 19%|█▉        | 32440320/170498071 [00:09<00:39, 3489110.36it/s]

.. parsed-literal::


 19%|█▉        | 32800768/170498071 [00:09<00:41, 3354779.15it/s]

.. parsed-literal::


 19%|█▉        | 33193984/170498071 [00:09<00:40, 3389082.55it/s]

.. parsed-literal::


 20%|█▉        | 33587200/170498071 [00:10<00:40, 3379294.85it/s]

.. parsed-literal::


 20%|█▉        | 33980416/170498071 [00:10<00:39, 3416156.92it/s]

.. parsed-literal::


 20%|██        | 34373632/170498071 [00:10<00:39, 3428191.58it/s]

.. parsed-literal::


 20%|██        | 34766848/170498071 [00:10<00:39, 3439254.46it/s]

.. parsed-literal::


 21%|██        | 35160064/170498071 [00:10<00:39, 3437244.74it/s]

.. parsed-literal::


 21%|██        | 35520512/170498071 [00:10<00:38, 3475115.72it/s]

.. parsed-literal::


 21%|██        | 35913728/170498071 [00:10<00:38, 3470586.40it/s]

.. parsed-literal::


 21%|██▏       | 36306944/170498071 [00:10<00:39, 3419369.14it/s]

.. parsed-literal::


 22%|██▏       | 36700160/170498071 [00:10<00:38, 3442300.38it/s]

.. parsed-literal::


 22%|██▏       | 37093376/170498071 [00:11<00:38, 3444542.53it/s]

.. parsed-literal::


 22%|██▏       | 37486592/170498071 [00:11<00:38, 3477088.38it/s]

.. parsed-literal::


 22%|██▏       | 37879808/170498071 [00:11<00:37, 3490508.01it/s]

.. parsed-literal::


 22%|██▏       | 38273024/170498071 [00:11<00:38, 3474669.47it/s]

.. parsed-literal::


 23%|██▎       | 38666240/170498071 [00:11<00:37, 3502424.53it/s]

.. parsed-literal::


 23%|██▎       | 39059456/170498071 [00:11<00:37, 3483716.65it/s]

.. parsed-literal::


 23%|██▎       | 39452672/170498071 [00:11<00:37, 3512742.80it/s]

.. parsed-literal::


 23%|██▎       | 39845888/170498071 [00:11<00:38, 3393584.89it/s]

.. parsed-literal::


 24%|██▎       | 40239104/170498071 [00:12<00:38, 3425070.90it/s]

.. parsed-literal::


 24%|██▍       | 40599552/170498071 [00:12<00:38, 3369064.94it/s]

.. parsed-literal::


 24%|██▍       | 40992768/170498071 [00:12<00:38, 3398625.11it/s]

.. parsed-literal::


 24%|██▍       | 41385984/170498071 [00:12<00:37, 3416331.41it/s]

.. parsed-literal::


 25%|██▍       | 41779200/170498071 [00:12<00:37, 3437354.72it/s]

.. parsed-literal::


 25%|██▍       | 42172416/170498071 [00:12<00:37, 3428553.23it/s]

.. parsed-literal::


 25%|██▍       | 42565632/170498071 [00:12<00:36, 3470767.69it/s]

.. parsed-literal::


 25%|██▌       | 42958848/170498071 [00:12<00:36, 3482105.50it/s]

.. parsed-literal::


 25%|██▌       | 43352064/170498071 [00:12<00:37, 3390977.11it/s]

.. parsed-literal::


 26%|██▌       | 43745280/170498071 [00:13<00:36, 3434282.19it/s]

.. parsed-literal::


 26%|██▌       | 44138496/170498071 [00:13<00:36, 3459745.15it/s]

.. parsed-literal::


 26%|██▌       | 44531712/170498071 [00:13<00:36, 3471665.21it/s]

.. parsed-literal::


 26%|██▋       | 44924928/170498071 [00:13<00:36, 3451591.98it/s]

.. parsed-literal::


 27%|██▋       | 45285376/170498071 [00:13<00:36, 3442447.00it/s]

.. parsed-literal::


 27%|██▋       | 45678592/170498071 [00:13<00:36, 3425773.86it/s]

.. parsed-literal::


 27%|██▋       | 46039040/170498071 [00:13<00:36, 3454436.65it/s]

.. parsed-literal::


 27%|██▋       | 46399488/170498071 [00:13<00:36, 3397259.02it/s]

.. parsed-literal::


 27%|██▋       | 46759936/170498071 [00:13<00:38, 3207414.91it/s]

.. parsed-literal::


 28%|██▊       | 47120384/170498071 [00:14<00:38, 3234051.88it/s]

.. parsed-literal::


 28%|██▊       | 47513600/170498071 [00:14<00:37, 3297023.79it/s]

.. parsed-literal::


 28%|██▊       | 47906816/170498071 [00:14<00:36, 3360749.12it/s]

.. parsed-literal::


 28%|██▊       | 48300032/170498071 [00:14<00:35, 3410282.64it/s]

.. parsed-literal::


 29%|██▊       | 48693248/170498071 [00:14<00:35, 3413834.59it/s]

.. parsed-literal::


 29%|██▉       | 49086464/170498071 [00:14<00:35, 3455809.78it/s]

.. parsed-literal::


 29%|██▉       | 49479680/170498071 [00:14<00:34, 3459182.80it/s]

.. parsed-literal::


 29%|██▉       | 49840128/170498071 [00:14<00:34, 3455437.06it/s]

.. parsed-literal::


 29%|██▉       | 50233344/170498071 [00:14<00:34, 3483611.95it/s]

.. parsed-literal::


 30%|██▉       | 50626560/170498071 [00:15<00:34, 3429144.44it/s]

.. parsed-literal::


 30%|██▉       | 51019776/170498071 [00:15<00:34, 3445957.30it/s]

.. parsed-literal::


 30%|███       | 51412992/170498071 [00:15<00:34, 3423624.04it/s]

.. parsed-literal::


 30%|███       | 51806208/170498071 [00:15<00:34, 3435761.35it/s]

.. parsed-literal::


 31%|███       | 52199424/170498071 [00:15<00:34, 3434733.06it/s]

.. parsed-literal::


 31%|███       | 52592640/170498071 [00:15<00:35, 3345274.38it/s]

.. parsed-literal::


 31%|███       | 52985856/170498071 [00:15<00:34, 3362368.41it/s]

.. parsed-literal::


 31%|███▏      | 53379072/170498071 [00:15<00:34, 3355621.56it/s]

.. parsed-literal::


 32%|███▏      | 53772288/170498071 [00:15<00:34, 3362535.78it/s]

.. parsed-literal::


 32%|███▏      | 54165504/170498071 [00:16<00:34, 3387908.95it/s]

.. parsed-literal::


 32%|███▏      | 54558720/170498071 [00:16<00:34, 3335367.77it/s]

.. parsed-literal::


 32%|███▏      | 54951936/170498071 [00:16<00:34, 3365169.21it/s]

.. parsed-literal::


 32%|███▏      | 55312384/170498071 [00:16<00:34, 3366559.80it/s]

.. parsed-literal::


 33%|███▎      | 55705600/170498071 [00:16<00:34, 3346483.03it/s]

.. parsed-literal::


 33%|███▎      | 56098816/170498071 [00:16<00:34, 3344016.06it/s]

.. parsed-literal::


 33%|███▎      | 56492032/170498071 [00:16<00:33, 3372292.47it/s]

.. parsed-literal::


 33%|███▎      | 56885248/170498071 [00:16<00:34, 3303634.78it/s]

.. parsed-literal::


 34%|███▎      | 57278464/170498071 [00:17<00:34, 3243964.90it/s]

.. parsed-literal::


 34%|███▍      | 57671680/170498071 [00:17<00:34, 3317754.86it/s]

.. parsed-literal::


 34%|███▍      | 58064896/170498071 [00:17<00:33, 3341935.53it/s]

.. parsed-literal::


 34%|███▍      | 58458112/170498071 [00:17<00:32, 3402132.73it/s]

.. parsed-literal::


 35%|███▍      | 58851328/170498071 [00:17<00:32, 3439859.50it/s]

.. parsed-literal::


 35%|███▍      | 59244544/170498071 [00:17<00:32, 3456994.07it/s]

.. parsed-literal::


 35%|███▍      | 59637760/170498071 [00:17<00:32, 3459968.36it/s]

.. parsed-literal::


 35%|███▌      | 60030976/170498071 [00:17<00:31, 3482909.83it/s]

.. parsed-literal::


 35%|███▌      | 60424192/170498071 [00:17<00:31, 3489179.26it/s]

.. parsed-literal::


 36%|███▌      | 60784640/170498071 [00:18<00:31, 3469195.80it/s]

.. parsed-literal::


 36%|███▌      | 61177856/170498071 [00:18<00:31, 3473813.52it/s]

.. parsed-literal::


 36%|███▌      | 61571072/170498071 [00:18<00:31, 3480664.25it/s]

.. parsed-literal::


 36%|███▋      | 61964288/170498071 [00:18<00:31, 3490879.44it/s]

.. parsed-literal::


 37%|███▋      | 62357504/170498071 [00:18<00:30, 3501907.46it/s]

.. parsed-literal::


 37%|███▋      | 62750720/170498071 [00:18<00:30, 3506862.10it/s]

.. parsed-literal::


 37%|███▋      | 63143936/170498071 [00:18<00:30, 3513662.77it/s]

.. parsed-literal::


 37%|███▋      | 63504384/170498071 [00:18<00:31, 3433760.20it/s]

.. parsed-literal::


 37%|███▋      | 63897600/170498071 [00:18<00:30, 3464553.47it/s]

.. parsed-literal::


 38%|███▊      | 64290816/170498071 [00:19<00:31, 3410071.66it/s]

.. parsed-literal::


 38%|███▊      | 64684032/170498071 [00:19<00:31, 3412002.79it/s]

.. parsed-literal::


 38%|███▊      | 65077248/170498071 [00:19<00:30, 3449396.24it/s]

.. parsed-literal::


 38%|███▊      | 65470464/170498071 [00:19<00:30, 3467649.41it/s]

.. parsed-literal::


 39%|███▊      | 65863680/170498071 [00:19<00:30, 3474258.76it/s]

.. parsed-literal::


 39%|███▉      | 66256896/170498071 [00:19<00:29, 3484429.72it/s]

.. parsed-literal::


 39%|███▉      | 66650112/170498071 [00:19<00:29, 3499178.96it/s]

.. parsed-literal::


 39%|███▉      | 67010560/170498071 [00:19<00:29, 3451382.53it/s]

.. parsed-literal::


 40%|███▉      | 67403776/170498071 [00:19<00:29, 3472290.56it/s]

.. parsed-literal::


 40%|███▉      | 67796992/170498071 [00:20<00:29, 3487959.76it/s]

.. parsed-literal::


 40%|███▉      | 68190208/170498071 [00:20<00:29, 3456294.00it/s]

.. parsed-literal::


 40%|████      | 68583424/170498071 [00:20<00:29, 3448161.31it/s]

.. parsed-literal::


 40%|████      | 68976640/170498071 [00:20<00:29, 3409374.50it/s]

.. parsed-literal::


 41%|████      | 69369856/170498071 [00:20<00:29, 3455663.04it/s]

.. parsed-literal::


 41%|████      | 69763072/170498071 [00:20<00:29, 3436132.66it/s]

.. parsed-literal::


 41%|████      | 70156288/170498071 [00:20<00:28, 3475615.63it/s]

.. parsed-literal::


 41%|████▏     | 70549504/170498071 [00:20<00:28, 3469796.94it/s]

.. parsed-literal::


 42%|████▏     | 70942720/170498071 [00:20<00:28, 3486314.34it/s]

.. parsed-literal::


 42%|████▏     | 71335936/170498071 [00:21<00:28, 3489919.35it/s]

.. parsed-literal::


 42%|████▏     | 71696384/170498071 [00:21<00:28, 3510865.14it/s]

.. parsed-literal::


 42%|████▏     | 72089600/170498071 [00:21<00:28, 3499380.01it/s]

.. parsed-literal::


 43%|████▎     | 72482816/170498071 [00:21<00:28, 3474081.03it/s]

.. parsed-literal::


 43%|████▎     | 72876032/170498071 [00:21<00:27, 3486946.61it/s]

.. parsed-literal::


 43%|████▎     | 73269248/170498071 [00:21<00:27, 3505388.99it/s]

.. parsed-literal::


 43%|████▎     | 73662464/170498071 [00:21<00:27, 3514219.02it/s]

.. parsed-literal::


 43%|████▎     | 74055680/170498071 [00:21<00:27, 3479576.58it/s]

.. parsed-literal::


 44%|████▎     | 74448896/170498071 [00:21<00:27, 3472060.77it/s]

.. parsed-literal::


 44%|████▍     | 74842112/170498071 [00:22<00:27, 3474806.03it/s]

.. parsed-literal::


 44%|████▍     | 75235328/170498071 [00:22<00:27, 3488193.43it/s]

.. parsed-literal::


 44%|████▍     | 75628544/170498071 [00:22<00:27, 3467762.77it/s]

.. parsed-literal::


 45%|████▍     | 76021760/170498071 [00:22<00:27, 3480694.18it/s]

.. parsed-literal::


 45%|████▍     | 76382208/170498071 [00:22<00:26, 3501794.61it/s]

.. parsed-literal::


 45%|████▌     | 76775424/170498071 [00:22<00:26, 3517269.88it/s]

.. parsed-literal::


 45%|████▌     | 77168640/170498071 [00:22<00:26, 3496405.13it/s]

.. parsed-literal::


 45%|████▌     | 77561856/170498071 [00:22<00:26, 3467400.47it/s]

.. parsed-literal::


 46%|████▌     | 77955072/170498071 [00:23<00:26, 3448958.01it/s]

.. parsed-literal::


 46%|████▌     | 78348288/170498071 [00:23<00:26, 3452329.37it/s]

.. parsed-literal::


 46%|████▌     | 78741504/170498071 [00:23<00:26, 3445231.29it/s]

.. parsed-literal::


 46%|████▋     | 79134720/170498071 [00:23<00:26, 3485462.91it/s]

.. parsed-literal::


 47%|████▋     | 79495168/170498071 [00:23<00:26, 3497444.68it/s]

.. parsed-literal::


 47%|████▋     | 79888384/170498071 [00:23<00:26, 3424816.63it/s]

.. parsed-literal::


 47%|████▋     | 80281600/170498071 [00:23<00:26, 3422703.30it/s]

.. parsed-literal::


 47%|████▋     | 80674816/170498071 [00:23<00:26, 3442874.52it/s]

.. parsed-literal::


 48%|████▊     | 81068032/170498071 [00:23<00:25, 3465691.38it/s]

.. parsed-literal::


 48%|████▊     | 81461248/170498071 [00:24<00:25, 3483984.88it/s]

.. parsed-literal::


 48%|████▊     | 81854464/170498071 [00:24<00:25, 3501251.13it/s]

.. parsed-literal::


 48%|████▊     | 82247680/170498071 [00:24<00:25, 3502824.82it/s]

.. parsed-literal::


 48%|████▊     | 82640896/170498071 [00:24<00:25, 3483134.45it/s]

.. parsed-literal::


 49%|████▊     | 83034112/170498071 [00:24<00:25, 3486838.66it/s]

.. parsed-literal::


 49%|████▉     | 83427328/170498071 [00:24<00:24, 3515248.64it/s]

.. parsed-literal::


 49%|████▉     | 83820544/170498071 [00:24<00:24, 3507111.20it/s]

.. parsed-literal::


 49%|████▉     | 84213760/170498071 [00:24<00:24, 3505983.09it/s]

.. parsed-literal::


 50%|████▉     | 84606976/170498071 [00:24<00:24, 3468846.41it/s]

.. parsed-literal::


 50%|████▉     | 84967424/170498071 [00:25<00:24, 3502454.07it/s]

.. parsed-literal::


 50%|█████     | 85360640/170498071 [00:25<00:24, 3444159.11it/s]

.. parsed-literal::


 50%|█████     | 85753856/170498071 [00:25<00:24, 3447739.69it/s]

.. parsed-literal::


 51%|█████     | 86147072/170498071 [00:25<00:24, 3466145.05it/s]

.. parsed-literal::


 51%|█████     | 86507520/170498071 [00:25<00:24, 3474412.66it/s]

.. parsed-literal::


 51%|█████     | 86900736/170498071 [00:25<00:24, 3482069.07it/s]

.. parsed-literal::


 51%|█████     | 87293952/170498071 [00:25<00:24, 3465485.70it/s]

.. parsed-literal::


 51%|█████▏    | 87687168/170498071 [00:25<00:23, 3460684.01it/s]

.. parsed-literal::


 52%|█████▏    | 88080384/170498071 [00:25<00:23, 3468887.27it/s]

.. parsed-literal::


 52%|█████▏    | 88440832/170498071 [00:26<00:24, 3384377.86it/s]

.. parsed-literal::


 52%|█████▏    | 88801280/170498071 [00:26<00:24, 3342320.01it/s]

.. parsed-literal::


 52%|█████▏    | 89161728/170498071 [00:26<00:24, 3302037.40it/s]

.. parsed-literal::


 53%|█████▎    | 89554944/170498071 [00:26<00:24, 3366849.23it/s]

.. parsed-literal::


 53%|█████▎    | 89915392/170498071 [00:26<00:23, 3431950.29it/s]

.. parsed-literal::


 53%|█████▎    | 90308608/170498071 [00:26<00:23, 3441640.76it/s]

.. parsed-literal::


 53%|█████▎    | 90701824/170498071 [00:26<00:23, 3415361.03it/s]

.. parsed-literal::


 53%|█████▎    | 91095040/170498071 [00:26<00:23, 3430726.31it/s]

.. parsed-literal::


 54%|█████▎    | 91488256/170498071 [00:26<00:22, 3445818.83it/s]

.. parsed-literal::


 54%|█████▍    | 91881472/170498071 [00:27<00:22, 3454890.02it/s]

.. parsed-literal::


 54%|█████▍    | 92241920/170498071 [00:27<00:22, 3487362.68it/s]

.. parsed-literal::


 54%|█████▍    | 92602368/170498071 [00:27<00:22, 3515510.18it/s]

.. parsed-literal::


 55%|█████▍    | 92962816/170498071 [00:27<00:22, 3471140.25it/s]

.. parsed-literal::


 55%|█████▍    | 93323264/170498071 [00:27<00:22, 3458951.39it/s]

.. parsed-literal::


 55%|█████▍    | 93683712/170498071 [00:27<00:22, 3384382.09it/s]

.. parsed-literal::


 55%|█████▌    | 94076928/170498071 [00:27<00:22, 3414757.60it/s]

.. parsed-literal::


 55%|█████▌    | 94437376/170498071 [00:27<00:21, 3467653.11it/s]

.. parsed-literal::


 56%|█████▌    | 94797824/170498071 [00:27<00:22, 3434428.20it/s]

.. parsed-literal::


 56%|█████▌    | 95158272/170498071 [00:27<00:22, 3414989.01it/s]

.. parsed-literal::


 56%|█████▌    | 95518720/170498071 [00:28<00:22, 3393452.38it/s]

.. parsed-literal::


 56%|█████▋    | 95911936/170498071 [00:28<00:21, 3433766.60it/s]

.. parsed-literal::


 56%|█████▋    | 96272384/170498071 [00:28<00:21, 3406734.36it/s]

.. parsed-literal::


 57%|█████▋    | 96665600/170498071 [00:28<00:21, 3447606.62it/s]

.. parsed-literal::


 57%|█████▋    | 97058816/170498071 [00:28<00:21, 3460594.84it/s]

.. parsed-literal::


 57%|█████▋    | 97452032/170498071 [00:28<00:20, 3480313.96it/s]

.. parsed-literal::


 57%|█████▋    | 97845248/170498071 [00:28<00:20, 3472783.65it/s]

.. parsed-literal::


 58%|█████▊    | 98238464/170498071 [00:28<00:20, 3473028.03it/s]

.. parsed-literal::


 58%|█████▊    | 98631680/170498071 [00:28<00:20, 3502740.79it/s]

.. parsed-literal::


 58%|█████▊    | 98992128/170498071 [00:29<00:20, 3487352.89it/s]

.. parsed-literal::


 58%|█████▊    | 99385344/170498071 [00:29<00:20, 3403661.06it/s]

.. parsed-literal::


 59%|█████▊    | 99778560/170498071 [00:29<00:21, 3355877.64it/s]

.. parsed-literal::


 59%|█████▉    | 100171776/170498071 [00:29<00:20, 3370361.62it/s]

.. parsed-literal::


 59%|█████▉    | 100564992/170498071 [00:29<00:20, 3398266.42it/s]

.. parsed-literal::


 59%|█████▉    | 100958208/170498071 [00:29<00:20, 3448536.59it/s]

.. parsed-literal::


 59%|█████▉    | 101318656/170498071 [00:29<00:19, 3477329.11it/s]

.. parsed-literal::


 60%|█████▉    | 101679104/170498071 [00:29<00:19, 3507611.52it/s]

.. parsed-literal::


 60%|█████▉    | 102039552/170498071 [00:29<00:20, 3367888.66it/s]

.. parsed-literal::


 60%|██████    | 102400000/170498071 [00:30<00:20, 3324244.54it/s]

.. parsed-literal::


 60%|██████    | 102760448/170498071 [00:30<00:20, 3385271.12it/s]

.. parsed-literal::


 61%|██████    | 103153664/170498071 [00:30<00:19, 3442700.09it/s]

.. parsed-literal::


 61%|██████    | 103514112/170498071 [00:30<00:19, 3484845.03it/s]

.. parsed-literal::


 61%|██████    | 103874560/170498071 [00:30<00:19, 3404156.62it/s]

.. parsed-literal::


 61%|██████    | 104235008/170498071 [00:30<00:19, 3322917.87it/s]

.. parsed-literal::


 61%|██████▏   | 104595456/170498071 [00:30<00:19, 3379783.36it/s]

.. parsed-literal::


 62%|██████▏   | 104988672/170498071 [00:30<00:19, 3378604.35it/s]

.. parsed-literal::


 62%|██████▏   | 105381888/170498071 [00:30<00:19, 3411433.95it/s]

.. parsed-literal::


 62%|██████▏   | 105775104/170498071 [00:31<00:18, 3416828.76it/s]

.. parsed-literal::


 62%|██████▏   | 106168320/170498071 [00:31<00:18, 3418513.24it/s]

.. parsed-literal::


 63%|██████▎   | 106561536/170498071 [00:31<00:18, 3448187.21it/s]

.. parsed-literal::


 63%|██████▎   | 106954752/170498071 [00:31<00:18, 3421043.57it/s]

.. parsed-literal::


 63%|██████▎   | 107347968/170498071 [00:31<00:18, 3455266.12it/s]

.. parsed-literal::


 63%|██████▎   | 107741184/170498071 [00:31<00:18, 3374949.45it/s]

.. parsed-literal::


 63%|██████▎   | 108134400/170498071 [00:31<00:18, 3421729.07it/s]

.. parsed-literal::


 64%|██████▎   | 108494848/170498071 [00:31<00:18, 3438429.14it/s]

.. parsed-literal::


 64%|██████▍   | 108888064/170498071 [00:32<00:18, 3382604.39it/s]

.. parsed-literal::


 64%|██████▍   | 109281280/170498071 [00:32<00:18, 3383537.51it/s]

.. parsed-literal::


 64%|██████▍   | 109674496/170498071 [00:32<00:17, 3401059.17it/s]

.. parsed-literal::


 65%|██████▍   | 110067712/170498071 [00:32<00:17, 3372428.68it/s]

.. parsed-literal::


 65%|██████▍   | 110460928/170498071 [00:32<00:17, 3424935.40it/s]

.. parsed-literal::


 65%|██████▌   | 110854144/170498071 [00:32<00:17, 3439823.11it/s]

.. parsed-literal::


 65%|██████▌   | 111247360/170498071 [00:32<00:17, 3449680.97it/s]

.. parsed-literal::


 65%|██████▌   | 111640576/170498071 [00:32<00:17, 3417852.93it/s]

.. parsed-literal::


 66%|██████▌   | 112033792/170498071 [00:32<00:17, 3370616.05it/s]

.. parsed-literal::


 66%|██████▌   | 112427008/170498071 [00:33<00:17, 3353485.03it/s]

.. parsed-literal::


 66%|██████▌   | 112787456/170498071 [00:33<00:17, 3344338.76it/s]

.. parsed-literal::


 66%|██████▋   | 113147904/170498071 [00:33<00:18, 3052509.57it/s]

.. parsed-literal::


 67%|██████▋   | 113475584/170498071 [00:33<00:18, 3094558.78it/s]

.. parsed-literal::


 67%|██████▋   | 113836032/170498071 [00:33<00:17, 3156146.61it/s]

.. parsed-literal::


 67%|██████▋   | 114229248/170498071 [00:33<00:17, 3278155.01it/s]

.. parsed-literal::


 67%|██████▋   | 114622464/170498071 [00:33<00:16, 3349755.76it/s]

.. parsed-literal::


 67%|██████▋   | 115015680/170498071 [00:33<00:16, 3399767.53it/s]

.. parsed-literal::


 68%|██████▊   | 115408896/170498071 [00:33<00:16, 3389118.37it/s]

.. parsed-literal::


 68%|██████▊   | 115802112/170498071 [00:34<00:15, 3437570.14it/s]

.. parsed-literal::


 68%|██████▊   | 116162560/170498071 [00:34<00:15, 3454047.37it/s]

.. parsed-literal::


 68%|██████▊   | 116555776/170498071 [00:34<00:15, 3455627.87it/s]

.. parsed-literal::


 69%|██████▊   | 116948992/170498071 [00:34<00:15, 3465270.43it/s]

.. parsed-literal::


 69%|██████▉   | 117342208/170498071 [00:34<00:15, 3425977.17it/s]

.. parsed-literal::


 69%|██████▉   | 117735424/170498071 [00:34<00:15, 3446727.72it/s]

.. parsed-literal::


 69%|██████▉   | 118128640/170498071 [00:34<00:15, 3413313.31it/s]

.. parsed-literal::


 70%|██████▉   | 118521856/170498071 [00:34<00:15, 3445688.41it/s]

.. parsed-literal::


 70%|██████▉   | 118915072/170498071 [00:34<00:15, 3388704.31it/s]

.. parsed-literal::


 70%|██████▉   | 119308288/170498071 [00:35<00:15, 3398745.88it/s]

.. parsed-literal::


 70%|███████   | 119701504/170498071 [00:35<00:14, 3428822.43it/s]

.. parsed-literal::


 70%|███████   | 120061952/170498071 [00:35<00:14, 3384391.19it/s]

.. parsed-literal::


 71%|███████   | 120455168/170498071 [00:35<00:14, 3418608.00it/s]

.. parsed-literal::


 71%|███████   | 120848384/170498071 [00:35<00:14, 3375745.77it/s]

.. parsed-literal::


 71%|███████   | 121241600/170498071 [00:35<00:14, 3385379.15it/s]

.. parsed-literal::


 71%|███████▏  | 121634816/170498071 [00:35<00:14, 3371951.45it/s]

.. parsed-literal::


 72%|███████▏  | 122028032/170498071 [00:35<00:14, 3342813.39it/s]

.. parsed-literal::


 72%|███████▏  | 122421248/170498071 [00:36<00:14, 3367397.94it/s]

.. parsed-literal::


 72%|███████▏  | 122814464/170498071 [00:36<00:14, 3308354.59it/s]

.. parsed-literal::


 72%|███████▏  | 123207680/170498071 [00:36<00:14, 3297109.27it/s]

.. parsed-literal::


 72%|███████▏  | 123600896/170498071 [00:36<00:14, 3269407.88it/s]

.. parsed-literal::


 73%|███████▎  | 123961344/170498071 [00:36<00:13, 3345891.62it/s]

.. parsed-literal::


 73%|███████▎  | 124321792/170498071 [00:36<00:13, 3400269.51it/s]

.. parsed-literal::


 73%|███████▎  | 124682240/170498071 [00:36<00:13, 3335605.84it/s]

.. parsed-literal::


 73%|███████▎  | 125042688/170498071 [00:36<00:13, 3310065.63it/s]

.. parsed-literal::


 74%|███████▎  | 125403136/170498071 [00:36<00:13, 3368417.41it/s]

.. parsed-literal::


 74%|███████▍  | 125796352/170498071 [00:37<00:13, 3399104.90it/s]

.. parsed-literal::


 74%|███████▍  | 126189568/170498071 [00:37<00:12, 3435022.09it/s]

.. parsed-literal::


 74%|███████▍  | 126582784/170498071 [00:37<00:12, 3457257.74it/s]

.. parsed-literal::


 74%|███████▍  | 126976000/170498071 [00:37<00:12, 3414931.78it/s]

.. parsed-literal::


 75%|███████▍  | 127369216/170498071 [00:37<00:12, 3416356.80it/s]

.. parsed-literal::


 75%|███████▍  | 127762432/170498071 [00:37<00:12, 3436211.98it/s]

.. parsed-literal::


 75%|███████▌  | 128122880/170498071 [00:37<00:12, 3476973.09it/s]

.. parsed-literal::


 75%|███████▌  | 128516096/170498071 [00:37<00:12, 3488771.26it/s]

.. parsed-literal::


 76%|███████▌  | 128909312/170498071 [00:37<00:12, 3423350.67it/s]

.. parsed-literal::


 76%|███████▌  | 129302528/170498071 [00:38<00:11, 3443302.05it/s]

.. parsed-literal::


 76%|███████▌  | 129695744/170498071 [00:38<00:11, 3436803.76it/s]

.. parsed-literal::


 76%|███████▋  | 130088960/170498071 [00:38<00:11, 3438437.44it/s]

.. parsed-literal::


 77%|███████▋  | 130482176/170498071 [00:38<00:11, 3459683.39it/s]

.. parsed-literal::


 77%|███████▋  | 130875392/170498071 [00:38<00:11, 3480207.68it/s]

.. parsed-literal::


 77%|███████▋  | 131268608/170498071 [00:38<00:11, 3428919.86it/s]

.. parsed-literal::


 77%|███████▋  | 131661824/170498071 [00:38<00:11, 3452629.72it/s]

.. parsed-literal::


 77%|███████▋  | 132055040/170498071 [00:38<00:11, 3449746.64it/s]

.. parsed-literal::


 78%|███████▊  | 132448256/170498071 [00:38<00:10, 3461748.40it/s]

.. parsed-literal::


 78%|███████▊  | 132808704/170498071 [00:39<00:10, 3436388.00it/s]

.. parsed-literal::


 78%|███████▊  | 133201920/170498071 [00:39<00:10, 3453774.41it/s]

.. parsed-literal::


 78%|███████▊  | 133595136/170498071 [00:39<00:10, 3471754.89it/s]

.. parsed-literal::


 79%|███████▊  | 133988352/170498071 [00:39<00:10, 3458692.99it/s]

.. parsed-literal::


 79%|███████▉  | 134381568/170498071 [00:39<00:10, 3471398.31it/s]

.. parsed-literal::


 79%|███████▉  | 134774784/170498071 [00:39<00:10, 3430729.12it/s]

.. parsed-literal::


 79%|███████▉  | 135168000/170498071 [00:39<00:10, 3459392.89it/s]

.. parsed-literal::


 80%|███████▉  | 135561216/170498071 [00:39<00:10, 3483413.60it/s]

.. parsed-literal::


 80%|███████▉  | 135954432/170498071 [00:39<00:09, 3510589.02it/s]

.. parsed-literal::


 80%|███████▉  | 136347648/170498071 [00:40<00:09, 3500075.56it/s]

.. parsed-literal::


 80%|████████  | 136708096/170498071 [00:40<00:09, 3491691.84it/s]

.. parsed-literal::


 80%|████████  | 137068544/170498071 [00:40<00:09, 3464089.20it/s]

.. parsed-literal::


 81%|████████  | 137428992/170498071 [00:40<00:09, 3454680.21it/s]

.. parsed-literal::


 81%|████████  | 137789440/170498071 [00:40<00:09, 3400386.43it/s]

.. parsed-literal::


 81%|████████  | 138149888/170498071 [00:40<00:09, 3454003.83it/s]

.. parsed-literal::


 81%|████████  | 138510336/170498071 [00:40<00:09, 3397734.21it/s]

.. parsed-literal::


 81%|████████▏ | 138870784/170498071 [00:40<00:09, 3397881.93it/s]

.. parsed-literal::


 82%|████████▏ | 139231232/170498071 [00:40<00:10, 2849352.58it/s]

.. parsed-literal::


 82%|████████▏ | 139558912/170498071 [00:41<00:10, 2883378.67it/s]

.. parsed-literal::


 82%|████████▏ | 139919360/170498071 [00:41<00:10, 3005166.98it/s]

.. parsed-literal::


 82%|████████▏ | 140312576/170498071 [00:41<00:09, 3157912.79it/s]

.. parsed-literal::


 82%|████████▏ | 140640256/170498071 [00:41<00:09, 3114713.91it/s]

.. parsed-literal::


 83%|████████▎ | 141033472/170498071 [00:41<00:09, 3240757.42it/s]

.. parsed-literal::


 83%|████████▎ | 141426688/170498071 [00:41<00:08, 3308493.23it/s]

.. parsed-literal::


 83%|████████▎ | 141819904/170498071 [00:41<00:08, 3384777.39it/s]

.. parsed-literal::


 83%|████████▎ | 142213120/170498071 [00:41<00:08, 3438043.48it/s]

.. parsed-literal::


 84%|████████▎ | 142606336/170498071 [00:41<00:08, 3440134.84it/s]

.. parsed-literal::


 84%|████████▍ | 142999552/170498071 [00:42<00:07, 3467078.45it/s]

.. parsed-literal::


 84%|████████▍ | 143360000/170498071 [00:42<00:07, 3477233.59it/s]

.. parsed-literal::


 84%|████████▍ | 143753216/170498071 [00:42<00:07, 3494551.39it/s]

.. parsed-literal::


 85%|████████▍ | 144146432/170498071 [00:42<00:07, 3462148.31it/s]

.. parsed-literal::


 85%|████████▍ | 144539648/170498071 [00:42<00:07, 3482508.37it/s]

.. parsed-literal::


 85%|████████▌ | 144932864/170498071 [00:42<00:07, 3492705.11it/s]

.. parsed-literal::


 85%|████████▌ | 145326080/170498071 [00:42<00:07, 3496094.45it/s]

.. parsed-literal::


 85%|████████▌ | 145719296/170498071 [00:42<00:07, 3521380.63it/s]

.. parsed-literal::


 86%|████████▌ | 146112512/170498071 [00:42<00:06, 3505326.16it/s]

.. parsed-literal::


 86%|████████▌ | 146505728/170498071 [00:43<00:06, 3513020.02it/s]

.. parsed-literal::


 86%|████████▌ | 146866176/170498071 [00:43<00:06, 3454629.01it/s]

.. parsed-literal::


 86%|████████▋ | 147259392/170498071 [00:43<00:06, 3488269.48it/s]

.. parsed-literal::


 87%|████████▋ | 147652608/170498071 [00:43<00:06, 3490423.20it/s]

.. parsed-literal::


 87%|████████▋ | 148045824/170498071 [00:43<00:06, 3487650.55it/s]

.. parsed-literal::


 87%|████████▋ | 148439040/170498071 [00:43<00:06, 3499103.63it/s]

.. parsed-literal::


 87%|████████▋ | 148832256/170498071 [00:43<00:06, 3506202.61it/s]

.. parsed-literal::


 88%|████████▊ | 149192704/170498071 [00:43<00:06, 3483893.78it/s]

.. parsed-literal::


 88%|████████▊ | 149585920/170498071 [00:43<00:05, 3500386.44it/s]

.. parsed-literal::


 88%|████████▊ | 149979136/170498071 [00:44<00:05, 3500222.03it/s]

.. parsed-literal::


 88%|████████▊ | 150372352/170498071 [00:44<00:05, 3507913.04it/s]

.. parsed-literal::


 88%|████████▊ | 150765568/170498071 [00:44<00:05, 3526426.96it/s]

.. parsed-literal::


 89%|████████▊ | 151158784/170498071 [00:44<00:05, 3543392.53it/s]

.. parsed-literal::


 89%|████████▉ | 151552000/170498071 [00:44<00:05, 3525724.08it/s]

.. parsed-literal::


 89%|████████▉ | 151945216/170498071 [00:44<00:05, 3527922.03it/s]

.. parsed-literal::


 89%|████████▉ | 152305664/170498071 [00:44<00:05, 3469811.78it/s]

.. parsed-literal::


 90%|████████▉ | 152698880/170498071 [00:44<00:05, 3465142.91it/s]

.. parsed-literal::


 90%|████████▉ | 153092096/170498071 [00:44<00:04, 3500121.91it/s]

.. parsed-literal::


 90%|█████████ | 153485312/170498071 [00:45<00:04, 3488756.43it/s]

.. parsed-literal::


 90%|█████████ | 153878528/170498071 [00:45<00:04, 3515217.10it/s]

.. parsed-literal::


 90%|█████████ | 154271744/170498071 [00:45<00:04, 3476587.61it/s]

.. parsed-literal::


 91%|█████████ | 154632192/170498071 [00:45<00:04, 3487106.60it/s]

.. parsed-literal::


 91%|█████████ | 155025408/170498071 [00:45<00:04, 3444560.52it/s]

.. parsed-literal::


 91%|█████████ | 155418624/170498071 [00:45<00:04, 3481828.30it/s]

.. parsed-literal::


 91%|█████████▏| 155811840/170498071 [00:45<00:04, 3483007.87it/s]

.. parsed-literal::


 92%|█████████▏| 156205056/170498071 [00:45<00:04, 3506830.99it/s]

.. parsed-literal::


 92%|█████████▏| 156598272/170498071 [00:45<00:03, 3521477.40it/s]

.. parsed-literal::


 92%|█████████▏| 156991488/170498071 [00:46<00:03, 3528103.58it/s]

.. parsed-literal::


 92%|█████████▏| 157384704/170498071 [00:46<00:03, 3494814.03it/s]

.. parsed-literal::


 93%|█████████▎| 157745152/170498071 [00:46<00:03, 3436216.79it/s]

.. parsed-literal::


 93%|█████████▎| 158138368/170498071 [00:46<00:03, 3478300.20it/s]

.. parsed-literal::


 93%|█████████▎| 158531584/170498071 [00:46<00:03, 3470881.95it/s]

.. parsed-literal::


 93%|█████████▎| 158924800/170498071 [00:46<00:03, 3474039.44it/s]

.. parsed-literal::


 93%|█████████▎| 159318016/170498071 [00:46<00:03, 3465326.42it/s]

.. parsed-literal::


 94%|█████████▎| 159711232/170498071 [00:46<00:03, 3484768.02it/s]

.. parsed-literal::


 94%|█████████▍| 160104448/170498071 [00:46<00:03, 3459582.06it/s]

.. parsed-literal::


 94%|█████████▍| 160497664/170498071 [00:47<00:02, 3485772.60it/s]

.. parsed-literal::


 94%|█████████▍| 160890880/170498071 [00:47<00:02, 3488609.58it/s]

.. parsed-literal::


 95%|█████████▍| 161251328/170498071 [00:47<00:02, 3464784.39it/s]

.. parsed-literal::


 95%|█████████▍| 161644544/170498071 [00:47<00:02, 3461796.87it/s]

.. parsed-literal::


 95%|█████████▌| 162037760/170498071 [00:47<00:02, 3448220.79it/s]

.. parsed-literal::


 95%|█████████▌| 162430976/170498071 [00:47<00:02, 3466057.60it/s]

.. parsed-literal::


 95%|█████████▌| 162824192/170498071 [00:47<00:02, 3450743.03it/s]

.. parsed-literal::


 96%|█████████▌| 163217408/170498071 [00:47<00:02, 3476214.85it/s]

.. parsed-literal::


 96%|█████████▌| 163610624/170498071 [00:48<00:01, 3493971.34it/s]

.. parsed-literal::


 96%|█████████▌| 164003840/170498071 [00:48<00:01, 3483692.26it/s]

.. parsed-literal::


 96%|█████████▋| 164397056/170498071 [00:48<00:01, 3504420.44it/s]

.. parsed-literal::


 97%|█████████▋| 164790272/170498071 [00:48<00:01, 3487537.72it/s]

.. parsed-literal::


 97%|█████████▋| 165183488/170498071 [00:48<00:01, 3461074.26it/s]

.. parsed-literal::


 97%|█████████▋| 165576704/170498071 [00:48<00:01, 3480207.21it/s]

.. parsed-literal::


 97%|█████████▋| 165969920/170498071 [00:48<00:01, 3497381.25it/s]

.. parsed-literal::


 98%|█████████▊| 166363136/170498071 [00:48<00:01, 3509693.81it/s]

.. parsed-literal::


 98%|█████████▊| 166756352/170498071 [00:48<00:01, 3529745.52it/s]

.. parsed-literal::


 98%|█████████▊| 167149568/170498071 [00:49<00:00, 3538063.60it/s]

.. parsed-literal::


 98%|█████████▊| 167542784/170498071 [00:49<00:00, 3526645.72it/s]

.. parsed-literal::


 98%|█████████▊| 167903232/170498071 [00:49<00:00, 3497290.73it/s]

.. parsed-literal::


 99%|█████████▊| 168296448/170498071 [00:49<00:00, 3505435.52it/s]

.. parsed-literal::


 99%|█████████▉| 168689664/170498071 [00:49<00:00, 3493854.87it/s]

.. parsed-literal::


   99%|█████████▉| 169082880/170498071 [00:49<00:00, 3491008.45it/s]

.. parsed-literal::


   99%|█████████▉| 169476096/170498071 [00:49<00:00, 3512550.85it/s]

.. parsed-literal::


   100%|█████████▉| 169869312/170498071 [00:49<00:00, 3525497.89it/s]

.. parsed-literal::


   100%|█████████▉| 170262528/170498071 [00:49<00:00, 3463408.54it/s]

.. parsed-literal::


   100%|██████████| 170498071/170498071 [00:49<00:00, 3412275.81it/s]



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
`page <https://docs.openvino.ai/2023.0/basic_quantization_flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)


.. parsed-literal::

    2024-01-18 23:03:17.359120: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-18 23:03:17.390780: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-18 23:03:18.021579: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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
    Accuracy of the optimized model: 93.54%


Compare Performance of the Original and Quantized Models
--------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
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
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 9.54 ms
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

    [ INFO ] Compile model took 212.86 ms
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
    [ INFO ]     PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: False
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 2.46 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            88380 iterations
    [ INFO ] Duration:         15003.08 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.82 ms
    [ INFO ]    Average:       1.83 ms
    [ INFO ]    Min:           1.20 ms
    [ INFO ]    Max:           18.41 ms
    [ INFO ] Throughput:   5890.79 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m "model/quantized_mobilenet_v2.xml" -d $device.value -api async -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 18.49 ms
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

    [ INFO ] Compile model took 316.79 ms
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
    [ INFO ]     PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: False
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.98 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            163416 iterations
    [ INFO ] Duration:         15001.85 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.02 ms
    [ INFO ]    Average:       1.05 ms
    [ INFO ]    Min:           0.70 ms
    [ INFO ]    Max:           7.76 ms
    [ INFO ] Throughput:   10893.06 FPS


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

