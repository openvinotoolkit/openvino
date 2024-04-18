Quantization of Image Classification Models
===========================================

This tutorial demonstrates how to apply ``INT8`` quantization to Image
Classification model using
`NNCF <https://github.com/openvinotoolkit/nncf>`__. It uses the
MobileNet V2 model, trained on Cifar10 dataset. The code is designed to
be extendable to custom models and datasets. The tutorial uses OpenVINO
backend for performing model quantization in NNCF, if you interested how
to apply quantization on PyTorch model, please check this
`tutorial <pytorch-post-training-quantization-nncf-with-output.html>`__.

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

    import platform

    # Install required packages
    %pip install -q "openvino>=2023.1.0" "nncf>=2.6.0" torch torchvision tqdm --extra-index-url https://download.pytorch.org/whl/cpu

    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path

    # Set the data and model directories
    DATA_DIR = Path("data")
    MODEL_DIR = Path("model")
    model_repo = "pytorch-cifar-models"

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

.. parsed-literal::

    Receiving objects:  77% (218/282)
Receiving objects:  78% (220/282)
Receiving objects:  79% (223/282)
Receiving objects:  80% (226/282)

.. parsed-literal::

    Receiving objects:  81% (229/282)
Receiving objects:  82% (232/282)
Receiving objects:  83% (235/282)

.. parsed-literal::

    Receiving objects:  84% (237/282)
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
Receiving objects: 100% (282/282), 9.22 MiB | 21.40 MiB/s, done.
    Resolving deltas:   0% (0/135)
Resolving deltas:   2% (4/135)
Resolving deltas:   4% (6/135)
Resolving deltas:   5% (7/135)
Resolving deltas:  14% (19/135)
Resolving deltas:  19% (26/135)
Resolving deltas:  20% (27/135)
Resolving deltas:  22% (31/135)
Resolving deltas:  25% (34/135)
Resolving deltas:  27% (37/135)
Resolving deltas:  28% (38/135)
Resolving deltas:  29% (40/135)
Resolving deltas:  30% (41/135)
Resolving deltas:  31% (42/135)
Resolving deltas:  32% (44/135)
Resolving deltas:  34% (47/135)
Resolving deltas:  40% (54/135)
Resolving deltas:  45% (62/135)
Resolving deltas:  46% (63/135)
Resolving deltas:  51% (69/135)
Resolving deltas:  57% (78/135)
Resolving deltas:  58% (79/135)
Resolving deltas:  59% (80/135)
Resolving deltas:  60% (82/135)
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

    ov_model = ov.convert_model(model, input=[1, 3, 32, 32])

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

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
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


  0%|          | 32768/170498071 [00:00<09:50, 288554.83it/s]

.. parsed-literal::


  0%|          | 65536/170498071 [00:00<10:02, 282666.29it/s]

.. parsed-literal::


  0%|          | 98304/170498071 [00:00<10:04, 281904.99it/s]

.. parsed-literal::


  0%|          | 229376/170498071 [00:00<04:37, 614087.82it/s]

.. parsed-literal::


  0%|          | 393216/170498071 [00:00<03:09, 898993.80it/s]

.. parsed-literal::


  0%|          | 819200/170498071 [00:00<01:32, 1832006.56it/s]

.. parsed-literal::


  1%|          | 1605632/170498071 [00:00<00:49, 3430452.29it/s]

.. parsed-literal::


  2%|▏         | 3211264/170498071 [00:00<00:24, 6709167.30it/s]

.. parsed-literal::


  4%|▎         | 6324224/170498071 [00:01<00:12, 12963647.37it/s]

.. parsed-literal::


  6%|▌         | 9994240/170498071 [00:01<00:08, 18035329.00it/s]

.. parsed-literal::


  8%|▊         | 13107200/170498071 [00:01<00:07, 20305770.58it/s]

.. parsed-literal::


 10%|▉         | 16285696/170498071 [00:01<00:06, 22104134.94it/s]

.. parsed-literal::


 11%|█▏        | 19365888/170498071 [00:01<00:06, 23124395.97it/s]

.. parsed-literal::


 13%|█▎        | 22478848/170498071 [00:01<00:06, 23948854.13it/s]

.. parsed-literal::


 15%|█▌        | 25722880/170498071 [00:01<00:05, 24802992.62it/s]

.. parsed-literal::


 17%|█▋        | 28835840/170498071 [00:01<00:05, 25083187.32it/s]

.. parsed-literal::


 18%|█▊        | 31358976/170498071 [00:02<00:07, 18085319.52it/s]

.. parsed-literal::


 21%|██        | 34963456/170498071 [00:02<00:06, 21073046.38it/s]

.. parsed-literal::


 22%|██▏       | 37322752/170498071 [00:02<00:08, 15989522.45it/s]

.. parsed-literal::


 24%|██▍       | 40566784/170498071 [00:02<00:06, 18605125.05it/s]

.. parsed-literal::


 26%|██▌       | 44630016/170498071 [00:02<00:05, 22608262.82it/s]

.. parsed-literal::


 28%|██▊       | 47874048/170498071 [00:02<00:05, 22840240.16it/s]

.. parsed-literal::


 30%|██▉       | 50429952/170498071 [00:03<00:06, 18450368.61it/s]

.. parsed-literal::


 31%|███       | 52723712/170498071 [00:03<00:06, 19207549.35it/s]

.. parsed-literal::


 33%|███▎      | 55574528/170498071 [00:03<00:05, 20537974.25it/s]

.. parsed-literal::


 34%|███▍      | 58327040/170498071 [00:03<00:05, 21165104.37it/s]

.. parsed-literal::


 36%|███▌      | 61243392/170498071 [00:03<00:04, 22232760.32it/s]

.. parsed-literal::


 38%|███▊      | 64159744/170498071 [00:03<00:04, 23057386.58it/s]

.. parsed-literal::


 39%|███▉      | 67108864/170498071 [00:03<00:04, 23641867.24it/s]

.. parsed-literal::


 41%|████      | 70057984/170498071 [00:03<00:04, 24152065.48it/s]

.. parsed-literal::


 43%|████▎     | 73039872/170498071 [00:04<00:03, 24552708.07it/s]

.. parsed-literal::


 45%|████▍     | 76054528/170498071 [00:04<00:03, 24853789.97it/s]

.. parsed-literal::


 46%|████▋     | 79101952/170498071 [00:04<00:03, 25166653.59it/s]

.. parsed-literal::


 48%|████▊     | 82149376/170498071 [00:04<00:03, 25402742.97it/s]

.. parsed-literal::


 50%|████▉     | 85196800/170498071 [00:04<00:03, 25611642.18it/s]

.. parsed-literal::


 52%|█████▏    | 88276992/170498071 [00:04<00:03, 25763124.59it/s]

.. parsed-literal::


 54%|█████▎    | 91324416/170498071 [00:04<00:03, 26057196.93it/s]

.. parsed-literal::


 55%|█████▌    | 94404608/170498071 [00:04<00:02, 27330510.47it/s]

.. parsed-literal::


 57%|█████▋    | 97157120/170498071 [00:04<00:02, 25996100.85it/s]

.. parsed-literal::


 59%|█████▊    | 99876864/170498071 [00:05<00:02, 25118577.70it/s]

.. parsed-literal::


 60%|██████    | 102989824/170498071 [00:05<00:02, 25307548.91it/s]

.. parsed-literal::


 62%|██████▏   | 106102784/170498071 [00:05<00:02, 25384602.06it/s]

.. parsed-literal::


 64%|██████▍   | 109150208/170498071 [00:05<00:02, 25329892.93it/s]

.. parsed-literal::


 66%|██████▌   | 112263168/170498071 [00:05<00:02, 25438506.31it/s]

.. parsed-literal::


 68%|██████▊   | 115441664/170498071 [00:05<00:02, 25587421.23it/s]

.. parsed-literal::


 70%|██████▉   | 118554624/170498071 [00:05<00:02, 25602205.09it/s]

.. parsed-literal::


 71%|███████▏  | 121733120/170498071 [00:05<00:01, 25724233.23it/s]

.. parsed-literal::


 73%|███████▎  | 124846080/170498071 [00:06<00:01, 25687741.69it/s]

.. parsed-literal::


 75%|███████▌  | 128024576/170498071 [00:06<00:01, 25727706.22it/s]

.. parsed-literal::


 77%|███████▋  | 131203072/170498071 [00:06<00:01, 25822105.27it/s]

.. parsed-literal::


 79%|███████▉  | 134381568/170498071 [00:06<00:01, 25850404.05it/s]

.. parsed-literal::


 81%|████████  | 137560064/170498071 [00:06<00:01, 25910266.42it/s]

.. parsed-literal::


 83%|████████▎ | 140738560/170498071 [00:06<00:01, 25956101.19it/s]

.. parsed-literal::


 84%|████████▍ | 143917056/170498071 [00:06<00:01, 25956043.80it/s]

.. parsed-literal::


 86%|████████▌ | 147030016/170498071 [00:06<00:00, 25801943.19it/s]

.. parsed-literal::


 88%|████████▊ | 150142976/170498071 [00:07<00:00, 25826195.79it/s]

.. parsed-literal::


 90%|████████▉ | 153321472/170498071 [00:07<00:00, 25694428.11it/s]

.. parsed-literal::


 92%|█████████▏| 156434432/170498071 [00:07<00:00, 25546020.81it/s]

.. parsed-literal::


 94%|█████████▎| 159481856/170498071 [00:07<00:00, 25542991.39it/s]

.. parsed-literal::


 95%|█████████▌| 162529280/170498071 [00:07<00:00, 25274374.04it/s]

.. parsed-literal::


 97%|█████████▋| 165085184/170498071 [00:07<00:00, 18172801.95it/s]

.. parsed-literal::


 98%|█████████▊| 167182336/170498071 [00:07<00:00, 17754138.87it/s]

.. parsed-literal::


 99%|█████████▉| 169148416/170498071 [00:08<00:00, 17753225.67it/s]

.. parsed-literal::


    100%|██████████| 170498071/170498071 [00:08<00:00, 21123642.42it/s]






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

    2024-04-17 23:59:04.099112: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-04-17 23:59:04.131641: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-04-17 23:59:04.768491: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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
        for batch_imgs, batch_labels in tqdm(data_loader):
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
        value="AUTO",
        description="Device:",
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


.. parsed-literal::

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
    [ INFO ] Read model took 9.91 ms
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

    [ INFO ] Compile model took 209.46 ms
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
    [ INFO ] First inference took 3.23 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            89376 iterations
    [ INFO ] Duration:         15002.82 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.83 ms
    [ INFO ]    Average:       1.84 ms
    [ INFO ]    Min:           1.22 ms
    [ INFO ]    Max:           8.55 ms
    [ INFO ] Throughput:   5957.28 FPS


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

    [ INFO ] Read model took 19.35 ms
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

    [ INFO ] Compile model took 356.68 ms
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
    [ INFO ] First inference took 2.03 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            165324 iterations
    [ INFO ] Duration:         15001.48 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.00 ms
    [ INFO ]    Average:       1.05 ms
    [ INFO ]    Min:           0.70 ms
    [ INFO ]    Max:           36.26 ms
    [ INFO ] Throughput:   11020.51 FPS


Compare results on four pictures
--------------------------------



.. code:: ipython3

    # Define all possible labels from the CIFAR10 dataset
    labels_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
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
        assert num_pics == 4, f"No enough indexes for pictures to be displayed, got {num_pics}"
        for idx in indexes:
            assert idx < 10000, "Cannot get such index, there are only 10000"
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
        """Inference model on a few pictures.
        :param net: model on which do inference
        :param indexes: list of indexes
        """
        output_key = model.output(0)
        predicted_labels = []
        for idx in indexes:
            assert idx < 10000, "Cannot get such index, there are only 10000"
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



.. image:: image-classification-quantization-with-output_files/image-classification-quantization-with-output_30_5.png

