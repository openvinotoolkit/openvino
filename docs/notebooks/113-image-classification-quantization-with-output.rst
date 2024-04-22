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

.. parsed-literal::

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

    Receiving objects:  79% (223/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  80% (226/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  81% (229/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  82% (232/282), 2.34 MiB | 4.38 MiB/s
Receiving objects:  83% (235/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  84% (237/282), 2.34 MiB | 4.38 MiB/s
Receiving objects:  85% (240/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  86% (243/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  87% (246/282), 2.34 MiB | 4.38 MiB/s
Receiving objects:  88% (249/282), 2.34 MiB | 4.38 MiB/s

.. parsed-literal::

    Receiving objects:  89% (251/282), 8.63 MiB | 8.35 MiB/s
remote: Total 282 (delta 135), reused 269 (delta 128), pack-reused 1[K
    Receiving objects:  90% (254/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  91% (257/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  92% (260/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  93% (263/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  94% (266/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  95% (268/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  96% (271/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  97% (274/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  98% (277/282), 8.63 MiB | 8.35 MiB/s
Receiving objects:  99% (280/282), 8.63 MiB | 8.35 MiB/s
Receiving objects: 100% (282/282), 8.63 MiB | 8.35 MiB/s
Receiving objects: 100% (282/282), 9.22 MiB | 8.26 MiB/s, done.
    Resolving deltas:   0% (0/135)
Resolving deltas:   3% (5/135)
Resolving deltas:   5% (7/135)
Resolving deltas:   6% (9/135)
Resolving deltas:  13% (18/135)
Resolving deltas:  21% (29/135)
Resolving deltas:  25% (34/135)
Resolving deltas:  29% (40/135)
Resolving deltas:  30% (41/135)
Resolving deltas:  31% (42/135)
Resolving deltas:  32% (44/135)
Resolving deltas:  33% (45/135)
Resolving deltas:  34% (46/135)
Resolving deltas:  40% (54/135)
Resolving deltas:  45% (62/135)
Resolving deltas:  46% (63/135)
Resolving deltas:  50% (68/135)
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


  0%|          | 32768/170498071 [00:00<10:17, 275946.37it/s]

.. parsed-literal::


  0%|          | 65536/170498071 [00:00<10:28, 271284.88it/s]

.. parsed-literal::


  0%|          | 98304/170498071 [00:00<10:31, 269970.74it/s]

.. parsed-literal::


  0%|          | 229376/170498071 [00:00<04:50, 586473.34it/s]

.. parsed-literal::


  0%|          | 393216/170498071 [00:00<03:17, 859242.88it/s]

.. parsed-literal::


  0%|          | 786432/170498071 [00:00<01:42, 1661012.29it/s]

.. parsed-literal::


  1%|          | 1507328/170498071 [00:00<00:55, 3048520.02it/s]

.. parsed-literal::


  2%|▏         | 3014656/170498071 [00:00<00:28, 5961632.12it/s]

.. parsed-literal::


  3%|▎         | 5963776/170498071 [00:01<00:14, 11664878.14it/s]

.. parsed-literal::


  6%|▌         | 9502720/170498071 [00:01<00:09, 16415065.07it/s]

.. parsed-literal::


  7%|▋         | 12484608/170498071 [00:01<00:08, 18713452.56it/s]

.. parsed-literal::


  9%|▉         | 15532032/170498071 [00:01<00:07, 20509245.76it/s]

.. parsed-literal::


 11%|█         | 18513920/170498071 [00:01<00:07, 21529733.73it/s]

.. parsed-literal::


 13%|█▎        | 21692416/170498071 [00:01<00:06, 22662379.12it/s]

.. parsed-literal::


 15%|█▍        | 24805376/170498071 [00:01<00:06, 23237932.55it/s]

.. parsed-literal::


 16%|█▋        | 28016640/170498071 [00:01<00:05, 23916419.04it/s]

.. parsed-literal::


 18%|█▊        | 31195136/170498071 [00:02<00:05, 24165013.75it/s]

.. parsed-literal::


 20%|██        | 34373632/170498071 [00:02<00:05, 24570741.60it/s]

.. parsed-literal::


 22%|██▏       | 37617664/170498071 [00:02<00:05, 24839298.54it/s]

.. parsed-literal::


 24%|██▍       | 40763392/170498071 [00:02<00:05, 24899513.26it/s]

.. parsed-literal::


 26%|██▌       | 43909120/170498071 [00:02<00:05, 24859956.00it/s]

.. parsed-literal::


 28%|██▊       | 47054848/170498071 [00:02<00:04, 24910085.77it/s]

.. parsed-literal::


 30%|██▉       | 50462720/170498071 [00:02<00:04, 25342462.67it/s]

.. parsed-literal::


 32%|███▏      | 53772288/170498071 [00:03<00:04, 25549574.14it/s]

.. parsed-literal::


 33%|███▎      | 56983552/170498071 [00:03<00:04, 25497496.83it/s]

.. parsed-literal::


 35%|███▌      | 60129280/170498071 [00:03<00:04, 25318471.74it/s]

.. parsed-literal::


 37%|███▋      | 63864832/170498071 [00:03<00:04, 26188505.17it/s]

.. parsed-literal::


 39%|███▉      | 67076096/170498071 [00:03<00:04, 25607940.52it/s]

.. parsed-literal::


 41%|████      | 70221824/170498071 [00:03<00:03, 25164439.49it/s]

.. parsed-literal::


 43%|████▎     | 73269248/170498071 [00:03<00:03, 24814859.91it/s]

.. parsed-literal::


 45%|████▍     | 76316672/170498071 [00:03<00:03, 24363754.63it/s]

.. parsed-literal::


 46%|████▋     | 79233024/170498071 [00:04<00:03, 23924798.97it/s]

.. parsed-literal::


 48%|████▊     | 82247680/170498071 [00:04<00:03, 23837747.76it/s]

.. parsed-literal::


 50%|█████     | 85426176/170498071 [00:04<00:03, 24006567.90it/s]

.. parsed-literal::


 52%|█████▏    | 88539136/170498071 [00:04<00:03, 24061028.14it/s]

.. parsed-literal::


 54%|█████▍    | 91783168/170498071 [00:04<00:03, 24193022.36it/s]

.. parsed-literal::


 56%|█████▌    | 94863360/170498071 [00:04<00:03, 24172569.31it/s]

.. parsed-literal::


 57%|█████▋    | 97943552/170498071 [00:04<00:02, 24211561.34it/s]

.. parsed-literal::


 59%|█████▉    | 101023744/170498071 [00:04<00:02, 24311644.55it/s]

.. parsed-literal::


 61%|██████    | 104071168/170498071 [00:05<00:02, 24218216.60it/s]

.. parsed-literal::


 63%|██████▎   | 107151360/170498071 [00:05<00:02, 24154999.17it/s]

.. parsed-literal::


 65%|██████▍   | 110133248/170498071 [00:05<00:02, 24021903.66it/s]

.. parsed-literal::


 66%|██████▋   | 113016832/170498071 [00:05<00:02, 23700128.22it/s]

.. parsed-literal::


 68%|██████▊   | 116162560/170498071 [00:05<00:02, 23748863.27it/s]

.. parsed-literal::


 70%|██████▉   | 119242752/170498071 [00:05<00:02, 23908673.56it/s]

.. parsed-literal::


 72%|███████▏  | 122355712/170498071 [00:05<00:02, 23981194.75it/s]

.. parsed-literal::


 74%|███████▎  | 125403136/170498071 [00:05<00:01, 23999264.79it/s]

.. parsed-literal::


 75%|███████▌  | 128581632/170498071 [00:06<00:01, 24304776.88it/s]

.. parsed-literal::


 77%|███████▋  | 131596288/170498071 [00:06<00:01, 24188415.22it/s]

.. parsed-literal::


 79%|███████▉  | 134709248/170498071 [00:06<00:01, 24194648.52it/s]

.. parsed-literal::


 81%|████████  | 137756672/170498071 [00:06<00:01, 24193549.64it/s]

.. parsed-literal::


 83%|████████▎ | 141066240/170498071 [00:06<00:01, 24630062.43it/s]

.. parsed-literal::


 85%|████████▍ | 144113664/170498071 [00:06<00:01, 24505080.69it/s]

.. parsed-literal::


 86%|████████▋ | 147062784/170498071 [00:06<00:00, 24261709.76it/s]

.. parsed-literal::


 88%|████████▊ | 150044672/170498071 [00:06<00:00, 24102172.86it/s]

.. parsed-literal::


 90%|████████▉ | 153059328/170498071 [00:07<00:00, 23913295.27it/s]

.. parsed-literal::


 92%|█████████▏| 156041216/170498071 [00:07<00:00, 23659468.91it/s]

.. parsed-literal::


 93%|█████████▎| 158892032/170498071 [00:07<00:00, 23391139.06it/s]

.. parsed-literal::


 95%|█████████▍| 161972224/170498071 [00:07<00:00, 23475130.31it/s]

.. parsed-literal::


 97%|█████████▋| 164954112/170498071 [00:07<00:00, 23499014.41it/s]

.. parsed-literal::


 99%|█████████▊| 168001536/170498071 [00:07<00:00, 23679867.24it/s]

.. parsed-literal::


   100%|██████████| 170498071/170498071 [00:07<00:00, 21918964.93it/s]






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

    2024-03-12 22:43:46.333703: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-12 22:43:46.368061: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-12 22:43:46.930366: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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


.. parsed-literal::

    [ INFO ] Read model took 10.11 ms
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

    [ INFO ] Compile model took 197.10 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:


.. parsed-literal::

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
    [ INFO ] First inference took 3.17 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            88452 iterations
    [ INFO ] Duration:         15002.60 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.86 ms
    [ INFO ]    Average:       1.86 ms
    [ INFO ]    Min:           1.57 ms
    [ INFO ]    Max:           8.68 ms
    [ INFO ] Throughput:   5895.78 FPS


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
    [ INFO ] Read model took 19.04 ms
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

    [ INFO ] Compile model took 335.52 ms
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
    [ INFO ] First inference took 2.10 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            167856 iterations
    [ INFO ] Duration:         15001.55 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1.00 ms
    [ INFO ]    Average:       1.03 ms
    [ INFO ]    Min:           0.68 ms
    [ INFO ]    Max:           6.97 ms
    [ INFO ] Throughput:   11189.25 FPS


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

