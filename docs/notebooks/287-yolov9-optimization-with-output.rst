Convert and Optimize YOLOv9 with OpenVINO™
==========================================

YOLOv9 marks a significant advancement in real-time object detection,
introducing groundbreaking techniques such as Programmable Gradient
Information (PGI) and the Generalized Efficient Layer Aggregation
Network (GELAN). This model demonstrates remarkable improvements in
efficiency, accuracy, and adaptability, setting new benchmarks on the MS
COCO dataset. More details about model can be found in
`paper <https://arxiv.org/abs/2402.13616>`__ and `original
repository <https://github.com/WongKinYiu/yolov9>`__ This tutorial
demonstrates step-by-step instructions on how to run and optimize
PyTorch YOLO V9 with OpenVINO.

The tutorial consists of the following steps:

-  Prepare PyTorch model
-  Convert PyTorch model to OpenVINO IR
-  Run model inference with OpenVINO
-  Prepare and run optimization pipeline
-  Compare performance of the FP32 and quantized models.
-  Run optimized model inference on video

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Get PyTorch model <#get-pytorch-model>`__
-  `Convert PyTorch model to OpenVINO
   IR <#convert-pytorch-model-to-openvino-ir>`__
-  `Verify model inference <#verify-model-inference>`__

   -  `Preprocessing <#preprocessing>`__
   -  `Postprocessing <#postprocessing>`__
   -  `Select inference device <#select-inference-device>`__

-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__

   -  `Prepare dataset <#prepare-dataset>`__
   -  `Perform model quantization <#perform-model-quantization>`__

-  `Run quantized model inference <#run-quantized-model-inference>`__
-  `Compare Performance of the Original and Quantized
   Models <#compare-performance-of-the-original-and-quantized-models>`__
-  `Run Live Object Detection <#run-live-object-detection>`__

Prerequisites
-------------



.. code:: ipython3

    import platform

    %pip install -q "openvino>=2023.3.0" "nncf>=2.8.1" "opencv-python" "seaborn" "pandas" "scikit-learn" "torch" "torchvision"  --extra-index-url https://download.pytorch.org/whl/cpu

    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path
    sys.path.append("../utils")
    from notebook_utils import download_file

    if not Path('yolov9').exists():
        !git clone https://github.com/WongKinYiu/yolov9
    %cd yolov9


.. parsed-literal::

    Cloning into 'yolov9'...


.. parsed-literal::

    remote: Enumerating objects: 621, done.[K
    remote: Counting objects:   0% (1/238)[K
    remote: Counting objects:   1% (3/238)[K
    remote: Counting objects:   2% (5/238)[K
    remote: Counting objects:   3% (8/238)[K
    remote: Counting objects:   4% (10/238)[K
    remote: Counting objects:   5% (12/238)[K
    remote: Counting objects:   6% (15/238)[K
    remote: Counting objects:   7% (17/238)[K
    remote: Counting objects:   8% (20/238)[K
    remote: Counting objects:   9% (22/238)[K
    remote: Counting objects:  10% (24/238)[K
    remote: Counting objects:  11% (27/238)[K
    remote: Counting objects:  12% (29/238)[K
    remote: Counting objects:  13% (31/238)[K
    remote: Counting objects:  14% (34/238)[K
    remote: Counting objects:  15% (36/238)[K
    remote: Counting objects:  16% (39/238)[K
    remote: Counting objects:  17% (41/238)[K
    remote: Counting objects:  18% (43/238)[K
    remote: Counting objects:  19% (46/238)[K
    remote: Counting objects:  20% (48/238)[K
    remote: Counting objects:  21% (50/238)[K
    remote: Counting objects:  22% (53/238)[K
    remote: Counting objects:  23% (55/238)[K
    remote: Counting objects:  24% (58/238)[K
    remote: Counting objects:  25% (60/238)[K
    remote: Counting objects:  26% (62/238)[K
    remote: Counting objects:  27% (65/238)[K
    remote: Counting objects:  28% (67/238)[K
    remote: Counting objects:  29% (70/238)[K
    remote: Counting objects:  30% (72/238)[K
    remote: Counting objects:  31% (74/238)[K
    remote: Counting objects:  32% (77/238)[K
    remote: Counting objects:  33% (79/238)[K
    remote: Counting objects:  34% (81/238)[K
    remote: Counting objects:  35% (84/238)[K
    remote: Counting objects:  36% (86/238)[K
    remote: Counting objects:  37% (89/238)[K
    remote: Counting objects:  38% (91/238)[K
    remote: Counting objects:  39% (93/238)[K
    remote: Counting objects:  40% (96/238)[K
    remote: Counting objects:  41% (98/238)[K
    remote: Counting objects:  42% (100/238)[K
    remote: Counting objects:  43% (103/238)[K
    remote: Counting objects:  44% (105/238)[K
    remote: Counting objects:  45% (108/238)[K
    remote: Counting objects:  46% (110/238)[K
    remote: Counting objects:  47% (112/238)[K
    remote: Counting objects:  48% (115/238)[K
    remote: Counting objects:  49% (117/238)[K
    remote: Counting objects:  50% (119/238)[K
    remote: Counting objects:  51% (122/238)[K
    remote: Counting objects:  52% (124/238)[K
    remote: Counting objects:  53% (127/238)[K
    remote: Counting objects:  54% (129/238)[K
    remote: Counting objects:  55% (131/238)[K
    remote: Counting objects:  56% (134/238)[K
    remote: Counting objects:  57% (136/238)[K
    remote: Counting objects:  58% (139/238)[K
    remote: Counting objects:  59% (141/238)[K
    remote: Counting objects:  60% (143/238)[K
    remote: Counting objects:  61% (146/238)[K
    remote: Counting objects:  62% (148/238)[K
    remote: Counting objects:  63% (150/238)[K
    remote: Counting objects:  64% (153/238)[K
    remote: Counting objects:  65% (155/238)[K
    remote: Counting objects:  66% (158/238)[K
    remote: Counting objects:  67% (160/238)[K
    remote: Counting objects:  68% (162/238)[K
    remote: Counting objects:  69% (165/238)[K
    remote: Counting objects:  70% (167/238)[K
    remote: Counting objects:  71% (169/238)[K
    remote: Counting objects:  72% (172/238)[K
    remote: Counting objects:  73% (174/238)[K
    remote: Counting objects:  74% (177/238)[K
    remote: Counting objects:  75% (179/238)[K
    remote: Counting objects:  76% (181/238)[K
    remote: Counting objects:  77% (184/238)[K
    remote: Counting objects:  78% (186/238)[K
    remote: Counting objects:  79% (189/238)[K
    remote: Counting objects:  80% (191/238)[K
    remote: Counting objects:  81% (193/238)[K
    remote: Counting objects:  82% (196/238)[K
    remote: Counting objects:  83% (198/238)[K
    remote: Counting objects:  84% (200/238)[K
    remote: Counting objects:  85% (203/238)[K
    remote: Counting objects:  86% (205/238)[K
    remote: Counting objects:  87% (208/238)[K
    remote: Counting objects:  88% (210/238)[K
    remote: Counting objects:  89% (212/238)[K
    remote: Counting objects:  90% (215/238)[K
    remote: Counting objects:  91% (217/238)[K
    remote: Counting objects:  92% (219/238)[K
    remote: Counting objects:  93% (222/238)[K
    remote: Counting objects:  94% (224/238)[K
    remote: Counting objects:  95% (227/238)[K
    remote: Counting objects:  96% (229/238)[K
    remote: Counting objects:  97% (231/238)[K
    remote: Counting objects:  98% (234/238)[K
    remote: Counting objects:  99% (236/238)[K
    remote: Counting objects: 100% (238/238)[K
    remote: Counting objects: 100% (238/238), done.[K
    remote: Compressing objects:   0% (1/116)[K
    remote: Compressing objects:   1% (2/116)[K
    remote: Compressing objects:   2% (3/116)[K

.. parsed-literal::

    remote: Compressing objects:   3% (4/116)[K
    remote: Compressing objects:   4% (5/116)[K
    remote: Compressing objects:   5% (6/116)[K
    remote: Compressing objects:   6% (7/116)[K
    remote: Compressing objects:   7% (9/116)[K
    remote: Compressing objects:   8% (10/116)[K
    remote: Compressing objects:   9% (11/116)[K
    remote: Compressing objects:  10% (12/116)[K
    remote: Compressing objects:  11% (13/116)[K
    remote: Compressing objects:  12% (14/116)[K
    remote: Compressing objects:  13% (16/116)[K
    remote: Compressing objects:  14% (17/116)[K
    remote: Compressing objects:  15% (18/116)[K
    remote: Compressing objects:  16% (19/116)[K
    remote: Compressing objects:  17% (20/116)[K
    remote: Compressing objects:  18% (21/116)[K
    remote: Compressing objects:  19% (23/116)[K
    remote: Compressing objects:  20% (24/116)[K
    remote: Compressing objects:  21% (25/116)[K
    remote: Compressing objects:  22% (26/116)[K
    remote: Compressing objects:  23% (27/116)[K
    remote: Compressing objects:  24% (28/116)[K
    remote: Compressing objects:  25% (29/116)[K
    remote: Compressing objects:  26% (31/116)[K
    remote: Compressing objects:  27% (32/116)[K
    remote: Compressing objects:  28% (33/116)[K
    remote: Compressing objects:  29% (34/116)[K
    remote: Compressing objects:  30% (35/116)[K
    remote: Compressing objects:  31% (36/116)[K
    remote: Compressing objects:  32% (38/116)[K
    remote: Compressing objects:  33% (39/116)[K
    remote: Compressing objects:  34% (40/116)[K
    remote: Compressing objects:  35% (41/116)[K
    remote: Compressing objects:  36% (42/116)[K
    remote: Compressing objects:  37% (43/116)[K
    remote: Compressing objects:  38% (45/116)[K
    remote: Compressing objects:  39% (46/116)[K
    remote: Compressing objects:  40% (47/116)[K
    remote: Compressing objects:  41% (48/116)[K
    remote: Compressing objects:  42% (49/116)[K
    remote: Compressing objects:  43% (50/116)[K
    remote: Compressing objects:  44% (52/116)[K
    remote: Compressing objects:  45% (53/116)[K
    remote: Compressing objects:  46% (54/116)[K
    remote: Compressing objects:  47% (55/116)[K
    remote: Compressing objects:  48% (56/116)[K
    remote: Compressing objects:  49% (57/116)[K
    remote: Compressing objects:  50% (58/116)[K
    remote: Compressing objects:  51% (60/116)[K
    remote: Compressing objects:  52% (61/116)[K
    remote: Compressing objects:  53% (62/116)[K
    remote: Compressing objects:  54% (63/116)[K
    remote: Compressing objects:  55% (64/116)[K
    remote: Compressing objects:  56% (65/116)[K
    remote: Compressing objects:  57% (67/116)[K
    remote: Compressing objects:  58% (68/116)[K
    remote: Compressing objects:  59% (69/116)[K
    remote: Compressing objects:  60% (70/116)[K
    remote: Compressing objects:  61% (71/116)[K
    remote: Compressing objects:  62% (72/116)[K
    remote: Compressing objects:  63% (74/116)[K
    remote: Compressing objects:  64% (75/116)[K
    remote: Compressing objects:  65% (76/116)[K
    remote: Compressing objects:  66% (77/116)[K
    remote: Compressing objects:  67% (78/116)[K
    remote: Compressing objects:  68% (79/116)[K
    remote: Compressing objects:  69% (81/116)[K
    remote: Compressing objects:  70% (82/116)[K
    remote: Compressing objects:  71% (83/116)[K
    remote: Compressing objects:  72% (84/116)[K
    remote: Compressing objects:  73% (85/116)[K
    remote: Compressing objects:  74% (86/116)[K
    remote: Compressing objects:  75% (87/116)[K
    remote: Compressing objects:  76% (89/116)[K
    remote: Compressing objects:  77% (90/116)[K
    remote: Compressing objects:  78% (91/116)[K
    remote: Compressing objects:  79% (92/116)[K
    remote: Compressing objects:  80% (93/116)[K
    remote: Compressing objects:  81% (94/116)[K
    remote: Compressing objects:  82% (96/116)[K
    remote: Compressing objects:  83% (97/116)[K
    remote: Compressing objects:  84% (98/116)[K
    remote: Compressing objects:  85% (99/116)[K
    remote: Compressing objects:  86% (100/116)[K
    remote: Compressing objects:  87% (101/116)[K
    remote: Compressing objects:  88% (103/116)[K
    remote: Compressing objects:  89% (104/116)[K
    remote: Compressing objects:  90% (105/116)[K
    remote: Compressing objects:  91% (106/116)[K
    remote: Compressing objects:  92% (107/116)[K
    remote: Compressing objects:  93% (108/116)[K
    remote: Compressing objects:  94% (110/116)[K
    remote: Compressing objects:  95% (111/116)[K
    remote: Compressing objects:  96% (112/116)[K
    remote: Compressing objects:  97% (113/116)[K
    remote: Compressing objects:  98% (114/116)[K
    remote: Compressing objects:  99% (115/116)[K
    remote: Compressing objects: 100% (116/116)[K
    remote: Compressing objects: 100% (116/116), done.[K
    Receiving objects:   0% (1/621)

.. parsed-literal::

    Receiving objects:   1% (7/621)
    Receiving objects:   2% (13/621)
    Receiving objects:   3% (19/621)
    Receiving objects:   4% (25/621)
    Receiving objects:   5% (32/621)
    Receiving objects:   6% (38/621)
    Receiving objects:   7% (44/621)
    Receiving objects:   8% (50/621)

.. parsed-literal::

    Receiving objects:   9% (56/621)
    Receiving objects:  10% (63/621)
    Receiving objects:  11% (69/621)
    Receiving objects:  12% (75/621)
    Receiving objects:  13% (81/621)
    Receiving objects:  14% (87/621)
    Receiving objects:  15% (94/621)

.. parsed-literal::

    Receiving objects:  16% (100/621)
    Receiving objects:  17% (106/621)
    Receiving objects:  18% (112/621)
    Receiving objects:  19% (118/621)
    Receiving objects:  20% (125/621)
    Receiving objects:  21% (131/621)
    Receiving objects:  22% (137/621)
    Receiving objects:  23% (143/621)
    Receiving objects:  24% (150/621)
    Receiving objects:  25% (156/621)
    Receiving objects:  26% (162/621)
    Receiving objects:  27% (168/621)
    Receiving objects:  28% (174/621)
    Receiving objects:  29% (181/621)
    Receiving objects:  30% (187/621)
    Receiving objects:  31% (193/621)
    Receiving objects:  32% (199/621)
    Receiving objects:  33% (205/621)
    Receiving objects:  34% (212/621)
    Receiving objects:  35% (218/621)
    Receiving objects:  36% (224/621)
    Receiving objects:  37% (230/621)
    Receiving objects:  38% (236/621)
    Receiving objects:  39% (243/621)
    Receiving objects:  40% (249/621)
    Receiving objects:  41% (255/621)
    Receiving objects:  42% (261/621)
    Receiving objects:  43% (268/621)
    Receiving objects:  44% (274/621)
    Receiving objects:  45% (280/621)
    Receiving objects:  46% (286/621)
    Receiving objects:  47% (292/621)
    Receiving objects:  48% (299/621)
    Receiving objects:  49% (305/621)
    Receiving objects:  50% (311/621)
    Receiving objects:  51% (317/621)
    Receiving objects:  52% (323/621)
    Receiving objects:  53% (330/621)
    Receiving objects:  54% (336/621)
    Receiving objects:  55% (342/621)
    Receiving objects:  56% (348/621)
    Receiving objects:  57% (354/621)

.. parsed-literal::

    Receiving objects:  58% (361/621)
    Receiving objects:  59% (367/621)
    Receiving objects:  60% (373/621)
    Receiving objects:  61% (379/621)
    Receiving objects:  62% (386/621)
    Receiving objects:  63% (392/621)
    Receiving objects:  64% (398/621)
    Receiving objects:  65% (404/621)
    Receiving objects:  66% (410/621)
    Receiving objects:  67% (417/621)
    Receiving objects:  68% (423/621)
    Receiving objects:  69% (429/621)
    Receiving objects:  70% (435/621)
    Receiving objects:  71% (441/621)
    Receiving objects:  72% (448/621)
    Receiving objects:  73% (454/621)
    Receiving objects:  74% (460/621)
    Receiving objects:  75% (466/621)
    Receiving objects:  76% (472/621)
    Receiving objects:  77% (479/621)
    Receiving objects:  78% (485/621)
    Receiving objects:  79% (491/621)
    Receiving objects:  80% (497/621)
    Receiving objects:  81% (504/621)
    Receiving objects:  82% (510/621)
    Receiving objects:  83% (516/621)
    Receiving objects:  84% (522/621)
    Receiving objects:  85% (528/621)
    Receiving objects:  86% (535/621)
    remote: Total 621 (delta 186), reused 122 (delta 122), pack-reused 383[K
    Receiving objects:  87% (541/621)
    Receiving objects:  88% (547/621)
    Receiving objects:  89% (553/621)
    Receiving objects:  90% (559/621)
    Receiving objects:  91% (566/621)
    Receiving objects:  92% (572/621)
    Receiving objects:  93% (578/621)
    Receiving objects:  94% (584/621)
    Receiving objects:  95% (590/621)
    Receiving objects:  96% (597/621)
    Receiving objects:  97% (603/621)
    Receiving objects:  98% (609/621)
    Receiving objects:  99% (615/621)
    Receiving objects: 100% (621/621)
    Receiving objects: 100% (621/621), 3.21 MiB | 15.72 MiB/s, done.
    Resolving deltas:   0% (0/238)
    Resolving deltas:   1% (3/238)
    Resolving deltas:   2% (6/238)
    Resolving deltas:   3% (9/238)
    Resolving deltas:   4% (10/238)
    Resolving deltas:   6% (16/238)
    Resolving deltas:   7% (17/238)
    Resolving deltas:   8% (21/238)
    Resolving deltas:   9% (23/238)
    Resolving deltas:  10% (25/238)
    Resolving deltas:  11% (28/238)
    Resolving deltas:  12% (30/238)
    Resolving deltas:  13% (31/238)
    Resolving deltas:  14% (34/238)
    Resolving deltas:  15% (36/238)
    Resolving deltas:  16% (39/238)
    Resolving deltas:  20% (48/238)
    Resolving deltas:  27% (66/238)
    Resolving deltas:  30% (72/238)
    Resolving deltas:  31% (74/238)
    Resolving deltas:  39% (94/238)
    Resolving deltas:  46% (110/238)
    Resolving deltas:  58% (140/238)
    Resolving deltas:  59% (142/238)
    Resolving deltas:  60% (144/238)
    Resolving deltas:  64% (154/238)
    Resolving deltas:  65% (156/238)
    Resolving deltas:  70% (167/238)
    Resolving deltas:  73% (175/238)
    Resolving deltas:  75% (180/238)
    Resolving deltas:  76% (181/238)
    Resolving deltas:  78% (186/238)
    Resolving deltas:  79% (190/238)
    Resolving deltas:  85% (204/238)
    Resolving deltas:  89% (213/238)
    Resolving deltas:  91% (218/238)
    Resolving deltas:  94% (225/238)
    Resolving deltas:  95% (227/238)
    Resolving deltas:  97% (231/238)
    Resolving deltas:  99% (237/238)
    Resolving deltas: 100% (238/238)
    Resolving deltas: 100% (238/238), done.


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9


Get PyTorch model
-----------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the ``gelan-c`` (light-weight version of yolov9) model pre-trained on a
COCO dataset, which is available in this
`repo <https://github.com/WongKinYiu/yolov9>`__, but the same steps are
applicable for other models from YOLO V9 family.

.. code:: ipython3

    # Download pre-trained model weights
    MODEL_LINK = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt"
    DATA_DIR = Path("data/")
    MODEL_DIR = Path("model/")
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)



.. parsed-literal::

    model/gelan-c.pt:   0%|          | 0.00/49.1M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9/model/gelan-c.pt')



Convert PyTorch model to OpenVINO IR
------------------------------------



OpenVINO supports PyTorch model conversion via Model Conversion API.
``ov.convert_model`` function accepts model object and example input for
tracing the model and returns an instance of ``ov.Model``, representing
this model in OpenVINO format. The Obtained model is ready for loading
on specific devices or can be saved on disk for the next deployment
using ``ov.save_model``.

.. code:: ipython3

    from models.experimental import attempt_load
    import torch
    import openvino as ov
    from models.yolo import Detect, DualDDetect
    from utils.general import yaml_save, yaml_load

    weights = MODEL_DIR / "gelan-c.pt"
    ov_model_path = MODEL_DIR / weights.name.replace(".pt", "_openvino_model") / weights.name.replace(".pt", ".xml")

    if not ov_model_path.exists():
        model = attempt_load(weights, device="cpu", inplace=True, fuse=True)
        metadata = {'stride': int(max(model.stride)), 'names': model.names}

        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, (Detect, DualDDetect)):
                m.inplace = False
                m.dynamic = True
                m.export = True


        example_input = torch.zeros((1, 3, 640, 640))
        model(example_input)

        ov_model = ov.convert_model(model, example_input=example_input)

        # specify input and output names for compatibility with yolov9 repo interface
        ov_model.outputs[0].get_tensor().set_names({"output0"})
        ov_model.inputs[0].get_tensor().set_names({"images"})
        ov.save_model(ov_model, ov_model_path)
        # save metadata
        yaml_save(ov_model_path.parent / weights.name.replace(".pt", ".yaml"), metadata)
    else:
        metadata = yaml_load(ov_model_path.parent / weights.name.replace(".pt", ".yaml"))


.. parsed-literal::

    Fusing layers...


.. parsed-literal::

    Model summary: 387 layers, 25288768 parameters, 0 gradients, 102.1 GFLOPs


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9/models/yolo.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif self.dynamic or self.shape != shape:


Verify model inference
----------------------



To test model work, we create inference pipeline similar to
``detect.py``. The pipeline consists of preprocessing step, inference of
OpenVINO model, and results post-processing to get bounding boxes.

Preprocessing
~~~~~~~~~~~~~



Model input is a tensor with the ``[1, 3, 640, 640]`` shape in
``N, C, H, W`` format, where

-  ``N`` - number of images in batch (batch size)
-  ``C`` - image channels
-  ``H`` - image height
-  ``W`` - image width

Model expects images in RGB channels format and normalized in [0, 1]
range. To resize images to fit model size ``letterbox`` resize approach
is used where the aspect ratio of width and height is preserved. It is
defined in yolov9 repository.

To keep specific shape, preprocessing automatically enables padding.

.. code:: ipython3

    import numpy as np
    import torch
    from PIL import Image
    from utils.augmentations import letterbox

    image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/7b6af406-4ccb-4ded-a13d-62b7c0e42e96"
    download_file(image_url, directory=DATA_DIR, filename="test_image.jpg", show_progress=True)

    def preprocess_image(img0: np.ndarray):
        """
        Preprocess image according to YOLOv9 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.

        Parameters:
          img0 (np.ndarray): image for preprocessing
        Returns:
          img (np.ndarray): image after preprocessing
          img0 (np.ndarray): original image
        """
        # resize
        img = letterbox(img0, auto=False)[0]

        # Convert
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0


    def prepare_input_tensor(image: np.ndarray):
        """
        Converts preprocessed image to tensor format according to YOLOv9 input requirements.
        Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

        Parameters:
          image (np.ndarray): image for conversion to tensor
        Returns:
          input_tensor (torch.Tensor): float tensor ready to use for YOLOv9 inference
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp16/32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor

    NAMES = metadata["names"]



.. parsed-literal::

    data/test_image.jpg:   0%|          | 0.00/101k [00:00<?, ?B/s]


Postprocessing
~~~~~~~~~~~~~~



Model output contains detection boxes candidates. It is a tensor with
the ``[1,25200,85]`` shape in the ``B, N, 85`` format, where:

-  ``B`` - batch size
-  ``N`` - number of detection boxes

Detection box has the [``x``, ``y``, ``h``, ``w``, ``box_score``,
``class_no_1``, …, ``class_no_80``] format, where:

-  (``x``, ``y``) - raw coordinates of box center
-  ``h``, ``w`` - raw height and width of box
-  ``box_score`` - confidence of detection box
-  ``class_no_1``, …, ``class_no_80`` - probability distribution over
   the classes.

For getting final prediction, we need to apply non maximum suppression
algorithm and rescale boxes coordinates to original image size.

.. code:: ipython3

    from utils.plots import Annotator, colors

    from typing import List, Tuple
    from utils.general import scale_boxes, non_max_suppression


    def detect(model: ov.Model, image_path: Path, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):
        """
        OpenVINO YOLOv9 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
        Parameters:
            model (Model): OpenVINO compiled model.
            image_path (Path): input image path.
            conf_thres (float, *optional*, 0.25): minimal accepted confidence for object filtering
            iou_thres (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
            agnostic_nms (bool, *optional*, False): apply class agnostic NMS approach or not
        Returns:
           pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
           orig_img (np.ndarray): image before preprocessing, can be used for results visualization
           inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
        """
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = np.array(Image.open(image_path))
        preprocessed_img, orig_img = preprocess_image(img)
        input_tensor = prepare_input_tensor(preprocessed_img)
        predictions = torch.from_numpy(model(input_tensor)[0])
        pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        return pred, orig_img, input_tensor.shape


    def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str]):
        """
        Utility function for drawing predicted bounding boxes on image
        Parameters:
            predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
            image (np.ndarray): image for boxes visualization
            names (List[str]): list of names for each class in dataset
            colors (Dict[str, int]): mapping between class name and drawing color
        Returns:
            image (np.ndarray): box visualization result
        """
        if not len(predictions):
            return image

        annotator = Annotator(image, line_width=1, example=str(names))
        # Rescale boxes from input size to original image size
        predictions[:, :4] = scale_boxes(input_shape[2:], predictions[:, :4], image.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(predictions):
            label = f'{names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
        return image

.. code:: ipython3

    core = ov.Core()
    # read converted model
    ov_model = core.read_model(ov_model_path)

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



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

    # load model on selected device
    if device.value != "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})
    compiled_model = core.compile_model(ov_model, device.value)

.. code:: ipython3

    boxes, image, input_shape = detect(compiled_model, DATA_DIR / "test_image.jpg")
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES)
    # visualize results
    Image.fromarray(image_with_boxes)




.. image:: 287-yolov9-optimization-with-output_files/287-yolov9-optimization-with-output_16_0.png



Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
YOLOv9. The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``ov.save_model`` function.

Prepare dataset
~~~~~~~~~~~~~~~



The code below downloads COCO dataset and prepares a dataloader that is
used to evaluate the yolov9 model accuracy. We reuse its subset for
quantization.

.. code:: ipython3

    from zipfile import ZipFile

    sys.path.append("../../utils")
    from notebook_utils import download_file


    DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
    LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"

    OUT_DIR = Path('.')

    download_file(DATA_URL, directory=OUT_DIR, show_progress=True)
    download_file(LABELS_URL, directory=OUT_DIR, show_progress=True)

    if not (OUT_DIR / "coco/labels").exists():
        with ZipFile('coco2017labels-segments.zip' , "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
        with ZipFile('val2017.zip' , "r") as zip_ref:
            zip_ref.extractall(OUT_DIR / 'coco/images')



.. parsed-literal::

    val2017.zip:   0%|          | 0.00/778M [00:00<?, ?B/s]



.. parsed-literal::

    coco2017labels-segments.zip:   0%|          | 0.00/169M [00:00<?, ?B/s]


.. code:: ipython3

    from collections import namedtuple
    import yaml
    from utils.dataloaders import create_dataloader
    from utils.general import colorstr

    # read dataset config
    DATA_CONFIG = 'data/coco.yaml'
    with open(DATA_CONFIG) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Dataloader
    TASK = 'val'  # path to train/val/test images
    Option = namedtuple('Options', ['single_cls'])  # imitation of commandline provided options for single class evaluation
    opt = Option(False)
    dataloader = create_dataloader(
        str(Path("coco") / data[TASK]), 640, 1, 32, opt, pad=0.5,
        prefix=colorstr(f'{TASK}: ')
    )[0]


.. parsed-literal::


    val: Scanning coco/val2017...:   0%|          | 0/5000 00:00

.. parsed-literal::


    val: Scanning coco/val2017... 839 images, 7 backgrounds, 0 corrupt:  17%|█▋        | 846/5000 00:00

.. parsed-literal::


    val: Scanning coco/val2017... 1889 images, 14 backgrounds, 0 corrupt:  38%|███▊      | 1903/5000 00:00

.. parsed-literal::


    val: Scanning coco/val2017... 3040 images, 29 backgrounds, 0 corrupt:  61%|██████▏   | 3069/5000 00:00

.. parsed-literal::


    val: Scanning coco/val2017... 4373 images, 41 backgrounds, 0 corrupt:  88%|████████▊ | 4414/5000 00:00

.. parsed-literal::


    val: Scanning coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00






.. parsed-literal::

    val: New cache created: coco/val2017.cache


NNCF provides ``nncf.Dataset`` wrapper for using native framework
dataloaders in quantization pipeline. Additionally, we specify transform
function that will be responsible for preparing input data in model
expected format.

.. code:: ipython3

    import nncf


    def transform_fn(data_item):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
        Parameters:
           data_item: Tuple with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        img = data_item[0].numpy()
        input_tensor = prepare_input_tensor(img)
        return input_tensor


    quantization_dataset = nncf.Dataset(dataloader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Perform model quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~



The ``nncf.quantize`` function provides an interface for model
quantization. It requires an instance of the OpenVINO Model and
quantization dataset. Optionally, some additional parameters for the
configuration quantization process (number of samples for quantization,
preset, ignored scope etc.) can be provided. YOLOv9 model contains
non-ReLU activation functions, which require asymmetric quantization of
activations. To achieve better results, we will use a ``mixed``
quantization preset. It provides symmetric quantization of weights and
asymmetric quantization of activations.

.. code:: ipython3

    ov_int8_model_path = MODEL_DIR / weights.name.replace(".pt","_int8_openvino_model") / weights.name.replace(".pt", "_int8.xml")

    if not ov_int8_model_path.exists():
        quantized_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

        ov.save_model(quantized_model, ov_int8_model_path)
        yaml_save(ov_int8_model_path.parent / weights.name.replace(".pt", "_int8.yaml"), metadata)


.. parsed-literal::

    2024-03-26 00:40:01.790402: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-26 00:40:01.823619: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-26 00:40:02.585673: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:84: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Run quantized model inference
-----------------------------



There are no changes in model usage after applying quantization. Let’s
check the model work on the previously used image.

.. code:: ipython3

    quantized_model = core.read_model(ov_int8_model_path)

    if device.value != "CPU":
        quantized_model.reshape({0: [1, 3, 640, 640]})

    compiled_model = core.compile_model(quantized_model, device.value)

.. code:: ipython3

    boxes, image, input_shape = detect(compiled_model, DATA_DIR / "test_image.jpg")
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES)
    # visualize results
    Image.fromarray(image_with_boxes)




.. image:: 287-yolov9-optimization-with-output_files/287-yolov9-optimization-with-output_27_0.png



Compare Performance of the Original and Quantized Models
--------------------------------------------------------



We use the OpenVINO `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
to measure the inference performance of the ``FP32`` and ``INT8``
models.

   **NOTE**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    !benchmark_app -m $ov_model_path -shape "[1,3,640,640]" -d $device.value -api async -t 15


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
    [ INFO ] Read model took 26.57 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [?,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [?,144,4..,4..]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [?,144,2..,2..]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [?,144,1..,1..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 8.27 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 561.11 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:


.. parsed-literal::

    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
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
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 186.65 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            228 iterations
    [ INFO ] Duration:         15553.98 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        409.81 ms
    [ INFO ]    Average:       407.12 ms
    [ INFO ]    Min:           323.15 ms
    [ INFO ]    Max:           422.59 ms
    [ INFO ] Throughput:   14.66 FPS


.. code:: ipython3

    !benchmark_app -m $ov_int8_model_path -shape "[1,3,640,640]" -d $device.value -api async -t 15


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

    [ INFO ] Read model took 51.34 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 0.04 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 1178.16 ms
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
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
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
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values


.. parsed-literal::

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 75.14 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            750 iterations
    [ INFO ] Duration:         15097.75 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        120.89 ms
    [ INFO ]    Average:       120.36 ms
    [ INFO ]    Min:           87.19 ms
    [ INFO ]    Max:           133.11 ms
    [ INFO ] Throughput:   49.68 FPS


Run Live Object Detection
-------------------------



.. code:: ipython3

    import collections
    import time
    from IPython import display
    from notebook_utils import VideoPlayer
    import cv2


    # Main processing function to run object detection.
    def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0, model=ov_model, device=device.value):
        player = None
        compiled_model = core.compile_model(model, device)
        try:
            # Create a video player to play with target fps.
            player = VideoPlayer(
                source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
            )
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(
                    winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
                )

            processing_times = collections.deque()
            while True:
                # Grab the frame.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
                scale = 1280 / max(frame.shape)
                if scale < 1:
                    frame = cv2.resize(
                        src=frame,
                        dsize=None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                # Get the results.
                input_image = np.array(frame)

                start_time = time.time()
                # model expects RGB image, while video capturing in BGR
                detections, _, input_shape = detect(compiled_model, input_image[:, :, ::-1])
                stop_time = time.time()

                image_with_boxes = draw_boxes(detections[0], input_shape, input_image, NAMES)
                frame = image_with_boxes

                processing_times.append(stop_time - start_time)
                # Use processing times from last 200 frames.
                if len(processing_times) > 200:
                    processing_times.popleft()

                _, f_width = frame.shape[:2]
                # Mean processing time [ms].
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
                cv2.putText(
                    img=frame,
                    text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=f_width / 1000,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                # Use this workaround if there is flickering.
                if use_popup:
                    cv2.imshow(winname=title, mat=frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(
                        ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
                    # Create an IPython image.⬆️
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

Use a webcam as the video input. By default, the primary webcam is set
with \ ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set \ ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering,
set \ ``use_popup=True``.

   **NOTE**: To use this notebook with a webcam, you need to run the
   notebook on a computer with a webcam. If you run the notebook on a
   remote server (for example, in Binder or Google Colab service), the
   webcam will not work. By default, the lower cell will run model
   inference on a video file. If you want to try live inference on your
   webcam set ``WEBCAM_INFERENCE = True``

Run the object detection:

.. code:: ipython3

    WEBCAM_INFERENCE = False

    if WEBCAM_INFERENCE:
        VIDEO_SOURCE = 0  # Webcam
    else:
        VIDEO_SOURCE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    quantized_model = core.read_model(ov_int8_model_path)

    run_object_detection(source=VIDEO_SOURCE, flip=True, use_popup=False, model=quantized_model, device=device.value)



.. image:: 287-yolov9-optimization-with-output_files/287-yolov9-optimization-with-output_36_0.png


.. parsed-literal::

    Source ended

