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
-  Run optimized model inference on video #### Table of contents:

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

## Prerequisites

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

    remote: Enumerating objects: 572, done.[K
    remote: Counting objects:   0% (1/192)[K
remote: Counting objects:   1% (2/192)[K
remote: Counting objects:   2% (4/192)[K
remote: Counting objects:   3% (6/192)[K
remote: Counting objects:   4% (8/192)[K
remote: Counting objects:   5% (10/192)[K
remote: Counting objects:   6% (12/192)[K
remote: Counting objects:   7% (14/192)[K
remote: Counting objects:   8% (16/192)[K
remote: Counting objects:   9% (18/192)[K
remote: Counting objects:  10% (20/192)[K
remote: Counting objects:  11% (22/192)[K
remote: Counting objects:  12% (24/192)[K
remote: Counting objects:  13% (25/192)[K
remote: Counting objects:  14% (27/192)[K
remote: Counting objects:  15% (29/192)[K
remote: Counting objects:  16% (31/192)[K
remote: Counting objects:  17% (33/192)[K
remote: Counting objects:  18% (35/192)[K
remote: Counting objects:  19% (37/192)[K
remote: Counting objects:  20% (39/192)[K
remote: Counting objects:  21% (41/192)[K
remote: Counting objects:  22% (43/192)[K
remote: Counting objects:  23% (45/192)[K
remote: Counting objects:  24% (47/192)[K
remote: Counting objects:  25% (48/192)[K
remote: Counting objects:  26% (50/192)[K
remote: Counting objects:  27% (52/192)[K
remote: Counting objects:  28% (54/192)[K
remote: Counting objects:  29% (56/192)[K
remote: Counting objects:  30% (58/192)[K
remote: Counting objects:  31% (60/192)[K
remote: Counting objects:  32% (62/192)[K
remote: Counting objects:  33% (64/192)[K
remote: Counting objects:  34% (66/192)[K
remote: Counting objects:  35% (68/192)[K
remote: Counting objects:  36% (70/192)[K
remote: Counting objects:  37% (72/192)[K
remote: Counting objects:  38% (73/192)[K
remote: Counting objects:  39% (75/192)[K
remote: Counting objects:  40% (77/192)[K
remote: Counting objects:  41% (79/192)[K
remote: Counting objects:  42% (81/192)[K
remote: Counting objects:  43% (83/192)[K
remote: Counting objects:  44% (85/192)[K
remote: Counting objects:  45% (87/192)[K
remote: Counting objects:  46% (89/192)[K
remote: Counting objects:  47% (91/192)[K
remote: Counting objects:  48% (93/192)[K
remote: Counting objects:  49% (95/192)[K
remote: Counting objects:  50% (96/192)[K
remote: Counting objects:  51% (98/192)[K
remote: Counting objects:  52% (100/192)[K
remote: Counting objects:  53% (102/192)[K
remote: Counting objects:  54% (104/192)[K
remote: Counting objects:  55% (106/192)[K
remote: Counting objects:  56% (108/192)[K
remote: Counting objects:  57% (110/192)[K
remote: Counting objects:  58% (112/192)[K
remote: Counting objects:  59% (114/192)[K
remote: Counting objects:  60% (116/192)[K
remote: Counting objects:  61% (118/192)[K
remote: Counting objects:  62% (120/192)[K
remote: Counting objects:  63% (121/192)[K
remote: Counting objects:  64% (123/192)[K
remote: Counting objects:  65% (125/192)[K
remote: Counting objects:  66% (127/192)[K
remote: Counting objects:  67% (129/192)[K
remote: Counting objects:  68% (131/192)[K
remote: Counting objects:  69% (133/192)[K
remote: Counting objects:  70% (135/192)[K
remote: Counting objects:  71% (137/192)[K
remote: Counting objects:  72% (139/192)[K
remote: Counting objects:  73% (141/192)[K
remote: Counting objects:  74% (143/192)[K
remote: Counting objects:  75% (144/192)[K
remote: Counting objects:  76% (146/192)[K
remote: Counting objects:  77% (148/192)[K
remote: Counting objects:  78% (150/192)[K
remote: Counting objects:  79% (152/192)[K
remote: Counting objects:  80% (154/192)[K
remote: Counting objects:  81% (156/192)[K
remote: Counting objects:  82% (158/192)[K
remote: Counting objects:  83% (160/192)[K
remote: Counting objects:  84% (162/192)[K
remote: Counting objects:  85% (164/192)[K
remote: Counting objects:  86% (166/192)[K
remote: Counting objects:  87% (168/192)[K
remote: Counting objects:  88% (169/192)[K
remote: Counting objects:  89% (171/192)[K
remote: Counting objects:  90% (173/192)[K
remote: Counting objects:  91% (175/192)[K
remote: Counting objects:  92% (177/192)[K
remote: Counting objects:  93% (179/192)[K
remote: Counting objects:  94% (181/192)[K
remote: Counting objects:  95% (183/192)[K
remote: Counting objects:  96% (185/192)[K
remote: Counting objects:  97% (187/192)[K
remote: Counting objects:  98% (189/192)[K
remote: Counting objects:  99% (191/192)[K
remote: Counting objects: 100% (192/192)[K
remote: Counting objects: 100% (192/192), done.[K
    remote: Compressing objects:   1% (1/84)[K
remote: Compressing objects:   2% (2/84)[K
remote: Compressing objects:   3% (3/84)[K
remote: Compressing objects:   4% (4/84)[K
remote: Compressing objects:   5% (5/84)[K
remote: Compressing objects:   7% (6/84)[K
remote: Compressing objects:   8% (7/84)[K
remote: Compressing objects:   9% (8/84)[K
remote: Compressing objects:  10% (9/84)[K
remote: Compressing objects:  11% (10/84)[K
remote: Compressing objects:  13% (11/84)[K
remote: Compressing objects:  14% (12/84)[K
remote: Compressing objects:  15% (13/84)[K
remote: Compressing objects:  16% (14/84)[K
remote: Compressing objects:  17% (15/84)[K
remote: Compressing objects:  19% (16/84)[K
remote: Compressing objects:  20% (17/84)[K
remote: Compressing objects:  21% (18/84)[K
remote: Compressing objects:  22% (19/84)[K
remote: Compressing objects:  23% (20/84)[K
remote: Compressing objects:  25% (21/84)[K
remote: Compressing objects:  26% (22/84)[K
remote: Compressing objects:  27% (23/84)[K
remote: Compressing objects:  28% (24/84)[K
remote: Compressing objects:  29% (25/84)[K
remote: Compressing objects:  30% (26/84)[K
remote: Compressing objects:  32% (27/84)[K
remote: Compressing objects:  33% (28/84)[K
remote: Compressing objects:  34% (29/84)[K
remote: Compressing objects:  35% (30/84)[K
remote: Compressing objects:  36% (31/84)[K
remote: Compressing objects:  38% (32/84)[K
remote: Compressing objects:  39% (33/84)[K
remote: Compressing objects:  40% (34/84)[K
remote: Compressing objects:  41% (35/84)[K
remote: Compressing objects:  42% (36/84)[K
remote: Compressing objects:  44% (37/84)[K
remote: Compressing objects:  45% (38/84)[K
remote: Compressing objects:  46% (39/84)[K
remote: Compressing objects:  47% (40/84)[K
remote: Compressing objects:  48% (41/84)[K
remote: Compressing objects:  50% (42/84)[K
remote: Compressing objects:  51% (43/84)[K
remote: Compressing objects:  52% (44/84)[K
remote: Compressing objects:  53% (45/84)[K
remote: Compressing objects:  54% (46/84)[K
remote: Compressing objects:  55% (47/84)[K
remote: Compressing objects:  57% (48/84)[K
remote: Compressing objects:  58% (49/84)[K
remote: Compressing objects:  59% (50/84)[K
remote: Compressing objects:  60% (51/84)[K
remote: Compressing objects:  61% (52/84)[K
remote: Compressing objects:  63% (53/84)[K
remote: Compressing objects:  64% (54/84)[K
remote: Compressing objects:  65% (55/84)[K
remote: Compressing objects:  66% (56/84)[K
remote: Compressing objects:  67% (57/84)[K
remote: Compressing objects:  69% (58/84)[K
remote: Compressing objects:  70% (59/84)[K
remote: Compressing objects:  71% (60/84)[K
remote: Compressing objects:  72% (61/84)[K
remote: Compressing objects:  73% (62/84)[K
remote: Compressing objects:  75% (63/84)[K
remote: Compressing objects:  76% (64/84)[K
remote: Compressing objects:  77% (65/84)[K
remote: Compressing objects:  78% (66/84)[K
remote: Compressing objects:  79% (67/84)[K
remote: Compressing objects:  80% (68/84)[K
remote: Compressing objects:  82% (69/84)[K
remote: Compressing objects:  83% (70/84)[K
remote: Compressing objects:  84% (71/84)[K
remote: Compressing objects:  85% (72/84)[K
remote: Compressing objects:  86% (73/84)[K
remote: Compressing objects:  88% (74/84)[K
remote: Compressing objects:  89% (75/84)[K
remote: Compressing objects:  90% (76/84)[K
remote: Compressing objects:  91% (77/84)[K
remote: Compressing objects:  92% (78/84)[K
remote: Compressing objects:  94% (79/84)[K
remote: Compressing objects:  95% (80/84)[K
remote: Compressing objects:  96% (81/84)[K
remote: Compressing objects:  97% (82/84)[K
remote: Compressing objects:  98% (83/84)[K
remote: Compressing objects: 100% (84/84)[K
remote: Compressing objects: 100% (84/84), done.[K
    Receiving objects:   0% (1/572)
Receiving objects:   1% (6/572)

.. parsed-literal::

    Receiving objects:   2% (12/572)
Receiving objects:   3% (18/572)
Receiving objects:   4% (23/572)
Receiving objects:   5% (29/572)
Receiving objects:   6% (35/572)
Receiving objects:   7% (41/572)
Receiving objects:   8% (46/572)

.. parsed-literal::

    Receiving objects:   9% (52/572)
Receiving objects:  10% (58/572)
Receiving objects:  11% (63/572)
Receiving objects:  12% (69/572)

.. parsed-literal::

    Receiving objects:  13% (75/572)
Receiving objects:  14% (81/572)
Receiving objects:  15% (86/572)
Receiving objects:  16% (92/572)
Receiving objects:  17% (98/572)
Receiving objects:  18% (103/572)
Receiving objects:  19% (109/572)
Receiving objects:  20% (115/572)
Receiving objects:  21% (121/572)
Receiving objects:  22% (126/572)
Receiving objects:  23% (132/572)
Receiving objects:  24% (138/572)
Receiving objects:  25% (143/572)
Receiving objects:  26% (149/572)
Receiving objects:  27% (155/572)
Receiving objects:  28% (161/572)
Receiving objects:  29% (166/572)
Receiving objects:  30% (172/572)
Receiving objects:  31% (178/572)
Receiving objects:  32% (184/572)
Receiving objects:  33% (189/572)
Receiving objects:  34% (195/572)

.. parsed-literal::

    Receiving objects:  35% (201/572)
Receiving objects:  36% (206/572)
Receiving objects:  37% (212/572)
Receiving objects:  38% (218/572)
Receiving objects:  39% (224/572)
Receiving objects:  40% (229/572)
Receiving objects:  41% (235/572)
Receiving objects:  42% (241/572)
Receiving objects:  43% (246/572)
Receiving objects:  44% (252/572)
Receiving objects:  45% (258/572)
Receiving objects:  46% (264/572)
Receiving objects:  47% (269/572)
Receiving objects:  48% (275/572)
Receiving objects:  49% (281/572)
Receiving objects:  50% (286/572)
Receiving objects:  51% (292/572)
Receiving objects:  52% (298/572)
Receiving objects:  53% (304/572)
Receiving objects:  54% (309/572)
Receiving objects:  55% (315/572)
Receiving objects:  56% (321/572)
Receiving objects:  57% (327/572)
Receiving objects:  58% (332/572)
Receiving objects:  59% (338/572)
Receiving objects:  60% (344/572)
Receiving objects:  61% (349/572)
Receiving objects:  62% (355/572)
Receiving objects:  63% (361/572)

.. parsed-literal::

    Receiving objects:  64% (367/572)
Receiving objects:  65% (372/572)
Receiving objects:  66% (378/572)
Receiving objects:  67% (384/572)
Receiving objects:  68% (389/572)
Receiving objects:  69% (395/572)
Receiving objects:  70% (401/572)
Receiving objects:  71% (407/572)
Receiving objects:  72% (412/572)
Receiving objects:  73% (418/572)
Receiving objects:  74% (424/572)
Receiving objects:  75% (429/572)
Receiving objects:  76% (435/572)
Receiving objects:  77% (441/572)
Receiving objects:  78% (447/572)
Receiving objects:  79% (452/572)
Receiving objects:  80% (458/572)
Receiving objects:  81% (464/572)
Receiving objects:  82% (470/572)
Receiving objects:  83% (475/572)
Receiving objects:  84% (481/572)
Receiving objects:  85% (487/572)
Receiving objects:  86% (492/572)
Receiving objects:  87% (498/572)
Receiving objects:  88% (504/572)
Receiving objects:  89% (510/572)
Receiving objects:  90% (515/572)
Receiving objects:  91% (521/572)
Receiving objects:  92% (527/572)
Receiving objects:  93% (532/572)
Receiving objects:  94% (538/572)
Receiving objects:  95% (544/572)
Receiving objects:  96% (550/572)
Receiving objects:  97% (555/572)
Receiving objects:  98% (561/572)
Receiving objects:  99% (567/572)
remote: Total 572 (delta 143), reused 118 (delta 107), pack-reused 380[K
    Receiving objects: 100% (572/572)
Receiving objects: 100% (572/572), 3.20 MiB | 13.83 MiB/s, done.
    Resolving deltas:   0% (0/204)
Resolving deltas:   2% (6/204)
Resolving deltas:   3% (7/204)
Resolving deltas:   4% (9/204)
Resolving deltas:   6% (13/204)
Resolving deltas:   7% (15/204)
Resolving deltas:   8% (17/204)
Resolving deltas:   9% (20/204)
Resolving deltas:  10% (21/204)
Resolving deltas:  11% (23/204)
Resolving deltas:  12% (25/204)
Resolving deltas:  13% (27/204)
Resolving deltas:  14% (30/204)
Resolving deltas:  15% (31/204)
Resolving deltas:  18% (37/204)
Resolving deltas:  19% (40/204)
Resolving deltas:  20% (41/204)
Resolving deltas:  21% (43/204)
Resolving deltas:  22% (45/204)
Resolving deltas:  25% (53/204)
Resolving deltas:  27% (57/204)
Resolving deltas:  40% (83/204)
Resolving deltas:  43% (88/204)
Resolving deltas:  44% (91/204)
Resolving deltas:  46% (94/204)
Resolving deltas:  65% (134/204)
Resolving deltas:  66% (136/204)
Resolving deltas:  67% (137/204)
Resolving deltas:  68% (140/204)
Resolving deltas:  82% (169/204)
Resolving deltas:  83% (171/204)
Resolving deltas:  84% (173/204)
Resolving deltas:  86% (177/204)
Resolving deltas:  87% (178/204)
Resolving deltas:  89% (183/204)
Resolving deltas:  93% (190/204)
Resolving deltas:  97% (199/204)
Resolving deltas:  98% (200/204)
Resolving deltas:  99% (203/204)
Resolving deltas: 100% (204/204)
Resolving deltas: 100% (204/204), done.


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9


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

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9/model/gelan-c.pt')



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

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/287-yolov9-optimization/yolov9/models/yolo.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
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


   val: Scanning coco/val2017... 985 images, 8 backgrounds, 0 corrupt:  20%|█▉        | 993/5000 00:00

.. parsed-literal::


   val: Scanning coco/val2017... 2174 images, 17 backgrounds, 0 corrupt:  44%|████▍     | 2191/5000 00:00

.. parsed-literal::


   val: Scanning coco/val2017... 3257 images, 31 backgrounds, 0 corrupt:  66%|██████▌   | 3288/5000 00:00

.. parsed-literal::


   val: Scanning coco/val2017... 4334 images, 41 backgrounds, 0 corrupt:  88%|████████▊ | 4375/5000 00:00

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

    2024-03-13 00:53:25.630509: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-13 00:53:25.674983: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-03-13 00:53:26.269021: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:84: RuntimeWarning: invalid value encountered in multiply
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
    [ INFO ] Read model took 27.34 ms
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
    [ INFO ] Reshape model took 8.24 ms
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

    [ INFO ] Compile model took 590.72 ms
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

    [ INFO ] First inference took 190.30 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            222 iterations
    [ INFO ] Duration:         15451.04 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        417.09 ms
    [ INFO ]    Average:       414.67 ms
    [ INFO ]    Min:           298.90 ms
    [ INFO ]    Max:           436.39 ms
    [ INFO ] Throughput:   14.37 FPS


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

    [ INFO ] Read model took 52.40 ms
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

    [ INFO ] Compile model took 1163.81 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU


.. parsed-literal::

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

    [ INFO ] First inference took 75.25 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            750 iterations
    [ INFO ] Duration:         15106.68 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        120.85 ms
    [ INFO ]    Average:       120.49 ms
    [ INFO ]    Min:           103.28 ms
    [ INFO ]    Max:           138.46 ms
    [ INFO ] Throughput:   49.65 FPS


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

