Convert and Optimize YOLOv7 with OpenVINO™
==========================================

The YOLOv7 algorithm is making big waves in the computer vision and
machine learning communities. It is a real-time object detection
algorithm that performs image recognition tasks by taking an image as
input and then predicting bounding boxes and class probabilities for
each object in the image.

YOLO stands for “You Only Look Once”, it is a popular family of
real-time object detection algorithms. The original YOLO object detector
was first released in 2016. Since then, different versions and variants
of YOLO have been proposed, each providing a significant increase in
performance and efficiency. YOLOv7 is next stage of evolution of YOLO
models family, which provides a greatly improved real-time object
detection accuracy without increasing the inference costs. More details
about its realization can be found in original model
`paper <https://arxiv.org/abs/2207.02696>`__ and
`repository <https://github.com/WongKinYiu/yolov7>`__

Real-time object detection is often used as a key component in computer
vision systems. Applications that use real-time object detection models
include video analytics, robotics, autonomous vehicles, multi-object
tracking and object counting, medical image analysis, and many others.

This tutorial demonstrates step-by-step instructions on how to run and
optimize PyTorch YOLO V7 with OpenVINO.

The tutorial consists of the following steps:

-  Prepare PyTorch model
-  Download and prepare dataset
-  Validate original model
-  Convert PyTorch model to ONNX
-  Convert ONNX model to OpenVINO IR
-  Validate converted model
-  Prepare and run optimization pipeline
-  Compare accuracy of the FP32 and quantized models.
-  Compare performance of the FP32 and quantized models.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Get Pytorch model <#get-pytorch-model>`__
-  `Prerequisites <#prerequisites>`__
-  `Check model inference <#check-model-inference>`__
-  `Export to ONNX <#export-to-onnx>`__
-  `Convert ONNX Model to OpenVINO Intermediate Representation
   (IR) <#convert-onnx-model-to-openvino-intermediate-representation-ir>`__
-  `Verify model inference <#verify-model-inference>`__

   -  `Preprocessing <#preprocessing>`__
   -  `Postprocessing <#postprocessing>`__
   -  `Select inference device <#select-inference-device>`__

-  `Verify model accuracy <#verify-model-accuracy>`__

   -  `Download dataset <#download-dataset>`__
   -  `Create dataloader <#create-dataloader>`__
   -  `Define validation function <#define-validation-function>`__

-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__
-  `Validate Quantized model
   inference <#validate-quantized-model-inference>`__
-  `Validate quantized model
   accuracy <#validate-quantized-model-accuracy>`__
-  `Compare Performance of the Original and Quantized
   Models <#compare-performance-of-the-original-and-quantized-models>`__

Get Pytorch model
-----------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv7 tiny model pre-trained on a COCO dataset, which is available
in this `repo <https://github.com/WongKinYiu/yolov7>`__. Typical steps
to obtain pre-trained model:

1. Create instance of model class.
2. Load checkpoint state dict, which contains pre-trained model weights.
3. Turn model to evaluation for switching some operations to inference
   mode.

In this case, the model creators provide a tool that enables converting
the YOLOv7 model to ONNX, so we do not need to do these steps manually.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "openvino>=2023.1.0" "nncf>=2.5.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path
    sys.path.append("../utils")
    from notebook_utils import download_file

.. code:: ipython3

    # Clone YOLOv7 repo
    if not Path('yolov7').exists():
        !git clone https://github.com/WongKinYiu/yolov7
    %cd yolov7


.. parsed-literal::

    Cloning into 'yolov7'...


.. parsed-literal::

    remote: Enumerating objects: 1197, done.[K
    Receiving objects:   0% (1/1197)
Receiving objects:   1% (12/1197)
Receiving objects:   2% (24/1197)
Receiving objects:   3% (36/1197)
Receiving objects:   4% (48/1197)
Receiving objects:   5% (60/1197)
Receiving objects:   6% (72/1197)
Receiving objects:   7% (84/1197)
Receiving objects:   8% (96/1197)

.. parsed-literal::

    Receiving objects:   9% (108/1197)
Receiving objects:  10% (120/1197)
Receiving objects:  11% (132/1197)
Receiving objects:  12% (144/1197)
Receiving objects:  13% (156/1197)
Receiving objects:  14% (168/1197)
Receiving objects:  15% (180/1197)
Receiving objects:  16% (192/1197)
Receiving objects:  17% (204/1197)
Receiving objects:  18% (216/1197)
Receiving objects:  19% (228/1197)
Receiving objects:  20% (240/1197)
Receiving objects:  21% (252/1197)
Receiving objects:  22% (264/1197)
Receiving objects:  23% (276/1197)
Receiving objects:  24% (288/1197)
Receiving objects:  25% (300/1197)

.. parsed-literal::

    Receiving objects:  26% (312/1197)

.. parsed-literal::

    Receiving objects:  26% (322/1197), 3.61 MiB | 3.50 MiB/s

.. parsed-literal::

    Receiving objects:  27% (324/1197), 3.61 MiB | 3.50 MiB/s

.. parsed-literal::

    Receiving objects:  27% (334/1197), 7.27 MiB | 3.52 MiB/s

.. parsed-literal::

    Receiving objects:  27% (334/1197), 10.93 MiB | 3.53 MiB/s

.. parsed-literal::

    Receiving objects:  28% (336/1197), 10.93 MiB | 3.53 MiB/s

.. parsed-literal::

    Receiving objects:  28% (338/1197), 14.60 MiB | 3.53 MiB/s

.. parsed-literal::

    Receiving objects:  28% (339/1197), 18.26 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  28% (339/1197), 21.92 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  28% (340/1197), 23.75 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  28% (343/1197), 29.25 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  28% (345/1197), 31.09 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  28% (347/1197), 34.76 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  29% (348/1197), 38.43 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  29% (350/1197), 38.43 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  30% (360/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  31% (372/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  32% (384/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  33% (396/1197), 38.43 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  34% (407/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  35% (419/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  36% (431/1197), 38.43 MiB | 3.55 MiB/s
Receiving objects:  37% (443/1197), 38.43 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  38% (455/1197), 40.26 MiB | 3.55 MiB/s
Receiving objects:  39% (467/1197), 40.26 MiB | 3.55 MiB/s
Receiving objects:  40% (479/1197), 40.26 MiB | 3.55 MiB/s
Receiving objects:  41% (491/1197), 40.26 MiB | 3.55 MiB/s
Receiving objects:  42% (503/1197), 40.26 MiB | 3.55 MiB/s
Receiving objects:  43% (515/1197), 40.26 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  44% (527/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  45% (539/1197), 42.10 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  46% (551/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  47% (563/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  48% (575/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  49% (587/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  50% (599/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  51% (611/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  52% (623/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  53% (635/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  54% (647/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  55% (659/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  56% (671/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  57% (683/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  58% (695/1197), 42.10 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  58% (700/1197), 42.10 MiB | 3.55 MiB/s
Receiving objects:  59% (707/1197), 42.10 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  59% (715/1197), 47.60 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  59% (715/1197), 51.26 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  59% (715/1197), 54.93 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  59% (715/1197), 56.75 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  60% (719/1197), 56.75 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  61% (731/1197), 56.75 MiB | 3.55 MiB/s
Receiving objects:  62% (743/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  63% (755/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  64% (767/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  65% (779/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  66% (791/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  67% (802/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  68% (814/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  69% (826/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  70% (838/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  71% (850/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  72% (862/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  73% (874/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  74% (886/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  75% (898/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  76% (910/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  77% (922/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  78% (934/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  79% (946/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  80% (958/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  81% (970/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  82% (982/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  83% (994/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  84% (1006/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  85% (1018/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  86% (1030/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  87% (1042/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  88% (1054/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  89% (1066/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  90% (1078/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  91% (1090/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  92% (1102/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  93% (1114/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  94% (1126/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  95% (1138/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  96% (1150/1197), 58.59 MiB | 3.55 MiB/s
Receiving objects:  97% (1162/1197), 58.59 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  97% (1172/1197), 60.42 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  97% (1172/1197), 64.08 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  97% (1172/1197), 67.75 MiB | 3.55 MiB/s

.. parsed-literal::

    Receiving objects:  97% (1172/1197), 71.41 MiB | 3.55 MiB/s

.. parsed-literal::

    remote: Total 1197 (delta 0), reused 0 (delta 0), pack-reused 1197[K
    Receiving objects:  98% (1174/1197), 73.25 MiB | 3.55 MiB/s
Receiving objects:  99% (1186/1197), 73.25 MiB | 3.55 MiB/s
Receiving objects: 100% (1197/1197), 73.25 MiB | 3.55 MiB/s
Receiving objects: 100% (1197/1197), 74.23 MiB | 3.54 MiB/s, done.
    Resolving deltas:   0% (0/520)
Resolving deltas:   1% (9/520)
Resolving deltas:   2% (15/520)

.. parsed-literal::

    Resolving deltas:   3% (17/520)
Resolving deltas:   4% (21/520)
Resolving deltas:   5% (26/520)
Resolving deltas:   6% (32/520)
Resolving deltas:   8% (42/520)
Resolving deltas:   9% (50/520)
Resolving deltas:  10% (52/520)
Resolving deltas:  11% (58/520)
Resolving deltas:  13% (68/520)
Resolving deltas:  14% (73/520)
Resolving deltas:  16% (87/520)
Resolving deltas:  17% (91/520)
Resolving deltas:  21% (113/520)
Resolving deltas:  22% (116/520)
Resolving deltas:  23% (123/520)
Resolving deltas:  26% (140/520)
Resolving deltas:  32% (171/520)
Resolving deltas:  33% (172/520)
Resolving deltas:  34% (181/520)
Resolving deltas:  35% (182/520)
Resolving deltas:  36% (188/520)
Resolving deltas:  38% (202/520)
Resolving deltas:  39% (204/520)
Resolving deltas:  40% (211/520)
Resolving deltas:  48% (252/520)
Resolving deltas:  49% (255/520)
Resolving deltas:  51% (267/520)
Resolving deltas:  52% (271/520)
Resolving deltas:  53% (279/520)
Resolving deltas:  57% (300/520)
Resolving deltas:  66% (345/520)
Resolving deltas:  67% (349/520)
Resolving deltas:  68% (354/520)
Resolving deltas:  69% (361/520)
Resolving deltas:  70% (365/520)
Resolving deltas:  71% (371/520)
Resolving deltas:  72% (375/520)
Resolving deltas:  73% (380/520)
Resolving deltas:  74% (385/520)
Resolving deltas:  75% (394/520)
Resolving deltas:  76% (396/520)
Resolving deltas:  77% (401/520)
Resolving deltas:  78% (406/520)
Resolving deltas:  80% (416/520)
Resolving deltas:  81% (424/520)
Resolving deltas:  83% (436/520)
Resolving deltas:  84% (437/520)
Resolving deltas:  85% (442/520)
Resolving deltas:  86% (450/520)
Resolving deltas:  87% (454/520)
Resolving deltas:  89% (463/520)
Resolving deltas:  90% (469/520)
Resolving deltas:  91% (477/520)
Resolving deltas:  93% (488/520)
Resolving deltas:  94% (489/520)
Resolving deltas:  95% (494/520)
Resolving deltas:  96% (502/520)
Resolving deltas:  97% (507/520)
Resolving deltas:  98% (514/520)
Resolving deltas:  99% (515/520)
Resolving deltas: 100% (520/520)
Resolving deltas: 100% (520/520), done.


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/226-yolov7-optimization/yolov7


.. code:: ipython3

    # Download pre-trained model weights
    MODEL_LINK = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    DATA_DIR = Path("data/")
    MODEL_DIR = Path("model/")
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)



.. parsed-literal::

    model/yolov7-tiny.pt:   0%|          | 0.00/12.1M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/226-yolov7-optimization/yolov7/model/yolov7-tiny.pt')



Check model inference
---------------------



``detect.py`` script run pytorch model inference and save image as
result,

.. code:: ipython3

    !python -W ignore detect.py --weights model/yolov7-tiny.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg


.. parsed-literal::

    Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', no_trace=False, nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='inference/images/horses.jpg', update=False, view_img=False, weights=['model/yolov7-tiny.pt'])
    YOLOR 🚀 v0.1-128-ga207844 torch 1.13.1+cpu CPU



.. parsed-literal::

    Fusing layers...


.. parsed-literal::

    Model Summary: 200 layers, 6219709 parameters, 229245 gradients
     Convert model to Traced-model...


.. parsed-literal::

     traced_script_module saved!
     model is traced!



.. parsed-literal::

    5 horses, Done. (70.4ms) Inference, (0.8ms) NMS


.. parsed-literal::

     The image with the result is saved in: runs/detect/exp/horses.jpg
    Done. (0.084s)


.. code:: ipython3

    from PIL import Image
    # visualize prediction result
    Image.open('runs/detect/exp/horses.jpg')




.. image:: 226-yolov7-optimization-with-output_files/226-yolov7-optimization-with-output_10_0.png



Export to ONNX
--------------



To export an ONNX format of the model, we will use ``export.py`` script.
Let us check its arguments.

.. code:: ipython3

    !python export.py --help


.. parsed-literal::

    Import onnx_graphsurgeon failure: No module named 'onnx_graphsurgeon'
    usage: export.py [-h] [--weights WEIGHTS] [--img-size IMG_SIZE [IMG_SIZE ...]]
                     [--batch-size BATCH_SIZE] [--dynamic] [--dynamic-batch]
                     [--grid] [--end2end] [--max-wh MAX_WH] [--topk-all TOPK_ALL]
                     [--iou-thres IOU_THRES] [--conf-thres CONF_THRES]
                     [--device DEVICE] [--simplify] [--include-nms] [--fp16]
                     [--int8]

    optional arguments:
      -h, --help            show this help message and exit
      --weights WEIGHTS     weights path
      --img-size IMG_SIZE [IMG_SIZE ...]
                            image size
      --batch-size BATCH_SIZE
                            batch size
      --dynamic             dynamic ONNX axes
      --dynamic-batch       dynamic batch onnx for tensorrt and onnx-runtime
      --grid                export Detect() layer grid
      --end2end             export end2end onnx
      --max-wh MAX_WH       None for tensorrt nms, int value for onnx-runtime nms
      --topk-all TOPK_ALL   topk objects for every images
      --iou-thres IOU_THRES
                            iou threshold for NMS
      --conf-thres CONF_THRES
                            conf threshold for NMS
      --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
      --simplify            simplify onnx model
      --include-nms         export end2end onnx
      --fp16                CoreML FP16 half-precision export
      --int8                CoreML INT8 quantization


The most important parameters:

-  ``--weights`` - path to model weights checkpoint
-  ``--img-size`` - size of input image for onnx tracing

When exporting the ONNX model from PyTorch, there is an opportunity to
setup configurable parameters for including post-processing results in
model:

-  ``--end2end`` - export full model to onnx including post-processing
-  ``--grid`` - export Detect layer as part of model
-  ``--topk-all`` - top k elements for all images
-  ``--iou-thres`` - intersection over union threshold for NMS
-  ``--conf-thres`` - minimal confidence threshold
-  ``--max-wh`` - max bounding box width and height for NMS

Including whole post-processing to model can help to achieve more
performant results, but in the same time it makes the model less
flexible and does not guarantee full accuracy reproducibility. It is the
reason why we will add only ``--grid`` parameter to preserve original
pytorch model result format. If you want to understand how to work with
an end2end ONNX model, you can check this
`notebook <https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb>`__.

.. code:: ipython3

    !python -W ignore export.py --weights model/yolov7-tiny.pt --grid


.. parsed-literal::

    Import onnx_graphsurgeon failure: No module named 'onnx_graphsurgeon'
    Namespace(batch_size=1, conf_thres=0.25, device='cpu', dynamic=False, dynamic_batch=False, end2end=False, fp16=False, grid=True, img_size=[640, 640], include_nms=False, int8=False, iou_thres=0.45, max_wh=None, simplify=False, topk_all=100, weights='model/yolov7-tiny.pt')


.. parsed-literal::

    YOLOR 🚀 v0.1-128-ga207844 torch 1.13.1+cpu CPU

    Fusing layers...


.. parsed-literal::

    Model Summary: 200 layers, 6219709 parameters, 6219709 gradients


.. parsed-literal::


    Starting TorchScript export with torch 1.13.1+cpu...


.. parsed-literal::

    TorchScript export success, saved as model/yolov7-tiny.torchscript.pt
    CoreML export failure: No module named 'coremltools'

    Starting TorchScript-Lite export with torch 1.13.1+cpu...


.. parsed-literal::

    TorchScript-Lite export success, saved as model/yolov7-tiny.torchscript.ptl

    Starting ONNX export with onnx 1.15.0...


.. parsed-literal::

    ONNX export success, saved as model/yolov7-tiny.onnx

    Export complete (2.36s). Visualize with https://github.com/lutzroeder/netron.


Convert ONNX Model to OpenVINO Intermediate Representation (IR)
---------------------------------------------------------------

While ONNX models are directly
supported by OpenVINO runtime, it can be useful to convert them to IR
format to take the advantage of OpenVINO model conversion API features.
The ``ov.convert_model`` python function of `model conversion
API <https://docs.openvino.ai/2023.3/openvino_docs_model_processing_introduction.html>`__
can be used for converting the model. The function returns instance of
OpenVINO Model class, which is ready to use in Python interface.
However, it can also be save on device in OpenVINO IR format using
``ov.save_model`` for future execution.

.. code:: ipython3

    import openvino as ov

    model = ov.convert_model('model/yolov7-tiny.onnx')
    # serialize model for saving IR
    ov.save_model(model, 'model/yolov7-tiny.xml')

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
defined in yolov7 repository.

To keep specific shape, preprocessing automatically enables padding.

.. code:: ipython3

    import numpy as np
    import torch
    from PIL import Image
    from utils.datasets import letterbox
    from utils.plots import plot_one_box


    def preprocess_image(img0: np.ndarray):
        """
        Preprocess image according to YOLOv7 input requirements.
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
        Converts preprocessed image to tensor format according to YOLOv7 input requirements.
        Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

        Parameters:
          image (np.ndarray): image for conversion to tensor
        Returns:
          input_tensor (torch.Tensor): float tensor ready to use for YOLOv7 inference
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp16/32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor


    # label names for visualization
    DEFAULT_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                     'hair drier', 'toothbrush']

    # obtain class names from model checkpoint
    state_dict = torch.load("model/yolov7-tiny.pt", map_location="cpu")
    if hasattr(state_dict["model"], "module"):
        NAMES = getattr(state_dict["model"].module, "names", DEFAULT_NAMES)
    else:
        NAMES = getattr(state_dict["model"], "names", DEFAULT_NAMES)

    del state_dict

    # colors for visualization
    COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
              for i, name in enumerate(NAMES)}

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

    from typing import List, Tuple, Dict
    from utils.general import scale_coords, non_max_suppression


    def detect(model: ov.Model, image_path: Path, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):
        """
        OpenVINO YOLOv7 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
        Parameters:
            model (Model): OpenVINO compiled model.
            image_path (Path): input image path.
            conf_thres (float, *optional*, 0.25): minimal accpeted confidence for object filtering
            iou_thres (float, *optional*, 0.45): minimal overlap score for remloving objects duplicates in NMS
            classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        Returns:
           pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
           orig_img (np.ndarray): image before preprocessing, can be used for results visualization
           inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
        """
        output_blob = model.output(0)
        img = np.array(Image.open(image_path))
        preprocessed_img, orig_img = preprocess_image(img)
        input_tensor = prepare_input_tensor(preprocessed_img)
        predictions = torch.from_numpy(model(input_tensor)[output_blob])
        pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        return pred, orig_img, input_tensor.shape


    def draw_boxes(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str], colors: Dict[str, int]):
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
        # Rescale boxes from input size to original image size
        predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(predictions):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)
        return image

.. code:: ipython3

    core = ov.Core()
    # read converted model
    model = core.read_model('model/yolov7-tiny.xml')

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

    # load model on CPU device
    compiled_model = core.compile_model(model, device.value)

.. code:: ipython3

    boxes, image, input_shape = detect(compiled_model, 'inference/images/horses.jpg')
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES, COLORS)
    # visualize results
    Image.fromarray(image_with_boxes)




.. image:: 226-yolov7-optimization-with-output_files/226-yolov7-optimization-with-output_27_0.png



Verify model accuracy
---------------------



Download dataset
~~~~~~~~~~~~~~~~



YOLOv7 tiny is pre-trained on the COCO dataset, so in order to evaluate
the model accuracy, we need to download it. According to the
instructions provided in the YOLOv7 repo, we also need to download
annotations in the format used by the author of the model, for use with
the original model evaluation scripts.

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


Create dataloader
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from collections import namedtuple
    import yaml
    from utils.datasets import create_dataloader
    from utils.general import check_dataset, box_iou, xywh2xyxy, colorstr

    # read dataset config
    DATA_CONFIG = 'data/coco.yaml'
    with open(DATA_CONFIG) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Dataloader
    TASK = 'val'  # path to train/val/test images
    Option = namedtuple('Options', ['single_cls'])  # imitation of commandline provided options for single class evaluation
    opt = Option(False)
    dataloader = create_dataloader(
        data[TASK], 640, 1, 32, opt, pad=0.5,
        prefix=colorstr(f'{TASK}: ')
    )[0]


.. parsed-literal::


    Scanning images:   0%|          | 0/5000 [00:00<?, ?it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 292 found, 1 missing, 0 empty, 0 corrupted:   6%|▌         | 293/5000 [00:00<00:01, 2923.29it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 580 found, 6 missing, 0 empty, 0 corrupted:  12%|█▏        | 586/5000 [00:00<00:01, 2924.54it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 871 found, 8 missing, 0 empty, 0 corrupted:  18%|█▊        | 879/5000 [00:00<00:01, 2922.27it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 1170 found, 10 missing, 0 empty, 0 corrupted:  24%|██▎       | 1180/5000 [00:00<00:01, 2955.18it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 1466 found, 10 missing, 0 empty, 0 corrupted:  30%|██▉       | 1476/5000 [00:00<00:01, 2934.54it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 1756 found, 14 missing, 0 empty, 0 corrupted:  35%|███▌      | 1770/5000 [00:00<00:01, 2776.60it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 2049 found, 16 missing, 0 empty, 0 corrupted:  41%|████▏     | 2065/5000 [00:00<00:01, 2829.21it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 2351 found, 22 missing, 0 empty, 0 corrupted:  47%|████▋     | 2373/5000 [00:00<00:00, 2905.88it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 2641 found, 24 missing, 0 empty, 0 corrupted:  53%|█████▎    | 2665/5000 [00:00<00:00, 2898.47it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 2938 found, 28 missing, 0 empty, 0 corrupted:  59%|█████▉    | 2966/5000 [00:01<00:00, 2930.42it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 3236 found, 31 missing, 0 empty, 0 corrupted:  65%|██████▌   | 3267/5000 [00:01<00:00, 2952.50it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 3529 found, 34 missing, 0 empty, 0 corrupted:  71%|███████▏  | 3563/5000 [00:01<00:00, 2951.52it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 3824 found, 35 missing, 0 empty, 0 corrupted:  77%|███████▋  | 3859/5000 [00:01<00:00, 2947.45it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 4117 found, 39 missing, 0 empty, 0 corrupted:  83%|████████▎ | 4156/5000 [00:01<00:00, 2953.34it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 4413 found, 42 missing, 0 empty, 0 corrupted:  89%|████████▉ | 4455/5000 [00:01<00:00, 2962.81it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 4709 found, 46 missing, 0 empty, 0 corrupted:  95%|█████████▌| 4755/5000 [00:01<00:00, 2971.05it/s]

.. parsed-literal::


    val: Scanning 'coco/val2017' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|██████████| 5000/5000 [00:01<00:00, 2932.19it/s]



Define validation function
~~~~~~~~~~~~~~~~~~~~~~~~~~



We will reuse validation metrics provided in the YOLOv7 repo with a
modification for this case (removing extra steps). The original model
evaluation procedure can be found in this
`file <https://github.com/WongKinYiu/yolov7/blob/main/test.py>`__

.. code:: ipython3

    import numpy as np
    from tqdm.notebook import tqdm
    from utils.metrics import ap_per_class
    from openvino.runtime import Tensor


    def test(data,
             model: ov.Model,
             dataloader: torch.utils.data.DataLoader,
             conf_thres: float = 0.001,
             iou_thres: float = 0.65,  # for NMS
             single_cls: bool = False,
             v5_metric: bool = False,
             names: List[str] = None,
             num_samples: int = None
            ):
        """
        YOLOv7 accuracy evaluation. Processes validation dataset and compites metrics.

        Parameters:
            model (ov.Model): OpenVINO compiled model.
            dataloader (torch.utils.DataLoader): validation dataset.
            conf_thres (float, *optional*, 0.001): minimal confidence threshold for keeping detections
            iou_thres (float, *optional*, 0.65): IOU threshold for NMS
            single_cls (bool, *optional*, False): class agnostic evaluation
            v5_metric (bool, *optional*, False): use YOLOv5 evaluation approach for metrics calculation
            names (List[str], *optional*, None): names for each class in dataset
            num_samples (int, *optional*, None): number samples for testing
        Returns:
            mp (float): mean precision
            mr (float): mean recall
            map50 (float): mean average precision at 0.5 IOU threshold
            map (float): mean average precision at 0.5:0.95 IOU thresholds
            maps (Dict(int, float): average precision per class
            seen (int): number of evaluated images
            labels (int): number of labels
        """

        model_output = model.output(0)
        check_dataset(data)  # check
        nc = 1 if single_cls else int(data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        if v5_metric:
            print("Testing with YOLOv5 AP metric...")

        seen = 0
        p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        for sample_id, (img, targets, _, shapes) in enumerate(tqdm(dataloader)):
            if num_samples is not None and sample_id == num_samples:
                break
            img = prepare_input_tensor(img.numpy())
            targets = targets
            height, width = img.shape[2:]

            with torch.no_grad():
                # Run model
                out = torch.from_numpy(model(Tensor(img))[model_output])  # inference output
                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels

                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True)
            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cpu')
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=v5_metric, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return mp, mr, map50, map, maps, seen, nt.sum()

Validation function reports following list of accuracy metrics:

-  ``Precision`` is the degree of exactness of the model in identifying
   only relevant objects.
-  ``Recall`` measures the ability of the model to detect all ground
   truths objects.
-  ``mAP@t`` - mean average precision, represented as area under the
   Precision-Recall curve aggregated over all classes in the dataset,
   where ``t`` is Intersection Over Union (IOU) threshold, degree of
   overlapping between ground truth and predicted objects. Therefore,
   ``mAP@.5`` indicates that mean average precision calculated at 0.5
   IOU threshold, ``mAP@.5:.95`` - calculated on range IOU thresholds
   from 0.5 to 0.95 with step 0.05.

.. code:: ipython3

    mp, mr, map50, map, maps, num_images, labels = test(data=data, model=compiled_model, dataloader=dataloader, names=NAMES)
    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', num_images, labels, mp, mr, map50, map))



.. parsed-literal::

      0%|          | 0/5000 [00:00<?, ?it/s]


.. parsed-literal::

                   Class      Images      Labels   Precision      Recall      mAP@.5  mAP@.5:.95
                     all        5000       36335       0.651       0.507       0.544       0.359


Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
YOLOv7.

   **NOTE**: NNCF Post-training Quantization is available as a preview
   feature in OpenVINO 2022.3 release. Fully functional support will be
   provided in the next releases.

The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the
   ``openvino.runtime.serialize`` function.

Reuse validation dataloader in accuracy testing for quantization. For
that, it should be wrapped into the ``nncf.Dataset`` object and define
transformation function for getting only input tensors.

.. code:: ipython3

    import nncf  # noqa: F811


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


The ``nncf.quantize`` function provides interface for model
quantization. It requires instance of OpenVINO Model and quantization
dataset. Optionally, some additional parameters for configuration
quantization process (number of samples for quantization, preset,
ignored scope etc.) can be provided. YOLOv7 model contains non-ReLU
activation functions, which require asymmetric quantization of
activations. To achieve better result, we will use ``mixed``
quantization preset. It provides symmetric quantization of weights and
asymmetric quantization of activations.

.. code:: ipython3

    quantized_model = nncf.quantize(model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

    ov.save_model(quantized_model, 'model/yolov7-tiny_int8.xml')


.. parsed-literal::

    2024-02-09 23:58:09.840364: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-09 23:58:09.872358: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-09 23:58:10.416753: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()








.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/nncf/experimental/tensor/tensor.py:84: RuntimeWarning: invalid value encountered in multiply
      return Tensor(self.data * unwrap_tensor_data(other))








.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Validate Quantized model inference
----------------------------------



.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    int8_compiled_model = core.compile_model(quantized_model, device.value)
    boxes, image, input_shape = detect(int8_compiled_model, 'inference/images/horses.jpg')
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES, COLORS)
    Image.fromarray(image_with_boxes)




.. image:: 226-yolov7-optimization-with-output_files/226-yolov7-optimization-with-output_44_0.png



Validate quantized model accuracy
---------------------------------



.. code:: ipython3

    int8_result = test(data=data, model=int8_compiled_model, dataloader=dataloader, names=NAMES)



.. parsed-literal::

      0%|          | 0/5000 [00:00<?, ?it/s]


.. code:: ipython3

    mp, mr, map50, map, maps, num_images, labels = int8_result
    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', num_images, labels, mp, mr, map50, map))


.. parsed-literal::

                   Class      Images      Labels   Precision      Recall      mAP@.5  mAP@.5:.95
                     all        5000       36335       0.634       0.509        0.54       0.353


As we can see, model accuracy slightly changed after quantization.
However, if we look at the output image, these changes are not
significant.

Compare Performance of the Original and Quantized Models
--------------------------------------------------------



Finally, use the OpenVINO `Benchmark
Tool <https://docs.openvino.ai/2023.3/openvino_sample_benchmark_tool.html>`__
to measure the inference performance of the ``FP32`` and ``INT8``
models.

   **NOTE**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Inference FP32 model (OpenVINO IR)
    !benchmark_app -m model/yolov7-tiny.xml -d $device.value -api async


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
    [ INFO ] Read model took 13.39 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 272.70 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
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
    [ INFO ]     NETWORK_NAME: torch_jit
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

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 43.02 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            11586 iterations
    [ INFO ] Duration:         120047.32 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        61.86 ms
    [ INFO ]    Average:       62.03 ms
    [ INFO ]    Min:           44.24 ms
    [ INFO ]    Max:           86.89 ms
    [ INFO ] Throughput:   96.51 FPS


.. code:: ipython3

    # Inference INT8 model (OpenVINO IR)
    !benchmark_app -m model/yolov7-tiny_int8.xml -d $device.value -api async


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
    [ INFO ] Read model took 22.28 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output (node: output) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 489.58 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
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
    [ INFO ]     NETWORK_NAME: torch_jit
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

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 25.89 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            33210 iterations
    [ INFO ] Duration:         120019.44 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        21.49 ms
    [ INFO ]    Average:       21.57 ms
    [ INFO ]    Min:           15.74 ms
    [ INFO ]    Max:           43.27 ms
    [ INFO ] Throughput:   276.71 FPS

